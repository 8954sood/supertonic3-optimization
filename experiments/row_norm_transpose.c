/* Experimental tile-local exact data transpose; Welford and FMA order unchanged. */
/* Independent row-parallel Welford kernel. No reassociation within a row.
 * FMA sites match the target ORT kernel; the Python loader verifies bitidentity. */
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <pthread.h>
#include "onnxruntime_c_api.h"

#ifndef ROW_TILE
#define ROW_TILE 4
#endif
#if ROW_TILE != 4 && ROW_TILE != 8 && ROW_TILE != 16 && ROW_TILE != 32
#error "ROW_TILE must be 4, 8, 16 or 32"
#endif

static void row_norm_four(const float* x, const float* scale, const float* bias,
              float* y, int64_t rows, int64_t width, float epsilon) {
    int64_t row = 0;
    for (; row + 4 <= rows; row += 4) {
        float32x4_t mean = vdupq_n_f32(0), m2 = mean;
        for (int64_t h = 0; h < width; ++h) {
            float32x4_t value = {x[row*width+h], x[(row+1)*width+h],
                                x[(row+2)*width+h], x[(row+3)*width+h]};
            float32x4_t delta = vsubq_f32(value, mean);
            mean = vaddq_f32(mean, vdivq_f32(delta, vdupq_n_f32((float)(h+1))));
            float32x4_t delta2 = vsubq_f32(value, mean);
#ifdef FUSED_M2
            m2 = vfmaq_f32(m2, delta, delta2);
#else
            m2 = vaddq_f32(m2, vmulq_f32(delta, delta2));
#endif
        }
        float32x4_t denominator = vsqrtq_f32(vaddq_f32(
            vdivq_f32(m2, vdupq_n_f32((float)width)), vdupq_n_f32(epsilon)));
        float means[4], deviations[4];
        vst1q_f32(means, mean); vst1q_f32(deviations, denominator);
        for (int r = 0; r < 4; ++r) {
            int64_t h = 0;
            const float* input = x + (row+r)*width;
            float* output = y + (row+r)*width;
            for (; h + 4 <= width; h += 4) {
                float32x4_t v = vdivq_f32(vsubq_f32(vld1q_f32(input+h),
                    vdupq_n_f32(means[r])), vdupq_n_f32(deviations[r]));
#ifdef FUSED_OUTPUT
                v = vfmaq_f32(vld1q_f32(bias+h), v, vld1q_f32(scale+h));
#else
                v = vaddq_f32(vmulq_f32(v, vld1q_f32(scale+h)), vld1q_f32(bias+h));
#endif
                vst1q_f32(output+h, v);
            }
            for (; h < width; ++h) output[h] = (input[h]-means[r])/deviations[r]*scale[h]+bias[h];
        }
    }
    for (; row < rows; ++row) {
        float mean = 0, m2 = 0;
        for (int64_t h = 0; h < width; ++h) {
            float value=x[row*width+h], delta=value-mean;
            mean += delta / (float)(h+1);
#ifdef FUSED_M2
            m2 = fmaf(delta, value-mean, m2);
#else
            m2 += delta * (value-mean);
#endif
        }
        float denominator=sqrtf(m2/(float)width+epsilon);
        for (int64_t h=0; h<width; ++h) {
            float normalized=(x[row*width+h]-mean)/denominator;
#ifdef FUSED_OUTPUT
            y[row*width+h]=fmaf(normalized,scale[h],bias[h]);
#else
            y[row*width+h]=normalized*scale[h]+bias[h];
#endif
        }
    }
}

#if ROW_TILE > 4
static __attribute__((always_inline)) inline int64_t row_norm_tile(
    const float* x,const float* scale,const float* bias,float* y,
    int64_t rows,int64_t width,float epsilon,int tile) {
    float transposed[ROW_TILE*512];
    const int packed_width=width<=512 && width%4==0;
    int64_t row=0;
    for(;row+tile<=rows;row+=tile) {
        if(packed_width) {
            for(int group=0;group<tile/4;++group)for(int64_t h=0;h<width;h+=4) {
                int64_t r=row+4*group;
                float32x4_t a=vld1q_f32(x+r*width+h),b=vld1q_f32(x+(r+1)*width+h);
                float32x4_t c=vld1q_f32(x+(r+2)*width+h),d=vld1q_f32(x+(r+3)*width+h);
                float32x4_t ab0=vtrn1q_f32(a,b),ab1=vtrn2q_f32(a,b);
                float32x4_t cd0=vtrn1q_f32(c,d),cd1=vtrn2q_f32(c,d);
                vst1q_f32(transposed+(h+0)*tile+4*group,vcombine_f32(vget_low_f32(ab0),vget_low_f32(cd0)));
                vst1q_f32(transposed+(h+1)*tile+4*group,vcombine_f32(vget_low_f32(ab1),vget_low_f32(cd1)));
                vst1q_f32(transposed+(h+2)*tile+4*group,vcombine_f32(vget_high_f32(ab0),vget_high_f32(cd0)));
                vst1q_f32(transposed+(h+3)*tile+4*group,vcombine_f32(vget_high_f32(ab1),vget_high_f32(cd1)));
            }
        }
        float32x4_t means[tile/4],m2[tile/4];
        for(int group=0;group<tile/4;++group) {
            means[group]=vdupq_n_f32(0);m2[group]=vdupq_n_f32(0);
        }
        for(int64_t h=0;h<width;++h) {
            float32x4_t count=vdupq_n_f32((float)(h+1));
#pragma clang loop unroll(full)
            for(int group=0;group<tile/4;++group) {
                int64_t r=row+4*group;
                float32x4_t value;
                if(packed_width)value=vld1q_f32(transposed+h*tile+4*group);
                else value=(float32x4_t){x[r*width+h],x[(r+1)*width+h],
                                       x[(r+2)*width+h],x[(r+3)*width+h]};
                float32x4_t delta=vsubq_f32(value,means[group]);
                means[group]=vaddq_f32(means[group],vdivq_f32(delta,count));
                float32x4_t delta2=vsubq_f32(value,means[group]);
#ifdef FUSED_M2
                m2[group]=vfmaq_f32(m2[group],delta,delta2);
#else
                m2[group]=vaddq_f32(m2[group],vmulq_f32(delta,delta2));
#endif
            }
        }
        float mean_values[tile],deviations[tile];
        for(int group=0;group<tile/4;++group) {
            vst1q_f32(mean_values+4*group,means[group]);
            vst1q_f32(deviations+4*group,vsqrtq_f32(vaddq_f32(
                vdivq_f32(m2[group],vdupq_n_f32((float)width)),vdupq_n_f32(epsilon))));
        }
        for(int r=0;r<tile;++r) {
            const float* input=x+(row+r)*width;float* output=y+(row+r)*width;
            int64_t h=0;
            for(;h+4<=width;h+=4) {
                float32x4_t value=vdivq_f32(vsubq_f32(vld1q_f32(input+h),
                    vdupq_n_f32(mean_values[r])),vdupq_n_f32(deviations[r]));
#ifdef FUSED_OUTPUT
                value=vfmaq_f32(vld1q_f32(bias+h),value,vld1q_f32(scale+h));
#else
                value=vaddq_f32(vmulq_f32(value,vld1q_f32(scale+h)),vld1q_f32(bias+h));
#endif
                vst1q_f32(output+h,value);
            }
            for(;h<width;++h) {
                float value=(input[h]-mean_values[r])/deviations[r];
#ifdef FUSED_OUTPUT
                output[h]=fmaf(value,scale[h],bias[h]);
#else
                output[h]=value*scale[h]+bias[h];
#endif
            }
        }
    }
    return row;
}
#endif

__attribute__((visibility("default")))
void row_norm(const float* x,const float* scale,const float* bias,
              float* y,int64_t rows,int64_t width,float epsilon) {
    int64_t done=0;
#if ROW_TILE > 4
    done=row_norm_tile(x,scale,bias,y,rows,width,epsilon,ROW_TILE);
#if ROW_TILE > 16
    done+=row_norm_tile(x+done*width,scale,bias,y+done*width,rows-done,width,epsilon,16);
#endif
#if ROW_TILE > 8
    done+=row_norm_tile(x+done*width,scale,bias,y+done*width,rows-done,width,epsilon,8);
#endif
#endif
    row_norm_four(x+done*width,scale,bias,y+done*width,rows-done,width,epsilon);
}

/* ORT adapter. Only the last-axis FP32, one-dimensional affine case is accepted. */
typedef struct { const OrtApi* api; float epsilon; } RowKernel;
static OrtCustomOpDomain* row_domain = NULL;
static pthread_mutex_t domain_mutex = PTHREAD_MUTEX_INITIALIZER;
#define TRY(expr) do { OrtStatus* status=(expr); if(status)return status; } while(0)
static const char* op_name(const OrtCustomOp* op) { (void)op; return "RowLayerNorm"; }
static const char* provider(const OrtCustomOp* op) { (void)op; return "CPUExecutionProvider"; }
static size_t input_count(const OrtCustomOp* op) { (void)op; return 3; }
static size_t output_count(const OrtCustomOp* op) { (void)op; return 1; }
static ONNXTensorElementDataType tensor_type(const OrtCustomOp* op, size_t index) {
    (void)op; (void)index; return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
static OrtCustomOpInputOutputCharacteristic characteristic(const OrtCustomOp* op, size_t index) {
    (void)op; (void)index; return INPUT_OUTPUT_REQUIRED;
}
static OrtMemType memory_type(const OrtCustomOp* op, size_t index) {
    (void)op; (void)index; return OrtMemTypeDefault;
}
static int first_version(const OrtCustomOp* op) { (void)op; return 1; }
static int last_version(const OrtCustomOp* op) { (void)op; return INT_MAX; }
static OrtStatus* create_kernel(const OrtCustomOp* op, const OrtApi* api,
                                const OrtKernelInfo* info, void** out) {
    (void)op; float epsilon; int64_t axis;
    TRY(api->KernelInfoGetAttribute_float(info,"epsilon",&epsilon));
    TRY(api->KernelInfoGetAttribute_int64(info,"axis",&axis));
    if(axis!=-1 || !isfinite(epsilon) || epsilon<0)
        return api->CreateStatus(ORT_INVALID_ARGUMENT,"Expected last-axis normalization and finite nonnegative epsilon");
    RowKernel* kernel=malloc(sizeof(RowKernel));
    if(!kernel)return api->CreateStatus(ORT_FAIL,"Row norm allocation failed");
    kernel->api=api;kernel->epsilon=epsilon;*out=kernel;return NULL;
}
static void destroy_kernel(void* state) { free(state); }
static OrtStatus* get_shape(const OrtApi* api,const OrtValue* value,
                            int64_t* dims,size_t* rank,size_t* count) {
    OrtTensorTypeAndShapeInfo* info=NULL;
    TRY(api->GetTensorTypeAndShape(value,&info));
    OrtStatus* status=api->GetDimensionsCount(info,rank);
    if(!status && *rank>8)status=api->CreateStatus(ORT_INVALID_ARGUMENT,"Rank exceeds eight");
    if(!status)status=api->GetDimensions(info,dims,*rank);
    if(!status)status=api->GetTensorShapeElementCount(info,count);
    api->ReleaseTensorTypeAndShapeInfo(info);return status;
}
static OrtStatus* compute_kernel(void* state,OrtKernelContext* context) {
    RowKernel* kernel=state;const OrtApi* api=kernel->api;
    const OrtValue *input,*scale,*bias;
    TRY(api->KernelContext_GetInput(context,0,&input));
    TRY(api->KernelContext_GetInput(context,1,&scale));
    TRY(api->KernelContext_GetInput(context,2,&bias));
    int64_t dims[8],sd[8],bd[8];size_t rank,sr,br,count,sc,bc;
    TRY(get_shape(api,input,dims,&rank,&count));
    TRY(get_shape(api,scale,sd,&sr,&sc));TRY(get_shape(api,bias,bd,&br,&bc));
    if(!rank || dims[rank-1]<=0 || sr!=1 || br!=1 || sc!=(size_t)dims[rank-1] || bc!=sc)
        return api->CreateStatus(ORT_INVALID_ARGUMENT,"Expected matching one-dimensional affine tensors");
    const void *x,*s,*b;void* y;OrtValue* output;
    TRY(api->GetTensorData(input,&x));TRY(api->GetTensorData(scale,&s));TRY(api->GetTensorData(bias,&b));
    TRY(api->KernelContext_GetOutput(context,0,dims,rank,&output));
    TRY(api->GetTensorMutableData(output,&y));
    row_norm(x,s,b,y,(int64_t)(count/(size_t)dims[rank-1]),dims[rank-1],kernel->epsilon);
    return NULL;
}
static OrtCustomOp row_op={.version=ORT_API_VERSION,.GetName=op_name,.GetExecutionProviderType=provider,
    .GetInputType=tensor_type,.GetInputTypeCount=input_count,.GetOutputType=tensor_type,
    .GetOutputTypeCount=output_count,.KernelDestroy=destroy_kernel,
    .GetInputCharacteristic=characteristic,.GetOutputCharacteristic=characteristic,
    .GetInputMemoryType=memory_type,.CreateKernelV2=create_kernel,.KernelComputeV2=compute_kernel,
    .GetStartVersion=first_version,.GetEndVersion=last_version};
__attribute__((visibility("default")))
OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base) {
    const OrtApi* api=base->GetApi(ORT_API_VERSION);
    if(!api)return NULL;
    pthread_mutex_lock(&domain_mutex);
    OrtStatus* status=NULL;
    if(!row_domain) {
        status=api->CreateCustomOpDomain("com.supertonic.rownorm",&row_domain);
        if(!status)status=api->CustomOpDomain_Add(row_domain,&row_op);
        if(status && row_domain) {
            api->ReleaseCustomOpDomain(row_domain);row_domain=NULL;
        }
    }
    pthread_mutex_unlock(&domain_mutex);
    if(status)return status;
    return api->AddCustomOpDomain(options,row_domain);
}
