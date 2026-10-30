/* Independent row-parallel Welford kernel. No reassociation within a row.
 * FMA sites match the target ORT kernel; the Python loader verifies bitidentity. */
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <pthread.h>
#include "onnxruntime_c_api.h"

__attribute__((visibility("default")))
void row_norm(const float* x, const float* scale, const float* bias,
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
