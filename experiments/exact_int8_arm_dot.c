/* Experimental exact ARM64 SDOT kernel, no approximation or requantization.
 * Center U8 by XOR128; compensate using the original column sums.
 * Target requires ARMv8.2-A dot-product support. */
#include "onnxruntime_c_api.h"
#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
static _Thread_local int8_t *packed = NULL;
static _Thread_local size_t packed_capacity = 0;
static _Thread_local int32_t *sums = NULL;
static _Thread_local size_t sums_capacity = 0;
static _Thread_local int64_t prepared_k = 0, prepared_n = 0;
static OrtCustomOpDomain* domain = NULL;

typedef struct { const OrtApi* api; int8_t* packed; int32_t* sums; int64_t k,n; const void* weight; } Kernel;
#define TRY(expr) do { OrtStatus* s=(expr); if(s) return s; } while(0)
static inline uint32_t word(const uint8_t* a) { uint32_t v; memcpy(&v,a,4); return v; }

#ifndef ROW_TILE
#define ROW_TILE 4
#endif
#if ROW_TILE != 4 && ROW_TILE != 6
#error "Supported experimental row tiles are 4 or 6"
#endif
static int pack_weights(const int8_t* b,int64_t k,int64_t n,int8_t* out,int32_t* sums) {
    memset(sums,0,(size_t)n*sizeof(int32_t));
    for(int64_t col=0;col<n;col+=4)for(int64_t q=0;q<k;q+=4) {
        int8_t* p=out+(col/4)*k*4+q*4;
        for(int c=0;c<4;++c)for(int j=0;j<4;++j) {
            int v=b[(q+j)*n+col+c];if(v < -64 || v>64)return -3;
            p[c*4+j]=(int8_t)v;sums[col+c]+=v;
        }
    }
    return 0;
}
static inline int8x16_t activation(const uint8_t* a) {
    return vreinterpretq_s8_u32(vdupq_n_u32(word(a)^UINT32_C(0x80808080)));
}
static int gemm_packed(const uint8_t* a,const int8_t* packed,const int32_t* sums,int32_t* y,int64_t m,int64_t k,int64_t n,uint8_t zero) {
    const int32x4_t zp=vdupq_n_s32((int32_t)zero-128);
    int64_t row=0;
    if(n%16==0)for(;row+ROW_TILE<=m;row+=ROW_TILE)for(int64_t col=0;col<n;col+=16) {
#define DECLARE(r) int32x4_t c##r##0=vdupq_n_s32(0),c##r##1=c##r##0,c##r##2=c##r##0,c##r##3=c##r##0
        DECLARE(0);DECLARE(1);DECLARE(2);DECLARE(3);
#if ROW_TILE == 6
        DECLARE(4);DECLARE(5);
#endif
#undef DECLARE
        const int8_t* p0=packed+(col/4)*k*4;
        const int8_t* p1=p0+k*4;const int8_t* p2=p1+k*4;const int8_t* p3=p2+k*4;
        for(int64_t q=0;q<k;q+=4,p0+=16,p1+=16,p2+=16,p3+=16) {
            int8x16_t b0=vld1q_s8(p0),b1=vld1q_s8(p1),b2=vld1q_s8(p2),b3=vld1q_s8(p3);
#define DOT(r) do {int8x16_t av=activation(a+(row+(r))*k+q);c##r##0=vdotq_s32(c##r##0,b0,av);c##r##1=vdotq_s32(c##r##1,b1,av);c##r##2=vdotq_s32(c##r##2,b2,av);c##r##3=vdotq_s32(c##r##3,b3,av);}while(0)
            DOT(0);DOT(1);DOT(2);DOT(3);
#if ROW_TILE == 6
            DOT(4);DOT(5);
#endif
#undef DOT
        }
        int32x4_t z0=vmulq_s32(vld1q_s32(sums+col),zp),z1=vmulq_s32(vld1q_s32(sums+col+4),zp),z2=vmulq_s32(vld1q_s32(sums+col+8),zp),z3=vmulq_s32(vld1q_s32(sums+col+12),zp);
#define SAVE(r) do {vst1q_s32(y+(row+(r))*n+col,vsubq_s32(c##r##0,z0));vst1q_s32(y+(row+(r))*n+col+4,vsubq_s32(c##r##1,z1));vst1q_s32(y+(row+(r))*n+col+8,vsubq_s32(c##r##2,z2));vst1q_s32(y+(row+(r))*n+col+12,vsubq_s32(c##r##3,z3));}while(0)
        SAVE(0);SAVE(1);SAVE(2);SAVE(3);
#if ROW_TILE == 6
        SAVE(4);SAVE(5);
#endif
#undef SAVE
    }
    for(;row<m;++row)for(int64_t col=0;col<n;col+=4) {
        int32x4_t acc=vdupq_n_s32(0);const int8_t* p=packed+(col/4)*k*4;
        for(int64_t q=0;q<k;q+=4,p+=16)acc=vdotq_s32(acc,vld1q_s8(p),activation(a+row*k+q));
        vst1q_s32(y+row*n+col,vsubq_s32(acc,vmulq_s32(vld1q_s32(sums+col),zp)));
    }
    return 0;
}
/* Standalone benchmark API separates one-time packing from warm execution. */
__attribute__((visibility("default")))
int prepare_exact_u8s8(const int8_t* b,int64_t k,int64_t n) {
    prepared_k=0;prepared_n=0;
    if(k<1||n<1||k%4||n%8||k>65536||(uint64_t)k>SIZE_MAX/(uint64_t)n)return -1;
    size_t bytes=(size_t)k*(size_t)n;
    if(bytes>packed_capacity){void* p=realloc(packed,bytes);if(!p)return -2;packed=p;packed_capacity=bytes;}
    size_t needed=(size_t)n+(k%8?0:((size_t)k*(size_t)n/64+3)/4);
    if(needed>sums_capacity){void* p=realloc(sums,needed*sizeof(int32_t));if(!p)return -2;sums=p;sums_capacity=needed;}
    int status=pack_weights(b,k,n,packed,sums);
    if(!status){prepared_k=k;prepared_n=n;}
    return status;
}
__attribute__((visibility("default")))
int run_prepared_u8s8(const uint8_t* a,int32_t* y,int64_t m,int64_t k,int64_t n,uint8_t zero) {
    if(m<1||k<1||n<1||k!=prepared_k||n!=prepared_n)return -1;
    return gemm_packed(a,packed,sums,y,m,k,n,zero);
}
__attribute__((visibility("default")))
int exact_u8s8(const uint8_t* a,const int8_t* b,int32_t* y,int64_t m,int64_t k,int64_t n,uint8_t zero) {
    int status=prepare_exact_u8s8(b,k,n);return status?status:run_prepared_u8s8(a,y,m,k,n,zero);
}

static const char* name(const OrtCustomOp* op){(void)op;return "ExactMatMulInteger";}
static const char* provider(const OrtCustomOp* op){(void)op;return "CPUExecutionProvider";}
static size_t inputs(const OrtCustomOp* op){(void)op;return 4;}
static size_t outputs(const OrtCustomOp* op){(void)op;return 1;}
static ONNXTensorElementDataType input_type(const OrtCustomOp* op,size_t i){(void)op;return i%2?ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;}
static ONNXTensorElementDataType output_type(const OrtCustomOp* op,size_t i){(void)op;(void)i;return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;}
static OrtCustomOpInputOutputCharacteristic characteristic(const OrtCustomOp* op,size_t i){(void)op;(void)i;return INPUT_OUTPUT_REQUIRED;}
static OrtMemType memory_type(const OrtCustomOp* op,size_t i){(void)op;(void)i;return OrtMemTypeDefault;}
static int first_version(const OrtCustomOp* op){(void)op;return 1;}
static int last_version(const OrtCustomOp* op){(void)op;return INT_MAX;}
static OrtStatus* create(const OrtCustomOp* op,const OrtApi* api,const OrtKernelInfo* info,void** out){
    (void)op;int constant=0;const OrtValue* value=NULL;
    TRY(api->KernelInfoGetConstantInput_tensor(info,1,&constant,&value));
    if(!constant||!value)return api->CreateStatus(ORT_INVALID_ARGUMENT,"exact packed kernel requires immutable initializer weights");
    Kernel* k=calloc(1,sizeof(Kernel));if(!k)return api->CreateStatus(ORT_FAIL,"allocation failed");k->api=api;*out=k;return NULL;
}
static void destroy(void* state){Kernel* k=state;free(k->packed);free(k->sums);free(k);}
static OrtStatus* shape(const OrtApi* api,const OrtValue* value,int64_t* dims,size_t* rank,size_t* count){OrtTensorTypeAndShapeInfo* info=NULL;TRY(api->GetTensorTypeAndShape(value,&info));OrtStatus* s=api->GetDimensionsCount(info,rank);if(!s&&*rank>8)s=api->CreateStatus(ORT_INVALID_ARGUMENT,"rank exceeds eight");if(!s)s=api->GetDimensions(info,dims,*rank);if(!s)s=api->GetTensorShapeElementCount(info,count);api->ReleaseTensorTypeAndShapeInfo(info);return s;}
static OrtStatus* compute(void* state,OrtKernelContext* context){
    const OrtApi* api=((Kernel*)state)->api;const OrtValue *av,*bv,*azv,*bzv;
    TRY(api->KernelContext_GetInput(context,0,&av));TRY(api->KernelContext_GetInput(context,1,&bv));TRY(api->KernelContext_GetInput(context,2,&azv));TRY(api->KernelContext_GetInput(context,3,&bzv));
    int64_t ad[8],bd[8],zd[8];size_t ar,br,zr,ac,bc,azc,bzc;TRY(shape(api,av,ad,&ar,&ac));TRY(shape(api,bv,bd,&br,&bc));TRY(shape(api,azv,zd,&zr,&azc));TRY(shape(api,bzv,zd,&zr,&bzc));
    if(ar<2||br!=2||bd[0]<=0||bd[1]<=0||bd[0]%4||bd[1]%8||bd[0]>65536||(uint64_t)bd[0]>SIZE_MAX/(uint64_t)bd[1]||ad[ar-1]!=bd[0]||azc!=1||(bzc!=1&&bzc!=(size_t)bd[1]))return api->CreateStatus(ORT_INVALID_ARGUMENT,"unsupported integer matmul shapes");
    const void *ap,*bp,*azp,*bzp;TRY(api->GetTensorData(av,&ap));TRY(api->GetTensorData(bv,&bp));TRY(api->GetTensorData(azv,&azp));TRY(api->GetTensorData(bzv,&bzp));
    for(size_t i=0;i<bzc;++i)if(((const int8_t*)bzp)[i])return api->CreateStatus(ORT_INVALID_ARGUMENT,"weight zero point must be zero");
    int64_t m=(int64_t)(ac/(size_t)bd[0]);ad[ar-1]=bd[1];OrtValue* yv;void* yp;TRY(api->KernelContext_GetOutput(context,0,ad,ar,&yv));TRY(api->GetTensorMutableData(yv,&yp));
    Kernel* kernel=state;
    if(bd[0]<=0||bd[1]<=0||bd[0]%4||bd[1]%8||bd[0]>65536)return api->CreateStatus(ORT_INVALID_ARGUMENT,"unsupported kernel dimensions");
    if(kernel->weight!=bp||kernel->k!=bd[0]||kernel->n!=bd[1]) {
        free(kernel->packed);free(kernel->sums);kernel->packed=NULL;kernel->sums=NULL;kernel->weight=NULL;
        kernel->packed=malloc((size_t)bd[0]*(size_t)bd[1]);kernel->sums=malloc((size_t)bd[1]*sizeof(int32_t)+(bd[0]%8?0:(size_t)bd[0]*bd[1]/64));
        if(!kernel->packed||!kernel->sums)return api->CreateStatus(ORT_FAIL,"weight packing allocation failed");
        int status=pack_weights(bp,bd[0],bd[1],kernel->packed,kernel->sums);
        if(status)return api->CreateStatus(ORT_INVALID_ARGUMENT,"weights exceed reduced range [-64,64]");
        kernel->weight=bp;kernel->k=bd[0];kernel->n=bd[1];
    }
    int result=gemm_packed(ap,kernel->packed,kernel->sums,yp,m,bd[0],bd[1],*(const uint8_t*)azp);
    if(result)return api->CreateStatus(ORT_FAIL,result==-3?"weights exceed reduced range [-64,64]":"integer kernel failed");return NULL;
}
static OrtCustomOp op={.version=ORT_API_VERSION,.GetName=name,.GetExecutionProviderType=provider,.GetInputType=input_type,.GetInputTypeCount=inputs,.GetOutputType=output_type,.GetOutputTypeCount=outputs,.KernelDestroy=destroy,.GetInputCharacteristic=characteristic,.GetOutputCharacteristic=characteristic,.GetInputMemoryType=memory_type,.CreateKernelV2=create,.KernelComputeV2=compute,.GetStartVersion=first_version,.GetEndVersion=last_version};
__attribute__((visibility("default"))) OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base){const OrtApi* api=base->GetApi(ORT_API_VERSION);if(!api)return NULL;if(!domain){TRY(api->CreateCustomOpDomain("com.supertonic.exact",&domain));TRY(api->CustomOpDomain_Add(domain,&op));}return api->AddCustomOpDomain(options,domain);}
