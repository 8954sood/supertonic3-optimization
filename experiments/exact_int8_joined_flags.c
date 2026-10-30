/* Experimental prejoined immutable safety flags; v4 arithmetic unchanged. */
/* Engine v4: exact U8 x reduced-range S8 -> S32, M4N16 safe-K8 tiles.
 * M8N8 handles unsupported tile shapes. No requantization or approximation. */
#include "onnxruntime_c_api.h"
#include <immintrin.h>
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
static inline __m256i dot4(__m256i b, const uint8_t* a) {
    __m256i av=_mm256_set1_epi32((int32_t)word(a));
    return _mm256_madd_epi16(_mm256_maddubs_epi16(av,b),_mm256_set1_epi16(1));
}
static int pack_weights(const int8_t* b,int64_t k,int64_t n,int8_t* packed,int32_t* sums) {
    for(int64_t col=0;col<n;col+=8) {
        __m256i sum=_mm256_setzero_si256();
        for(int64_t q=0;q<k;q+=4) {
            __m128i r0=_mm_loadl_epi64((const __m128i*)(b+q*n+col));
            __m128i r1=_mm_loadl_epi64((const __m128i*)(b+(q+1)*n+col));
            __m128i r2=_mm_loadl_epi64((const __m128i*)(b+(q+2)*n+col));
            __m128i r3=_mm_loadl_epi64((const __m128i*)(b+(q+3)*n+col));
            __m128i a01=_mm_unpacklo_epi8(r0,r1),a23=_mm_unpacklo_epi8(r2,r3);
            __m256i v=_mm256_set_m128i(_mm_unpackhi_epi16(a01,a23),_mm_unpacklo_epi16(a01,a23));
            __m256i bad=_mm256_or_si256(_mm256_cmpgt_epi8(v,_mm256_set1_epi8(64)),_mm256_cmpgt_epi8(_mm256_set1_epi8(-64),v));
            if(_mm256_movemask_epi8(bad))return -3;
            _mm256_storeu_si256((__m256i*)(packed+(col/8)*k*8+q*8),v);
            sum=_mm256_add_epi32(sum,_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1),v),_mm256_set1_epi16(1)));
        }
        _mm256_storeu_si256((__m256i*)(sums+col),sum);
    }
    if(k%8==0) {
        uint8_t* flags=(uint8_t*)(sums+n);
        for(int64_t col=0;col<n;col+=8)for(int64_t q=0;q<k;q+=8) {
            int bad=0;
            for(int c=0;c<8;++c)for(int lane=0;lane<2;++lane) {
                int pos=0,neg=0;
                for(int j=0;j<2;++j)for(int z=0;z<2;++z) {
                    int v=b[(q+4*j+2*lane+z)*n+col+c];if(v>0)pos+=v;else neg+=v;
                }
                if(pos>128||neg < -128)bad=1;
            }
            flags[(col/8)*(k/8)+q/8]=(uint8_t)bad;
        }
    }
    if(k%8==0 && n%16==0) {
        uint8_t* flags=(uint8_t*)(sums+n);
        uint8_t* joined=flags+k*n/64;
        for(int64_t col=0;col<n;col+=16)for(int64_t q=0;q<k/8;++q)
            joined[(col/16)*(k/8)+q]=flags[(col/8)*(k/8)+q]|flags[(col/8+1)*(k/8)+q];
    }
    return 0;
}
static int gemm_row8(const uint8_t* a,const int8_t* packed,const int32_t* sums,int32_t* y,int64_t m,int64_t k,int64_t n,uint8_t zero) {
    int64_t row=0;
    for(;row+8<=m;row+=8) {
        for(int64_t col=0;col<n;col+=8) {
            __m256i c0=_mm256_setzero_si256(),c1=c0,c2=c0,c3=c0,c4=c0,c5=c0,c6=c0,c7=c0;
            const int8_t* p=packed+(col/8)*k*8;
            for(int64_t q=0;q<k;q+=4,p+=32) {
                __m256i bv=_mm256_loadu_si256((const __m256i*)p);
                c0=_mm256_add_epi32(c0,dot4(bv,a+(row+0)*k+q));
                c1=_mm256_add_epi32(c1,dot4(bv,a+(row+1)*k+q));
                c2=_mm256_add_epi32(c2,dot4(bv,a+(row+2)*k+q));
                c3=_mm256_add_epi32(c3,dot4(bv,a+(row+3)*k+q));
                c4=_mm256_add_epi32(c4,dot4(bv,a+(row+4)*k+q));
                c5=_mm256_add_epi32(c5,dot4(bv,a+(row+5)*k+q));
                c6=_mm256_add_epi32(c6,dot4(bv,a+(row+6)*k+q));
                c7=_mm256_add_epi32(c7,dot4(bv,a+(row+7)*k+q));
            }
            __m256i z=_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)(sums+col)),_mm256_set1_epi32(zero));
#define STORE(r,c) _mm256_storeu_si256((__m256i*)(y+(row+(r))*n+col),_mm256_sub_epi32((c),z))
            STORE(0,c0);STORE(1,c1);STORE(2,c2);STORE(3,c3);STORE(4,c4);STORE(5,c5);STORE(6,c6);STORE(7,c7);
#undef STORE
        }
    }
    for(;row<m;++row)for(int64_t col=0;col<n;col+=8) {
        __m256i acc=_mm256_setzero_si256();const int8_t* p=packed+(col/8)*k*8;
        for(int64_t q=0;q<k;q+=4,p+=32)acc=_mm256_add_epi32(acc,dot4(_mm256_loadu_si256((const __m256i*)p),a+row*k+q));
        __m256i z=_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)(sums+col)),_mm256_set1_epi32(zero));
        _mm256_storeu_si256((__m256i*)(y+row*n+col),_mm256_sub_epi32(acc,z));
    }
    return 0;
}
/* Two four-coefficient dots can share the widening operation when their
 * signed bounds fit int16 for every possible U8 activation. Otherwise use
 * the exact original widening path. No data-dependent approximation. */
static int gemm_v2(const uint8_t* a,const int8_t* packed,const int32_t* sums,int32_t* y,int64_t m,int64_t k,int64_t n,uint8_t zero) {
    if(k%8||m<8)return gemm_row8(a,packed,sums,y,m,k,n,zero);
    const uint8_t* unsafe=(const uint8_t*)(sums+n);int64_t row=0;
    for(;row+8<=m;row+=8)for(int64_t col=0;col<n;col+=8) {
        __m256i c0=_mm256_setzero_si256(),c1=c0,c2=c0,c3=c0,c4=c0,c5=c0,c6=c0,c7=c0;
        const int8_t* p=packed+(col/8)*k*8;const uint8_t* flags=unsafe+(col/8)*(k/8);
        for(int64_t q=0;q<k;q+=8,p+=64) {
            __m256i b0=_mm256_loadu_si256((const __m256i*)p),b1=_mm256_loadu_si256((const __m256i*)(p+32));
            if(!flags[q/8]) {
#define FAST(r) do { __m256i a0=_mm256_set1_epi32((int32_t)word(a+(row+(r))*k+q)),a1=_mm256_set1_epi32((int32_t)word(a+(row+(r))*k+q+4)); __m256i partial=_mm256_add_epi16(_mm256_maddubs_epi16(a0,b0),_mm256_maddubs_epi16(a1,b1)); c##r=_mm256_add_epi32(c##r,_mm256_madd_epi16(partial,_mm256_set1_epi16(1))); } while(0)
                FAST(0);FAST(1);FAST(2);FAST(3);FAST(4);FAST(5);FAST(6);FAST(7);
#undef FAST
            } else {
#define SLOW(r) c##r=_mm256_add_epi32(c##r,_mm256_add_epi32(dot4(b0,a+(row+(r))*k+q),dot4(b1,a+(row+(r))*k+q+4)))
                SLOW(0);SLOW(1);SLOW(2);SLOW(3);SLOW(4);SLOW(5);SLOW(6);SLOW(7);
#undef SLOW
            }
        }
        __m256i z=_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)(sums+col)),_mm256_set1_epi32(zero));
#define SAVE(r) _mm256_storeu_si256((__m256i*)(y+(row+(r))*n+col),_mm256_sub_epi32(c##r,z))
        SAVE(0);SAVE(1);SAVE(2);SAVE(3);SAVE(4);SAVE(5);SAVE(6);SAVE(7);
#undef SAVE
    }
    if(row<m)return gemm_row8(a+row*k,packed,sums,y+row*n,m-row,k,n,zero);
    return 0;
}


/* Four rows and sixteen columns reuse activation broadcasts across two
 * output tiles, retaining v2's proven-safe K8 widening and exact fallback. */
static int gemm_packed(const uint8_t* a,const int8_t* packed,const int32_t* sums,int32_t* y,int64_t m,int64_t k,int64_t n,uint8_t zero) {
    if(k%8||n%16||m<4)return gemm_v2(a,packed,sums,y,m,k,n,zero);
    const uint8_t* unsafe=(const uint8_t*)(sums+n);int64_t row=0;
    for(;row+4<=m;row+=4)for(int64_t col=0;col<n;col+=16) {
        __m256i c00=_mm256_setzero_si256(),c01=c00,c10=c00,c11=c00,c20=c00,c21=c00,c30=c00,c31=c00;
        const int8_t* p0=packed+(col/8)*k*8;
        const int8_t* p1=p0+k*8;
        const uint8_t* flags=unsafe+k*n/64+(col/16)*(k/8);
        for(int64_t q=0;q<k;q+=8,p0+=64,p1+=64) {
            __m256i b00=_mm256_loadu_si256((const __m256i*)p0),b01=_mm256_loadu_si256((const __m256i*)(p0+32));
            __m256i b10=_mm256_loadu_si256((const __m256i*)p1),b11=_mm256_loadu_si256((const __m256i*)(p1+32));
            if(!flags[q/8]) {
#define FAST(r) do { __m256i a0=_mm256_set1_epi32((int32_t)word(a+(row+(r))*k+q)),a1=_mm256_set1_epi32((int32_t)word(a+(row+(r))*k+q+4)); __m256i v0=_mm256_add_epi16(_mm256_maddubs_epi16(a0,b00),_mm256_maddubs_epi16(a1,b01)); __m256i v1=_mm256_add_epi16(_mm256_maddubs_epi16(a0,b10),_mm256_maddubs_epi16(a1,b11)); c##r##0=_mm256_add_epi32(c##r##0,_mm256_madd_epi16(v0,_mm256_set1_epi16(1)));c##r##1=_mm256_add_epi32(c##r##1,_mm256_madd_epi16(v1,_mm256_set1_epi16(1)));} while(0)
                FAST(0);FAST(1);FAST(2);FAST(3);
#undef FAST
            } else {
#define SLOW(r) do {c##r##0=_mm256_add_epi32(c##r##0,_mm256_add_epi32(dot4(b00,a+(row+(r))*k+q),dot4(b01,a+(row+(r))*k+q+4)));c##r##1=_mm256_add_epi32(c##r##1,_mm256_add_epi32(dot4(b10,a+(row+(r))*k+q),dot4(b11,a+(row+(r))*k+q+4)));}while(0)
                SLOW(0);SLOW(1);SLOW(2);SLOW(3);
#undef SLOW
            }
        }
        __m256i z0=_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)(sums+col)),_mm256_set1_epi32(zero));
        __m256i z1=_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)(sums+col+8)),_mm256_set1_epi32(zero));
#define SAVE(r) do { _mm256_storeu_si256((__m256i*)(y+(row+(r))*n+col),_mm256_sub_epi32(c##r##0,z0));_mm256_storeu_si256((__m256i*)(y+(row+(r))*n+col+8),_mm256_sub_epi32(c##r##1,z1)); } while(0)
        SAVE(0);SAVE(1);SAVE(2);SAVE(3);
#undef SAVE
    }
    if(row<m)return gemm_v2(a+row*k,packed,sums,y+row*n,m-row,k,n,zero);
    return 0;
}

/* Standalone benchmark API separates one-time packing from warm execution. */
__attribute__((visibility("default")))
int prepare_exact_u8s8(const int8_t* b,int64_t k,int64_t n) {
    prepared_k=0;prepared_n=0;
    if(k<1||n<1||k%4||n%8||k>65536||(uint64_t)k>SIZE_MAX/(uint64_t)n)return -1;
    size_t bytes=(size_t)k*(size_t)n;
    if(bytes>packed_capacity){void* p=realloc(packed,bytes);if(!p)return -2;packed=p;packed_capacity=bytes;}
    size_t needed=(size_t)n+(k%8?0:((size_t)k*(size_t)n/64+(n%16?0:(size_t)k*n/128)+3)/4);
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
        kernel->packed=malloc((size_t)bd[0]*(size_t)bd[1]);kernel->sums=malloc((size_t)bd[1]*sizeof(int32_t)+(bd[0]%8?0:(size_t)bd[0]*bd[1]/64+(bd[1]%16?0:(size_t)bd[0]*bd[1]/128)));
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
