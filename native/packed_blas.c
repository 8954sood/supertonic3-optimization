/* Single-thread packed-weight matmul. ORT and NumPy-owned BLAS stay outside this library. */
#include "onnxruntime_c_api.h"
#include <dlfcn.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

typedef void (*Sgemm64)(int, int, int, int64_t, int64_t, int64_t,
                        float, const float*, int64_t, const float*, int64_t,
                        float, float*, int64_t);
static Sgemm64 gemm = NULL;
static void* blas_handle = NULL;
static OrtCustomOpDomain* domain = NULL;
/* Bounded thread-local arena: reuse one decoded layer buffer across all nodes. */
static _Thread_local float* scratch = NULL;
static _Thread_local size_t scratch_bytes = 0;
typedef struct { const OrtApi* api; } Kernel;
#define TRY(expr) do { OrtStatus* status = (expr); if (status) return status; } while (0)

static const char* name(const OrtCustomOp* op) { (void)op; return "PackedW8MatMul"; }
static const char* provider(const OrtCustomOp* op) { (void)op; return "CPUExecutionProvider"; }
static size_t inputs(const OrtCustomOp* op) { (void)op; return 3; }
static size_t outputs(const OrtCustomOp* op) { (void)op; return 1; }
static ONNXTensorElementDataType input_type(const OrtCustomOp* op, size_t index) {
  (void)op; return index == 1 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
static ONNXTensorElementDataType output_type(const OrtCustomOp* op, size_t index) {
  (void)op; (void)index; return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
static OrtCustomOpInputOutputCharacteristic characteristic(const OrtCustomOp* op, size_t index) {
  (void)op; (void)index; return INPUT_OUTPUT_REQUIRED;
}
static OrtMemType memory_type(const OrtCustomOp* op, size_t index) { (void)op; (void)index; return OrtMemTypeDefault; }
static int first_version(const OrtCustomOp* op) { (void)op; return 1; }
static int last_version(const OrtCustomOp* op) { (void)op; return INT_MAX; }
static OrtStatus* create(const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info, void** out) {
  (void)op; (void)info;
  Kernel* k = malloc(sizeof(Kernel));
  if (!k) return api->CreateStatus(ORT_FAIL, "kernel allocation failed");
  k->api = api; *out = k; return NULL;
}
static void destroy(void* state) { free(state); }
static OrtStatus* shape(const OrtApi* api, const OrtValue* value, int64_t* dims, size_t* rank) {
  OrtTensorTypeAndShapeInfo* info = NULL;
  TRY(api->GetTensorTypeAndShape(value, &info));
  OrtStatus* status = api->GetDimensionsCount(info, rank);
  if (!status && *rank > 8) status = api->CreateStatus(ORT_INVALID_ARGUMENT, "rank exceeds 8");
  if (!status) status = api->GetDimensions(info, dims, *rank);
  api->ReleaseTensorTypeAndShapeInfo(info); return status;
}
static OrtStatus* compute(void* state, OrtKernelContext* context) {
  const OrtApi* api = ((Kernel*)state)->api;
  const OrtValue *xv, *wv, *sv;
  TRY(api->KernelContext_GetInput(context, 0, &xv));
  TRY(api->KernelContext_GetInput(context, 1, &wv));
  TRY(api->KernelContext_GetInput(context, 2, &sv));
  int64_t xd[8], wd[8], sd[8]; size_t xr, wr, sr;
  TRY(shape(api, xv, xd, &xr)); TRY(shape(api, wv, wd, &wr)); TRY(shape(api, sv, sd, &sr));
  if (xr < 2 || wr != 2 || sr != 1 || xd[xr-1] != wd[0] || sd[0] != wd[1])
    return api->CreateStatus(ORT_INVALID_ARGUMENT, "incompatible packed matmul dimensions");
  int64_t m = 1, k = wd[0], n = wd[1];
  for (size_t d=0; d+1<xr; ++d) {
    if (xd[d] <= 0 || m > INT64_MAX / xd[d]) return api->CreateStatus(ORT_INVALID_ARGUMENT, "invalid batch dimensions");
    m *= xd[d];
  }
  if (k <= 0 || n <= 0 || (uint64_t)k > SIZE_MAX / sizeof(float) / (uint64_t)n)
    return api->CreateStatus(ORT_INVALID_ARGUMENT, "invalid weight dimensions");
  const void *xp, *wp, *sp;
  TRY(api->GetTensorData(xv, &xp)); TRY(api->GetTensorData(wv, &wp)); TRY(api->GetTensorData(sv, &sp));
  xd[xr-1] = n; OrtValue* yv; void* yp;
  TRY(api->KernelContext_GetOutput(context, 0, xd, xr, &yv));
  TRY(api->GetTensorMutableData(yv, &yp));
  size_t needed = (size_t)k * (size_t)n * sizeof(float);
  if (needed > scratch_bytes) {
    float* grown = realloc(scratch, needed);
    if (!grown) return api->CreateStatus(ORT_FAIL, "layer weight allocation failed");
    scratch = grown; scratch_bytes = needed;
  }
  float* dense = scratch;
  const int8_t* q = wp; const float* scale = sp;
  for (int64_t i=0; i<k; ++i)
    for (int64_t j=0; j<n; ++j) dense[i*n+j] = (float)q[i*n+j] * scale[j];
  gemm(101,111,111,m,n,k,1.0f,xp,k,dense,n,0.0f,yp,n);
  return NULL;
}
static OrtCustomOp op = {
  .version=ORT_API_VERSION, .GetName=name, .GetExecutionProviderType=provider,
  .GetInputType=input_type, .GetInputTypeCount=inputs,
  .GetOutputType=output_type, .GetOutputTypeCount=outputs,
  .KernelDestroy=destroy, .GetInputCharacteristic=characteristic,
  .GetOutputCharacteristic=characteristic, .GetInputMemoryType=memory_type,
  .CreateKernelV2=create, .KernelComputeV2=compute,
  .GetStartVersion=first_version, .GetEndVersion=last_version
};
__attribute__((visibility("default")))
OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* base) {
  const OrtApi* api = base->GetApi(ORT_API_VERSION);
  if (!api) return NULL; /* Runtime version is checked by the Python loader. */
  if (!gemm) {
    const char* path = getenv("TTS_OPENBLAS_LIBRARY");
    if (!path) return api->CreateStatus(ORT_FAIL, "TTS_OPENBLAS_LIBRARY is unset");
    blas_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!blas_handle) return api->CreateStatus(ORT_FAIL, dlerror());
    gemm = (Sgemm64)dlsym(blas_handle, "scipy_cblas_sgemm64_");
    if (!gemm) return api->CreateStatus(ORT_FAIL, "NumPy ILP64 SGEMM symbol missing");
  }
  if (!domain) {
    TRY(api->CreateCustomOpDomain("com.supertonic.blas", &domain));
    TRY(api->CustomOpDomain_Add(domain, &op));
  }
  return api->AddCustomOpDomain(options, domain);
}
