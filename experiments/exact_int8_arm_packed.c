/* Experimental lossless packed-A lane SDOT, ARMv8.2 dotprod. */
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
#if ROW_TILE != 4 && ROW_TILE != 8 && ROW_TILE != 12
#error "Supported packed activation tiles are 4, 8 or 12"
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
static _Thread_local int8_t* activation_scratch=NULL;
static _Thread_local size_t activation_capacity=0;
static inline void tile4x8(const int8_t* a,const int8_t* b,const int32_t* sums,int32_t* y,int64_t k,int64_t n,int valid,int zero) {
    int32x4_t c0_0=vdupq_n_s32(0);
    int32x4_t c0_1=vdupq_n_s32(0);
    int32x4_t c1_0=vdupq_n_s32(0);
    int32x4_t c1_1=vdupq_n_s32(0);
    int32x4_t c2_0=vdupq_n_s32(0);
    int32x4_t c2_1=vdupq_n_s32(0);
    int32x4_t c3_0=vdupq_n_s32(0);
    int32x4_t c3_1=vdupq_n_s32(0);
    for(int64_t q=0;q<k;q+=4) {
        int8x16_t a0=vld1q_s8(a+0*k*4+q*4);
        int8x16_t b0=vld1q_s8(b+0*k*4+q*4);
        int8x16_t b1=vld1q_s8(b+1*k*4+q*4);
        c0_0=vdotq_laneq_s32(c0_0,b0,a0,0);
        c0_1=vdotq_laneq_s32(c0_1,b1,a0,0);
        c1_0=vdotq_laneq_s32(c1_0,b0,a0,1);
        c1_1=vdotq_laneq_s32(c1_1,b1,a0,1);
        c2_0=vdotq_laneq_s32(c2_0,b0,a0,2);
        c2_1=vdotq_laneq_s32(c2_1,b1,a0,2);
        c3_0=vdotq_laneq_s32(c3_0,b0,a0,3);
        c3_1=vdotq_laneq_s32(c3_1,b1,a0,3);
    }
    int32x4_t z0=vmulq_s32(vld1q_s32(sums+0*4),vdupq_n_s32(zero-128));
    int32x4_t z1=vmulq_s32(vld1q_s32(sums+1*4),vdupq_n_s32(zero-128));
    if(valid>0) {
        vst1q_s32(y+0*n+0*4,vsubq_s32(c0_0,z0));
        vst1q_s32(y+0*n+1*4,vsubq_s32(c0_1,z1));
    }
    if(valid>1) {
        vst1q_s32(y+1*n+0*4,vsubq_s32(c1_0,z0));
        vst1q_s32(y+1*n+1*4,vsubq_s32(c1_1,z1));
    }
    if(valid>2) {
        vst1q_s32(y+2*n+0*4,vsubq_s32(c2_0,z0));
        vst1q_s32(y+2*n+1*4,vsubq_s32(c2_1,z1));
    }
    if(valid>3) {
        vst1q_s32(y+3*n+0*4,vsubq_s32(c3_0,z0));
        vst1q_s32(y+3*n+1*4,vsubq_s32(c3_1,z1));
    }
}
static inline void tile4x16(const int8_t* a,const int8_t* b,const int32_t* sums,int32_t* y,int64_t k,int64_t n,int valid,int zero) {
    int32x4_t c0_0=vdupq_n_s32(0);
    int32x4_t c0_1=vdupq_n_s32(0);
    int32x4_t c0_2=vdupq_n_s32(0);
    int32x4_t c0_3=vdupq_n_s32(0);
    int32x4_t c1_0=vdupq_n_s32(0);
    int32x4_t c1_1=vdupq_n_s32(0);
    int32x4_t c1_2=vdupq_n_s32(0);
    int32x4_t c1_3=vdupq_n_s32(0);
    int32x4_t c2_0=vdupq_n_s32(0);
    int32x4_t c2_1=vdupq_n_s32(0);
    int32x4_t c2_2=vdupq_n_s32(0);
    int32x4_t c2_3=vdupq_n_s32(0);
    int32x4_t c3_0=vdupq_n_s32(0);
    int32x4_t c3_1=vdupq_n_s32(0);
    int32x4_t c3_2=vdupq_n_s32(0);
    int32x4_t c3_3=vdupq_n_s32(0);
    for(int64_t q=0;q<k;q+=4) {
        int8x16_t a0=vld1q_s8(a+0*k*4+q*4);
        int8x16_t b0=vld1q_s8(b+0*k*4+q*4);
        int8x16_t b1=vld1q_s8(b+1*k*4+q*4);
        int8x16_t b2=vld1q_s8(b+2*k*4+q*4);
        int8x16_t b3=vld1q_s8(b+3*k*4+q*4);
        c0_0=vdotq_laneq_s32(c0_0,b0,a0,0);
        c0_1=vdotq_laneq_s32(c0_1,b1,a0,0);
        c0_2=vdotq_laneq_s32(c0_2,b2,a0,0);
        c0_3=vdotq_laneq_s32(c0_3,b3,a0,0);
        c1_0=vdotq_laneq_s32(c1_0,b0,a0,1);
        c1_1=vdotq_laneq_s32(c1_1,b1,a0,1);
        c1_2=vdotq_laneq_s32(c1_2,b2,a0,1);
        c1_3=vdotq_laneq_s32(c1_3,b3,a0,1);
        c2_0=vdotq_laneq_s32(c2_0,b0,a0,2);
        c2_1=vdotq_laneq_s32(c2_1,b1,a0,2);
        c2_2=vdotq_laneq_s32(c2_2,b2,a0,2);
        c2_3=vdotq_laneq_s32(c2_3,b3,a0,2);
        c3_0=vdotq_laneq_s32(c3_0,b0,a0,3);
        c3_1=vdotq_laneq_s32(c3_1,b1,a0,3);
        c3_2=vdotq_laneq_s32(c3_2,b2,a0,3);
        c3_3=vdotq_laneq_s32(c3_3,b3,a0,3);
    }
    int32x4_t z0=vmulq_s32(vld1q_s32(sums+0*4),vdupq_n_s32(zero-128));
    int32x4_t z1=vmulq_s32(vld1q_s32(sums+1*4),vdupq_n_s32(zero-128));
    int32x4_t z2=vmulq_s32(vld1q_s32(sums+2*4),vdupq_n_s32(zero-128));
    int32x4_t z3=vmulq_s32(vld1q_s32(sums+3*4),vdupq_n_s32(zero-128));
    if(valid>0) {
        vst1q_s32(y+0*n+0*4,vsubq_s32(c0_0,z0));
        vst1q_s32(y+0*n+1*4,vsubq_s32(c0_1,z1));
        vst1q_s32(y+0*n+2*4,vsubq_s32(c0_2,z2));
        vst1q_s32(y+0*n+3*4,vsubq_s32(c0_3,z3));
    }
    if(valid>1) {
        vst1q_s32(y+1*n+0*4,vsubq_s32(c1_0,z0));
        vst1q_s32(y+1*n+1*4,vsubq_s32(c1_1,z1));
        vst1q_s32(y+1*n+2*4,vsubq_s32(c1_2,z2));
        vst1q_s32(y+1*n+3*4,vsubq_s32(c1_3,z3));
    }
    if(valid>2) {
        vst1q_s32(y+2*n+0*4,vsubq_s32(c2_0,z0));
        vst1q_s32(y+2*n+1*4,vsubq_s32(c2_1,z1));
        vst1q_s32(y+2*n+2*4,vsubq_s32(c2_2,z2));
        vst1q_s32(y+2*n+3*4,vsubq_s32(c2_3,z3));
    }
    if(valid>3) {
        vst1q_s32(y+3*n+0*4,vsubq_s32(c3_0,z0));
        vst1q_s32(y+3*n+1*4,vsubq_s32(c3_1,z1));
        vst1q_s32(y+3*n+2*4,vsubq_s32(c3_2,z2));
        vst1q_s32(y+3*n+3*4,vsubq_s32(c3_3,z3));
    }
}
static inline void tile8x8(const int8_t* a,const int8_t* b,const int32_t* sums,int32_t* y,int64_t k,int64_t n,int valid,int zero) {
    int32x4_t c0_0=vdupq_n_s32(0);
    int32x4_t c0_1=vdupq_n_s32(0);
    int32x4_t c1_0=vdupq_n_s32(0);
    int32x4_t c1_1=vdupq_n_s32(0);
    int32x4_t c2_0=vdupq_n_s32(0);
    int32x4_t c2_1=vdupq_n_s32(0);
    int32x4_t c3_0=vdupq_n_s32(0);
    int32x4_t c3_1=vdupq_n_s32(0);
    int32x4_t c4_0=vdupq_n_s32(0);
    int32x4_t c4_1=vdupq_n_s32(0);
    int32x4_t c5_0=vdupq_n_s32(0);
    int32x4_t c5_1=vdupq_n_s32(0);
    int32x4_t c6_0=vdupq_n_s32(0);
    int32x4_t c6_1=vdupq_n_s32(0);
    int32x4_t c7_0=vdupq_n_s32(0);
    int32x4_t c7_1=vdupq_n_s32(0);
    for(int64_t q=0;q<k;q+=4) {
        int8x16_t a0=vld1q_s8(a+0*k*4+q*4);
        int8x16_t a1=vld1q_s8(a+1*k*4+q*4);
        int8x16_t b0=vld1q_s8(b+0*k*4+q*4);
        int8x16_t b1=vld1q_s8(b+1*k*4+q*4);
        c0_0=vdotq_laneq_s32(c0_0,b0,a0,0);
        c0_1=vdotq_laneq_s32(c0_1,b1,a0,0);
        c1_0=vdotq_laneq_s32(c1_0,b0,a0,1);
        c1_1=vdotq_laneq_s32(c1_1,b1,a0,1);
        c2_0=vdotq_laneq_s32(c2_0,b0,a0,2);
        c2_1=vdotq_laneq_s32(c2_1,b1,a0,2);
        c3_0=vdotq_laneq_s32(c3_0,b0,a0,3);
        c3_1=vdotq_laneq_s32(c3_1,b1,a0,3);
        c4_0=vdotq_laneq_s32(c4_0,b0,a1,0);
        c4_1=vdotq_laneq_s32(c4_1,b1,a1,0);
        c5_0=vdotq_laneq_s32(c5_0,b0,a1,1);
        c5_1=vdotq_laneq_s32(c5_1,b1,a1,1);
        c6_0=vdotq_laneq_s32(c6_0,b0,a1,2);
        c6_1=vdotq_laneq_s32(c6_1,b1,a1,2);
        c7_0=vdotq_laneq_s32(c7_0,b0,a1,3);
        c7_1=vdotq_laneq_s32(c7_1,b1,a1,3);
    }
    int32x4_t z0=vmulq_s32(vld1q_s32(sums+0*4),vdupq_n_s32(zero-128));
    int32x4_t z1=vmulq_s32(vld1q_s32(sums+1*4),vdupq_n_s32(zero-128));
    if(valid>0) {
        vst1q_s32(y+0*n+0*4,vsubq_s32(c0_0,z0));
        vst1q_s32(y+0*n+1*4,vsubq_s32(c0_1,z1));
    }
    if(valid>1) {
        vst1q_s32(y+1*n+0*4,vsubq_s32(c1_0,z0));
        vst1q_s32(y+1*n+1*4,vsubq_s32(c1_1,z1));
    }
    if(valid>2) {
        vst1q_s32(y+2*n+0*4,vsubq_s32(c2_0,z0));
        vst1q_s32(y+2*n+1*4,vsubq_s32(c2_1,z1));
    }
    if(valid>3) {
        vst1q_s32(y+3*n+0*4,vsubq_s32(c3_0,z0));
        vst1q_s32(y+3*n+1*4,vsubq_s32(c3_1,z1));
    }
    if(valid>4) {
        vst1q_s32(y+4*n+0*4,vsubq_s32(c4_0,z0));
        vst1q_s32(y+4*n+1*4,vsubq_s32(c4_1,z1));
    }
    if(valid>5) {
        vst1q_s32(y+5*n+0*4,vsubq_s32(c5_0,z0));
        vst1q_s32(y+5*n+1*4,vsubq_s32(c5_1,z1));
    }
    if(valid>6) {
        vst1q_s32(y+6*n+0*4,vsubq_s32(c6_0,z0));
        vst1q_s32(y+6*n+1*4,vsubq_s32(c6_1,z1));
    }
    if(valid>7) {
        vst1q_s32(y+7*n+0*4,vsubq_s32(c7_0,z0));
        vst1q_s32(y+7*n+1*4,vsubq_s32(c7_1,z1));
    }
}
static inline void tile12x8(const int8_t* a,const int8_t* b,const int32_t* sums,int32_t* y,int64_t k,int64_t n,int valid,int zero) {
    int32x4_t c0_0=vdupq_n_s32(0);
    int32x4_t c0_1=vdupq_n_s32(0);
    int32x4_t c1_0=vdupq_n_s32(0);
    int32x4_t c1_1=vdupq_n_s32(0);
    int32x4_t c2_0=vdupq_n_s32(0);
    int32x4_t c2_1=vdupq_n_s32(0);
    int32x4_t c3_0=vdupq_n_s32(0);
    int32x4_t c3_1=vdupq_n_s32(0);
    int32x4_t c4_0=vdupq_n_s32(0);
    int32x4_t c4_1=vdupq_n_s32(0);
    int32x4_t c5_0=vdupq_n_s32(0);
    int32x4_t c5_1=vdupq_n_s32(0);
    int32x4_t c6_0=vdupq_n_s32(0);
    int32x4_t c6_1=vdupq_n_s32(0);
    int32x4_t c7_0=vdupq_n_s32(0);
    int32x4_t c7_1=vdupq_n_s32(0);
    int32x4_t c8_0=vdupq_n_s32(0);
    int32x4_t c8_1=vdupq_n_s32(0);
    int32x4_t c9_0=vdupq_n_s32(0);
    int32x4_t c9_1=vdupq_n_s32(0);
    int32x4_t c10_0=vdupq_n_s32(0);
    int32x4_t c10_1=vdupq_n_s32(0);
    int32x4_t c11_0=vdupq_n_s32(0);
    int32x4_t c11_1=vdupq_n_s32(0);
    for(int64_t q=0;q<k;q+=4) {
        int8x16_t a0=vld1q_s8(a+0*k*4+q*4);
        int8x16_t a1=vld1q_s8(a+1*k*4+q*4);
        int8x16_t a2=vld1q_s8(a+2*k*4+q*4);
        int8x16_t b0=vld1q_s8(b+0*k*4+q*4);
        int8x16_t b1=vld1q_s8(b+1*k*4+q*4);
        c0_0=vdotq_laneq_s32(c0_0,b0,a0,0);
        c0_1=vdotq_laneq_s32(c0_1,b1,a0,0);
        c1_0=vdotq_laneq_s32(c1_0,b0,a0,1);
        c1_1=vdotq_laneq_s32(c1_1,b1,a0,1);
        c2_0=vdotq_laneq_s32(c2_0,b0,a0,2);
        c2_1=vdotq_laneq_s32(c2_1,b1,a0,2);
        c3_0=vdotq_laneq_s32(c3_0,b0,a0,3);
        c3_1=vdotq_laneq_s32(c3_1,b1,a0,3);
        c4_0=vdotq_laneq_s32(c4_0,b0,a1,0);
        c4_1=vdotq_laneq_s32(c4_1,b1,a1,0);
        c5_0=vdotq_laneq_s32(c5_0,b0,a1,1);
        c5_1=vdotq_laneq_s32(c5_1,b1,a1,1);
        c6_0=vdotq_laneq_s32(c6_0,b0,a1,2);
        c6_1=vdotq_laneq_s32(c6_1,b1,a1,2);
        c7_0=vdotq_laneq_s32(c7_0,b0,a1,3);
        c7_1=vdotq_laneq_s32(c7_1,b1,a1,3);
        c8_0=vdotq_laneq_s32(c8_0,b0,a2,0);
        c8_1=vdotq_laneq_s32(c8_1,b1,a2,0);
        c9_0=vdotq_laneq_s32(c9_0,b0,a2,1);
        c9_1=vdotq_laneq_s32(c9_1,b1,a2,1);
        c10_0=vdotq_laneq_s32(c10_0,b0,a2,2);
        c10_1=vdotq_laneq_s32(c10_1,b1,a2,2);
        c11_0=vdotq_laneq_s32(c11_0,b0,a2,3);
        c11_1=vdotq_laneq_s32(c11_1,b1,a2,3);
    }
    int32x4_t z0=vmulq_s32(vld1q_s32(sums+0*4),vdupq_n_s32(zero-128));
    int32x4_t z1=vmulq_s32(vld1q_s32(sums+1*4),vdupq_n_s32(zero-128));
    if(valid>0) {
        vst1q_s32(y+0*n+0*4,vsubq_s32(c0_0,z0));
        vst1q_s32(y+0*n+1*4,vsubq_s32(c0_1,z1));
    }
    if(valid>1) {
        vst1q_s32(y+1*n+0*4,vsubq_s32(c1_0,z0));
        vst1q_s32(y+1*n+1*4,vsubq_s32(c1_1,z1));
    }
    if(valid>2) {
        vst1q_s32(y+2*n+0*4,vsubq_s32(c2_0,z0));
        vst1q_s32(y+2*n+1*4,vsubq_s32(c2_1,z1));
    }
    if(valid>3) {
        vst1q_s32(y+3*n+0*4,vsubq_s32(c3_0,z0));
        vst1q_s32(y+3*n+1*4,vsubq_s32(c3_1,z1));
    }
    if(valid>4) {
        vst1q_s32(y+4*n+0*4,vsubq_s32(c4_0,z0));
        vst1q_s32(y+4*n+1*4,vsubq_s32(c4_1,z1));
    }
    if(valid>5) {
        vst1q_s32(y+5*n+0*4,vsubq_s32(c5_0,z0));
        vst1q_s32(y+5*n+1*4,vsubq_s32(c5_1,z1));
    }
    if(valid>6) {
        vst1q_s32(y+6*n+0*4,vsubq_s32(c6_0,z0));
        vst1q_s32(y+6*n+1*4,vsubq_s32(c6_1,z1));
    }
    if(valid>7) {
        vst1q_s32(y+7*n+0*4,vsubq_s32(c7_0,z0));
        vst1q_s32(y+7*n+1*4,vsubq_s32(c7_1,z1));
    }
    if(valid>8) {
        vst1q_s32(y+8*n+0*4,vsubq_s32(c8_0,z0));
        vst1q_s32(y+8*n+1*4,vsubq_s32(c8_1,z1));
    }
    if(valid>9) {
        vst1q_s32(y+9*n+0*4,vsubq_s32(c9_0,z0));
        vst1q_s32(y+9*n+1*4,vsubq_s32(c9_1,z1));
    }
    if(valid>10) {
        vst1q_s32(y+10*n+0*4,vsubq_s32(c10_0,z0));
        vst1q_s32(y+10*n+1*4,vsubq_s32(c10_1,z1));
    }
    if(valid>11) {
        vst1q_s32(y+11*n+0*4,vsubq_s32(c11_0,z0));
        vst1q_s32(y+11*n+1*4,vsubq_s32(c11_1,z1));
    }
}

static int gemm_packed(const uint8_t* a,const int8_t* packed,const int32_t* sums,int32_t* y,int64_t m,int64_t k,int64_t n,uint8_t zero) {
    uint64_t rows=((uint64_t)m+3)/4*4;
    if(rows>INT64_MAX||rows>SIZE_MAX/(uint64_t)k)return -1;
    size_t needed=(size_t)rows*(size_t)k;
    if(needed>activation_capacity) {
        void* pointer=realloc(activation_scratch,needed);
        if(!pointer)return -2;activation_scratch=pointer;activation_capacity=needed;
    }
    const uint32x4_t center=vdupq_n_u32(UINT32_C(0x80808080));
    int64_t row=0;
    for(;row+4<=m;row+=4)for(int64_t q=0;q<k;q+=4) {
        uint32x4_t v={word(a+row*k+q),word(a+(row+1)*k+q),word(a+(row+2)*k+q),word(a+(row+3)*k+q)};
        vst1q_u32((uint32_t*)(activation_scratch+(row/4)*k*4+q*4),veorq_u32(v,center));
    }
    if(row<m)for(int64_t q=0;q<k;q+=4) {
        uint32x4_t v={word(a+row*k+q),row+1<m?word(a+(row+1)*k+q):UINT32_C(0x80808080),row+2<m?word(a+(row+2)*k+q):UINT32_C(0x80808080),UINT32_C(0x80808080)};
        vst1q_u32((uint32_t*)(activation_scratch+(row/4)*k*4+q*4),veorq_u32(v,center));
    }
    row=0;
#if ROW_TILE == 8 || ROW_TILE == 12
    for(;row+ROW_TILE<=m;row+=ROW_TILE)for(int64_t col=0;col<n;col+=8) {
#if ROW_TILE == 8
        tile8x8(activation_scratch+(row/4)*k*4,packed+(col/4)*k*4,sums+col,y+row*n+col,k,n,8,zero);
#else
        tile12x8(activation_scratch+(row/4)*k*4,packed+(col/4)*k*4,sums+col,y+row*n+col,k,n,12,zero);
#endif
    }
#endif
    for(;row<m;row+=4) {
        int valid=m-row<4?(int)(m-row):4;
        if(n%16==0)for(int64_t col=0;col<n;col+=16)
            tile4x16(activation_scratch+(row/4)*k*4,packed+(col/4)*k*4,sums+col,y+row*n+col,k,n,valid,zero);
        else for(int64_t col=0;col<n;col+=8)
            tile4x8(activation_scratch+(row/4)*k*4,packed+(col/4)*k*4,sums+col,y+row*n+col,k,n,valid,zero);
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
