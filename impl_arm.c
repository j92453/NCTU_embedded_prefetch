#ifndef TRANSPOSE_IMPL
#define TRANSPOSE_IMPL
long neon_iteration;

int transpose_verify(int *test_src, int *test_dest, size_t w, size_t h);
void naive_transpose(int *src, int *dst, size_t w, size_t h);
void neon_transpose(int *src, int *dst, size_t w, size_t h);
void neon_prefetch_transpose(int *src, int *dst, size_t w, size_t h);

//verify
int transpose_verify(int *test_src, int *test_dest, size_t w, size_t h)
{
    int *expected  = (int *) malloc(sizeof(int) * w * h);
    int ret;
    naive_transpose(test_src, expected, w, h);
    if(memcmp(test_dest, expected, w*h*sizeof(int)) != 0) {
        ret = 1;
    } else {
        ret = 0;
    }
    free(expected);
    return ret;
}
void naive_transpose(int *src, int *dst, size_t w, size_t h)
{
    //naive_iteration = 0;
    for (size_t x = 0; x < w; x++)
        for (size_t y = 0; y < h; y++) {
            *(dst + x * h + y) = *(src + y * w + x);
        } //naive_iteration +=1;}
}


void neon_transpose(int *src, int *dst, size_t w, size_t h)
{
    for (size_t x = 0; x < w; x += 4) {
        for(size_t y = 0; y < h; y += 4) {
#ifndef _AARCH64
            int32_t * I0 = src + (y + 0) * w + x;
            int32_t * I1 = src + (y + 1) * w + x;
            int32_t * I2 = src + (y + 2) * w + x;
            int32_t * I3 = src + (y + 3) * w + x;
            int32_t * O0 = dst + ((x + 0) * h) + y;
            int32_t * O1 = dst + ((x + 1) * h) + y;
            int32_t * O2 = dst + ((x + 2) * h) + y;
            int32_t * O3 = dst + ((x + 3) * h) + y;

            asm volatile("ld1 {v0.4s}, [%[_I0]]\n"
                         "ld1 {v1.4s}, [%[_I1]]\n"
                         "ld1 {v2.4s}, [%[_I2]]\n"
                         "ld1 {v3.4s}, [%[_I3]]\n"
                         "trn1 v0.4s, v1.4s, v4.4s\n"
                         "trn1 v2.4s, v3.4s, v4.4s\n"
                         "ins v4.d[0], v0.d[1]\n" // d1=v0.d[1], d4=v2.d[0]
                         "ins v0.d[1], v2.d[0]\n" // d3=v1.d[1], d6=v3.d[0]
                         "ins v2.d[0], v4.d[0]\n"
                         "ins v4.d[1], v1.d[1]\n"
                         "ins v1.d[1], v3.d[0]\n"
                         "ins v3.d[0], v4.d[1]\n"
                         "st1 {v0.4s}, [%[_O0]]\n"
                         "st1 {v1.4s}, [%[_O1]]\n"
                         "st1 {v2.4s}, [%[_O2]]\n"
                         "st1 {v3.4s}, [%[_O3]]\n" : :
                         [_I0] "r" (I0),
                         [_I1] "r" (I1),
                         [_I2] "r" (I2),
                         [_I3] "r" (I3),
                         [_O0] "r" (O0),
                         [_O1] "r" (O1),
                         [_O2] "r" (O2),
                         [_O3] "r" (O3) :
                        );
#else
            int32x4_t I0 = vld1q_s32((int32_t *)(src + (y + 0) * w + x));
            int32x4_t I1 = vld1q_s32((int32_t *)(src + (y + 1) * w + x));
            int32x4_t I2 = vld1q_s32((int32_t *)(src + (y + 2) * w + x));
            int32x4_t I3 = vld1q_s32((int32_t *)(src + (y + 3) * w + x));

            vzipq_s32(I0, I1); //I0: T0, I1:T2
            vzipq_s32(I2, I3); //I2: T1, I3:T3

            int32x4_t T0 = vcombine_s32(vget_low_s32(I0), vget_low_s32(I1));//vcombine_s32(low,high)
            int32x4_t T1 = vcombine_s32(vget_high_s32(I0), vget_high_s32(I1));
            int32x4_t T2 = vcombine_s32(vget_low_s32(I2), vget_low_s32(I3));
            int32x4_t T3 = vcombine_s32(vget_high_s32(I2), vget_high_s32(I3));

            vst1q_s32((int32_t *)(dst + ((x + 0) * h) + y), T0);
            vst1q_s32((int32_t *)(dst + ((x + 1) * h) + y), T1);
            vst1q_s32((int32_t *)(dst + ((x + 2) * h) + y), T2);
            vst1q_s32((int32_t *)(dst + ((x + 3) * h) + y), T3);
            i
#endif
        }
    }
}


void neon_prefetch_transpose(int *src, int *dst, size_t w, size_t h)
{
    for (size_t x = 0; x < w; x += 4) {
        for(size_t y = 0; y < h; y += 4) {
#define PFDIST  8
            __builtin_prefetch(src+(y + PFDIST + 0) *w + x);
            __builtin_prefetch(src+(y + PFDIST + 1) *w + x);
            __builtin_prefetch(src+(y + PFDIST + 2) *w + x);
            __builtin_prefetch(src+(y + PFDIST + 3) *w + x);
#ifndef _AARCH64
            int32_t * I0 = src + (y + 0) * w + x;
            int32_t * I1 = src + (y + 1) * w + x;
            int32_t * I2 = src + (y + 2) * w + x;
            int32_t * I3 = src + (y + 3) * w + x;
            int32_t * O0 = dst + ((x + 0) * h) + y;
            int32_t * O1 = dst + ((x + 1) * h) + y;
            int32_t * O2 = dst + ((x + 2) * h) + y;
            int32_t * O3 = dst + ((x + 3) * h) + y;

            asm volatile("ld1 {v0.4s}, [%[_I0]]\n"
                         "ld1 {v1.4s}, [%[_I1]]\n"
                         "ld1 {v2.4s}, [%[_I2]]\n"
                         "ld1 {v3.4s}, [%[_I3]]\n"
                         "trn1 v0.4s, v1.4s, v4.4s\n"
                         "trn1 v2.4s, v3.4s, v4.4s\n"
                         "ins v4.d[0], v0.d[1]\n" // d1=v0.d[1], d4=v2.d[0]
                         "ins v0.d[1], v2.d[0]\n" // d3=v1.d[1], d6=v3.d[0]
                         "ins v2.d[0], v4.d[0]\n"
                         "ins v4.d[1], v1.d[1]\n"
                         "ins v1.d[1], v3.d[0]\n"
                         "ins v3.d[0], v4.d[1]\n"
                         "st1 {v0.4s}, [%[_O0]]\n"
                         "st1 {v1.4s}, [%[_O1]]\n"
                         "st1 {v2.4s}, [%[_O2]]\n"
                         "st1 {v3.4s}, [%[_O3]]\n" : :
                         [_I0] "r" (I0),
                         [_I1] "r" (I1),
                         [_I2] "r" (I2),
                         [_I3] "r" (I3),
                         [_O0] "r" (O0),
                         [_O1] "r" (O1),
                         [_O2] "r" (O2),
                         [_O3] "r" (O3) :
                        );
#else
            int32x4_t I0 = vld1q_s32((int32_t *)(src + (y + 0) * w + x));
            int32x4_t I1 = vld1q_s32((int32_t *)(src + (y + 1) * w + x));
            int32x4_t I2 = vld1q_s32((int32_t *)(src + (y + 2) * w + x));
            int32x4_t I3 = vld1q_s32((int32_t *)(src + (y + 3) * w + x));

            vzipq_s32(I0, I1); //I0: T0, I1:T2
            vzipq_s32(I2, I3); //I2: T1, I3:T3

            int32x4_t T0 = vcombine_s32(vget_low_s32(I0), vget_low_s32(I1));//vcombine_s32(low,high)
            int32x4_t T1 = vcombine_s32(vget_high_s32(I0), vget_high_s32(I1));
            int32x4_t T2 = vcombine_s32(vget_low_s32(I2), vget_low_s32(I3));
            int32x4_t T3 = vcombine_s32(vget_high_s32(I2), vget_high_s32(I3));

            vst1q_s32((int32_t *)(dst + ((x + 0) * h) + y), T0);
            vst1q_s32((int32_t *)(dst + ((x + 1) * h) + y), T1);
            vst1q_s32((int32_t *)(dst + ((x + 2) * h) + y), T2);
            vst1q_s32((int32_t *)(dst + ((x + 3) * h) + y), T3);
            i
#endif
        }
    }
}
#endif /* TRANSPOSE_IMPL */
