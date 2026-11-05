#include <stdint.h>
#include <stdio.h>

#include "../src/kernels/qgemm.h"

#define M 8
#define N 16
#define K 32

static int check_requantization(void)
{
    static const struct {
        int64_t accumulator;
        int32_t multiplier;
        uint8_t shift;
        int8_t zero_point;
        int8_t expected;
    } cases[] = {
        {0, 0, 0, 0, 0},
        {3, 1, 1, 0, 2},
        {-3, 1, 1, 0, -2},
        {5, 1, 1, 0, 3},
        {-5, 1, 1, 0, -3},
        {1000, 1, 0, 0, 127},
        {-1000, 1, 0, 0, -128},
        {10, 1, 0, 120, 127},
        {-10, 1, 0, -120, -128},
        {1, 1, 63, 0, 0},
    };
    uint32_t i;

    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        int8_t actual = requantize_reference(
            cases[i].accumulator, cases[i].multiplier, cases[i].shift,
            cases[i].zero_point);
        if (actual != cases[i].expected) {
            printf("requant case %u: expected %d, got %d\n", i,
                   cases[i].expected, actual);
            return 1;
        }
    }
    return 0;
}

int main(void)
{
    static int8_t a[M * K];
    static int8_t w[K * N];
    static int64_t bias[N];
    static int32_t multiplier[N];
    static uint8_t shift[N];
    static int8_t zero_point[N];
    static int8_t scalar_output[M * N];
    static int8_t custom_output[M * N];
    uint32_t i;
    uint32_t j;

    for (i = 0; i < M * K; ++i) {
        a[i] = (int8_t)((i * 13 + 7) % 31 - 15);
    }
    for (i = 0; i < K * N; ++i) {
        w[i] = (int8_t)((i * 11 + 3) % 23 - 11);
    }
    for (j = 0; j < N; ++j) {
        bias[j] = (int64_t)(j * 17) - 100;
        multiplier[j] = (j % 3 == 0) ? 3 : 1;
        shift[j] = (j % 3 == 0) ? 2 : 1;
        zero_point[j] = (int8_t)(j - 8);
    }

    if (check_requantization() != 0) {
        return 1;
    }

    qgemm_scalar(a, w, bias, scalar_output, M, N, K, multiplier, shift,
                 zero_point);
    qgemm_custom(a, w, bias, custom_output, M, N, K, multiplier, shift,
                 zero_point);

    for (i = 0; i < M * N; ++i) {
        if (scalar_output[i] != custom_output[i]) {
            printf("output[%u]: expected %d, got %d\n", i,
                   scalar_output[i], custom_output[i]);
            return 1;
        }
    }

    printf("qgemm scalar/custom comparison passed (%u outputs)\n", M * N);
    return 0;
}
