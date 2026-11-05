#include <stdint.h>

#include "../kernels/qgemm.h"
#include "../../model/exported/weights.h"
#include "../../model/exported/quant_params.h"

#define B0_M 8U
#define B0_N 16U
#define B0_K 32U

static int8_t b0_input[B0_M * B0_K];
static int8_t b0_output[B0_M * B0_N];
static volatile uint64_t b0_checksum = 0;

static void b0_fill_input(void)
{
    uint32_t idx;

    for (idx = 0; idx < B0_M * B0_K; ++idx) {
        int32_t value = (int32_t)((idx * 17u) % 29u) - 14;
        b0_input[idx] = (int8_t)value;
    }
}

static uint64_t b0_compute_checksum(const int8_t *data, uint32_t count)
{
    uint32_t idx;
    uint64_t checksum = 14695981039346656037ULL;

    for (idx = 0; idx < count; ++idx) {
        checksum ^= (uint64_t)(uint8_t)data[idx];
        checksum *= 1099511628211ULL;
    }

    return checksum;
}

int main(void)
{
    uint32_t idx;

    b0_fill_input();

    for (idx = 0; idx < B0_M * B0_N; ++idx) {
        b0_output[idx] = 0;
    }

    qgemm_scalar(b0_input,
                 model_weights,
                 model_bias,
                 b0_output,
                 B0_M,
                 B0_N,
                 B0_K,
                 model_multiplier,
                 model_shift,
                 model_zero_point);

    b0_checksum = b0_compute_checksum(b0_output, B0_M * B0_N);

    for (;;) {
        __asm__ volatile("nop" ::: "memory");
    }

    return 0;
}
