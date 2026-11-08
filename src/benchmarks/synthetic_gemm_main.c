#include <stdint.h>

#include "../kernels/qgemm.h"
#include "../../model/exported/synthetic/weights.h"
#include "../../model/exported/synthetic/quant_params.h"
#include "../../model/exported/synthetic/test_vectors.h"
#include "baremetal_status.h"

#define BENCHMARK_M MODEL_M
#define BENCHMARK_N MODEL_N
#define BENCHMARK_K MODEL_K

#ifndef BENCHMARK_MODE
#define BENCHMARK_MODE 0
#endif

static int8_t output[BENCHMARK_M * BENCHMARK_N];
static volatile uint64_t result_checksum;

static uint64_t checksum(const int8_t *values, uint32_t count)
{
    uint64_t value = 14695981039346656037ULL;
    uint32_t index;

    for (index = 0; index < count; ++index) {
        value ^= (uint64_t)(uint8_t)values[index];
        value *= 1099511628211ULL;
    }
    return value;
}

int main(void)
{
    uint32_t index;

    __asm__ volatile (".globl benchmark_start\nbenchmark_start:" ::: "memory");
#if BENCHMARK_MODE == 0
    qgemm_scalar(model_input, model_weights, model_bias, output,
                 BENCHMARK_M, BENCHMARK_N, BENCHMARK_K,
                 model_multiplier, model_shift, model_zero_point);
#elif BENCHMARK_MODE == 1
    qgemm_x1(model_input, model_weights, model_bias, output,
             BENCHMARK_M, BENCHMARK_N, BENCHMARK_K,
             model_multiplier, model_shift, model_zero_point);
#elif BENCHMARK_MODE == 2
    qgemm_x2(model_input, model_weights, model_bias, output,
             BENCHMARK_M, BENCHMARK_N, BENCHMARK_K,
             model_multiplier, model_shift, model_zero_point);
#else
#error "BENCHMARK_MODE must be 0 (B0), 1 (X1), or 2 (X2)"
#endif
    __asm__ volatile (".globl benchmark_end\nbenchmark_end:" ::: "memory");

    for (index = 0; index < BENCHMARK_M * BENCHMARK_N; ++index) {
        if (model_reference_output[index] != output[index]) {
            baremetal_finish(1, "EE645500_FAIL synthetic_gemm\n");
        }
    }
    result_checksum = checksum(output, BENCHMARK_M * BENCHMARK_N);
    baremetal_finish(0, "EE645500_PASS synthetic_gemm\n");
}
