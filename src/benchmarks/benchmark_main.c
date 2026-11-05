#include <stdint.h>

#include "../kernels/qgemm.h"
#include "../../model/exported/weights.h"
#include "../../model/exported/quant_params.h"

#define BENCHMARK_M 8U
#define BENCHMARK_N 16U
#define BENCHMARK_K 32U

#ifndef BENCHMARK_MODE
#define BENCHMARK_MODE 0
#endif

static int8_t input[BENCHMARK_M * BENCHMARK_K];
static int8_t reference[BENCHMARK_M * BENCHMARK_N];
static int8_t output[BENCHMARK_M * BENCHMARK_N];
static volatile uint64_t result_checksum;

static void fill_input(void)
{
    uint32_t index;

    for (index = 0; index < BENCHMARK_M * BENCHMARK_K; ++index) {
        input[index] = (int8_t)((index * 17U) % 29U - 14);
    }
}

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

__attribute__((noreturn))
static void benchmark_pass(void)
{
    for (;;) {
        __asm__ volatile ("nop" ::: "memory");
    }
}

__attribute__((noreturn))
static void benchmark_fail(void)
{
    for (;;) {
        __asm__ volatile ("c.ebreak");
    }
}

int main(void)
{
    uint32_t index;

    fill_input();
    qgemm_scalar(input, model_weights, model_bias, reference,
                 BENCHMARK_M, BENCHMARK_N, BENCHMARK_K,
                 model_multiplier, model_shift, model_zero_point);
    __asm__ volatile (".globl benchmark_start\nbenchmark_start:");
#if BENCHMARK_MODE == 0
    qgemm_scalar(input, model_weights, model_bias, output,
                 BENCHMARK_M, BENCHMARK_N, BENCHMARK_K,
                 model_multiplier, model_shift, model_zero_point);
#elif BENCHMARK_MODE == 1
    qgemm_x1(input, model_weights, model_bias, output,
             BENCHMARK_M, BENCHMARK_N, BENCHMARK_K,
             model_multiplier, model_shift, model_zero_point);
#else
    qgemm_x2(input, model_weights, model_bias, output,
             BENCHMARK_M, BENCHMARK_N, BENCHMARK_K,
             model_multiplier, model_shift, model_zero_point);
#endif
    __asm__ volatile (".globl benchmark_end\nbenchmark_end:");

    for (index = 0; index < BENCHMARK_M * BENCHMARK_N; ++index) {
        if (reference[index] != output[index]) {
            benchmark_fail();
        }
    }
    result_checksum = checksum(output, BENCHMARK_M * BENCHMARK_N);
    benchmark_pass();
}
