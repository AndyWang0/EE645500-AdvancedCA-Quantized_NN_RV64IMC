#include <stdint.h>

#include "../kernels/qgemm.h"
#include "../../model/exported/weights.h"
#include "../../model/exported/quant_params.h"

#define X1_M 8U
#define X1_N 16U
#define X1_K 32U

static int8_t x1_input[X1_M * X1_K];
static int8_t x1_reference[X1_M * X1_N];
static int8_t x1_output[X1_M * X1_N];

__attribute__((noreturn))
static void x1_pass(void)
{
    for (;;) {
        __asm__ volatile ("nop" ::: "memory");
    }
}

__attribute__((noreturn))
static void x1_fail(void)
{
    for (;;) {
        __asm__ volatile ("c.ebreak");
    }
}

static void x1_fill_input(void)
{
    uint32_t idx;

    for (idx = 0; idx < X1_M * X1_K; ++idx) {
        x1_input[idx] = (int8_t)((idx * 17U) % 29U - 14);
    }
}

int main(void)
{
    uint32_t idx;

    x1_fill_input();
    qgemm_scalar(x1_input, model_weights, model_bias, x1_reference,
                 X1_M, X1_N, X1_K, model_multiplier, model_shift,
                 model_zero_point);
    qgemm_custom(x1_input, model_weights, model_bias, x1_output,
                 X1_M, X1_N, X1_K, model_multiplier, model_shift,
                 model_zero_point);

    for (idx = 0; idx < X1_M * X1_N; ++idx) {
        if (x1_reference[idx] != x1_output[idx]) {
            x1_fail();
        }
    }

    x1_pass();
}
