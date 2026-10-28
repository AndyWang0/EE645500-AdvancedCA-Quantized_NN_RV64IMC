#ifndef QGEMM_H
#define QGEMM_H

#include <stdint.h>

/*
 * Computes C = requantize(A * W + bias) for row-major tensors.
 *
 * A has shape M x K, W has shape K x N, and C has shape M x N.
 * Quantization parameters are per output channel and have length N.
 * The project-defined fixed-point format is multiplier / 2^shift.
 */
void qgemm_scalar(const int8_t *a, const int8_t *w, const int64_t *bias,
                  int8_t *c, uint32_t m, uint32_t n, uint32_t k,
                  const int32_t *multiplier, const uint8_t *shift,
                  const int8_t *zero_point);

/*
 * Day 1 compatibility path. The custom implementation will replace this
 * function's inner loop after the custom ISA is available.
 */
void qgemm_custom(const int8_t *a, const int8_t *w, const int64_t *bias,
                  int8_t *c, uint32_t m, uint32_t n, uint32_t k,
                  const int32_t *multiplier, const uint8_t *shift,
                  const int8_t *zero_point);

int8_t requantize_reference(int64_t accumulator, int32_t multiplier,
                            uint8_t shift, int8_t zero_point);

#endif
