#ifndef MODEL_QUANT_PARAMS_H
#define MODEL_QUANT_PARAMS_H

#include <stdint.h>

static const int32_t model_multiplier[MODEL_N] = { 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1 };
static const uint8_t model_shift[MODEL_N] = { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 };
static const int8_t model_zero_point[MODEL_N] = { -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7 };

#endif
