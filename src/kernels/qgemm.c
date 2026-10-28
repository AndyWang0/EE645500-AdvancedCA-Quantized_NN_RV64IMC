#include "qgemm.h"

static int8_t clamp_int8(__int128 value)
{
    if (value > 127) {
        return 127;
    }
    if (value < -128) {
        return -128;
    }
    return (int8_t)value;
}

int8_t requantize_reference(int64_t accumulator, int32_t multiplier,
                            uint8_t shift, int8_t zero_point)
{
    __int128 product = (__int128)accumulator * (__int128)multiplier;
    __int128 scaled;

    if (shift == 0) {
        scaled = product;
    } else {
        __int128 magnitude = product < 0 ? -product : product;
        __int128 rounded = (magnitude + ((__int128)1 << (shift - 1))) >> shift;
        scaled = product < 0 ? -rounded : rounded;
    }

    return clamp_int8(scaled + (__int128)zero_point);
}

void qgemm_scalar(const int8_t *a, const int8_t *w, const int64_t *bias,
                  int8_t *c, uint32_t m, uint32_t n, uint32_t k,
                  const int32_t *multiplier, const uint8_t *shift,
                  const int8_t *zero_point)
{
    uint32_t i;
    uint32_t j;
    uint32_t r;

    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            uint64_t accumulator_bits = (uint64_t)bias[j];

            for (r = 0; r < k; ++r) {
                int64_t product = (int64_t)a[i * k + r] *
                                  (int64_t)w[r * n + j];
                accumulator_bits += (uint64_t)product;
            }

            c[i * n + j] = requantize_reference(
                (int64_t)accumulator_bits, multiplier[j], shift[j],
                zero_point[j]);
        }
    }
}

void qgemm_custom(const int8_t *a, const int8_t *w, const int64_t *bias,
                  int8_t *c, uint32_t m, uint32_t n, uint32_t k,
                  const int32_t *multiplier, const uint8_t *shift,
                  const int8_t *zero_point)
{
    /*
     * Keep B0 and the custom path comparable until xqmac8/xqrequant8 is
     * available in the toolchain. The interface and output are fixed now.
     */
    qgemm_scalar(a, w, bias, c, m, n, k, multiplier, shift, zero_point);
}
