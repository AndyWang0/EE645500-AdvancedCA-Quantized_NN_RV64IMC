#include "qgemm.h"

#if defined(__riscv) && (__riscv_xlen == 64)
#include "xq_ops.h"
#endif

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

static void qgemm_packed(const int8_t *a, const int8_t *w, const int64_t *bias,
                         int8_t *c, uint32_t m, uint32_t n, uint32_t k,
                         const int32_t *multiplier, const uint8_t *shift,
                         const int8_t *zero_point, int use_custom_requant)
{
#if defined(__riscv) && (__riscv_xlen == 64)
    uint32_t i;
    uint32_t j;
    uint32_t r;

    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            uint64_t accumulator_bits = (uint64_t)bias[j];

            for (r = 0; r + 8U <= k; r += 8U) {
                uint64_t packed_a = xq_pack_i8(&a[i * k + r], 8U, 1U);
                uint64_t packed_w = xq_pack_i8(&w[r * n + j], 8U, n);

                accumulator_bits = (uint64_t)xqmac8_emit(
                    (int64_t)accumulator_bits, packed_a, packed_w);
            }

            for (; r < k; ++r) {
                int64_t product = (int64_t)a[i * k + r] *
                                  (int64_t)w[r * n + j];
                accumulator_bits += (uint64_t)product;
            }

            if (use_custom_requant != 0) {
                uint64_t params = xq_pack_requant_params(
                    multiplier[j], shift[j], zero_point[j]);
                c[i * n + j] = (int8_t)xqrequant8_emit(
                    (int64_t)accumulator_bits, params);
            } else {
                c[i * n + j] = requantize_reference(
                    (int64_t)accumulator_bits, multiplier[j], shift[j],
                    zero_point[j]);
            }
        }
    }
#else
    /*
     * Keep host correctness tests executable. The RV64 build above is the
     * custom-instruction path; non-RISC-V builds use the same reference.
     */
    (void)use_custom_requant;
    qgemm_scalar(a, w, bias, c, m, n, k, multiplier, shift, zero_point);
#endif
}

void qgemm_x1(const int8_t *a, const int8_t *w, const int64_t *bias,
              int8_t *c, uint32_t m, uint32_t n, uint32_t k,
              const int32_t *multiplier, const uint8_t *shift,
              const int8_t *zero_point)
{
    qgemm_packed(a, w, bias, c, m, n, k, multiplier, shift, zero_point, 0);
}

void qgemm_x2(const int8_t *a, const int8_t *w, const int64_t *bias,
              int8_t *c, uint32_t m, uint32_t n, uint32_t k,
              const int32_t *multiplier, const uint8_t *shift,
              const int8_t *zero_point)
{
    qgemm_packed(a, w, bias, c, m, n, k, multiplier, shift, zero_point, 1);
}

void qgemm_custom(const int8_t *a, const int8_t *w, const int64_t *bias,
                  int8_t *c, uint32_t m, uint32_t n, uint32_t k,
                  const int32_t *multiplier, const uint8_t *shift,
                  const int8_t *zero_point)
{
    qgemm_x2(a, w, bias, c, m, n, k, multiplier, shift, zero_point);
}
