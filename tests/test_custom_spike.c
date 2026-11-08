#include <stdint.h>
#include "../src/benchmarks/baremetal_status.h"

static inline int64_t test_xqmac8(int64_t accumulator,
                                   uint64_t packed_a,
                                   uint64_t packed_w)
{
    __asm__ volatile (
        ".insn r 0x0b, 0x0, 0x00, %0, %1, %2"
        : "+r" (accumulator)
        : "r" (packed_a), "r" (packed_w));
    return accumulator;
}

static inline int64_t test_xqrequant8(int64_t accumulator, uint64_t params)
{
    __asm__ volatile (
        ".insn r 0x0b, 0x0, 0x01, %0, %0, %1"
        : "+r" (accumulator)
        : "r" (params));
    return accumulator;
}

static int64_t reference_xqmac8(int64_t accumulator,
                                uint64_t packed_a,
                                uint64_t packed_w)
{
    uint64_t accumulator_bits = (uint64_t)accumulator;
    uint32_t lane;

    for (lane = 0; lane < 8U; ++lane) {
        int64_t a = (int64_t)((packed_a >> (lane * 8U)) & 0xffU);
        int64_t w = (int64_t)((packed_w >> (lane * 8U)) & 0xffU);

        if ((a & 0x80) != 0) {
            a -= 0x100;
        }
        if ((w & 0x80) != 0) {
            w -= 0x100;
        }
        accumulator_bits += (uint64_t)(a * w);
    }

    return (int64_t)accumulator_bits;
}

static int8_t reference_xqrequant8(int64_t accumulator,
                                    int32_t multiplier,
                                    uint8_t shift,
                                    int8_t zero_point)
{
    __int128 product = (__int128)accumulator * (__int128)multiplier;
    __int128 scaled;
    __int128 quantized;

    if (shift == 0) {
        scaled = product;
    } else {
        __int128 magnitude = product < 0 ? -product : product;
        __int128 rounded =
            (magnitude + ((__int128)1 << (shift - 1))) >> shift;
        scaled = product < 0 ? -rounded : rounded;
    }

    quantized = scaled + (__int128)zero_point;
    if (quantized > 127) {
        return 127;
    }
    if (quantized < -128) {
        return -128;
    }
    return (int8_t)quantized;
}

static uint64_t pack_params(int32_t multiplier, uint8_t shift,
                            int8_t zero_point)
{
    return (uint64_t)(uint32_t)multiplier |
           ((uint64_t)shift << 32U) |
           ((uint64_t)(uint8_t)zero_point << 38U);
}

static uint64_t next_random(uint64_t *state)
{
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return *state;
}

static int test_xqmac8_cases(void)
{
    static const uint64_t packed_a[] = {
        0x0101010101010101ULL,
        0x8081828384858687ULL,
        0x7f0001fe10203040ULL
    };
    static const uint64_t packed_w[] = {
        0x0807060504030201ULL,
        0x0102030405060708ULL,
        0xff807f0110203040ULL
    };
    static const int64_t accumulators[] = { 10, -123456789, INT64_MAX - 100 };
    uint64_t state = 0x2d5a5a1234567890ULL;
    uint32_t index;

    for (index = 0; index < 3U; ++index) {
        int64_t expected = reference_xqmac8(
            accumulators[index], packed_a[index], packed_w[index]);
        if (test_xqmac8(accumulators[index], packed_a[index], packed_w[index])
            != expected) {
            return 0;
        }
    }

    for (index = 0; index < 64U; ++index) {
        uint64_t packed_a_random = next_random(&state);
        uint64_t packed_w_random = next_random(&state);
        int64_t accumulator = (int64_t)next_random(&state);
        int64_t expected = reference_xqmac8(
            accumulator, packed_a_random, packed_w_random);

        if (test_xqmac8(accumulator, packed_a_random, packed_w_random)
            != expected) {
            return 0;
        }
    }

    return 1;
}

static int test_xqrequant8_cases(void)
{
    static const int64_t accumulators[] = {
        0, 1, -1, 3, -3, 127, -128, INT64_MAX, INT64_MIN
    };
    static const int32_t multipliers[] = { 0, 1, -1, 3, -3, INT32_MAX };
    static const uint8_t shifts[] = { 0, 1, 2, 6, 31, 63 };
    static const int8_t zero_points[] = { 0, -128, 127, -7 };
    uint64_t state = 0x9b1d4e6f10293847ULL;
    uint32_t index;

    for (index = 0; index < 9U; ++index) {
        uint32_t parameter_index = index % 6U;
        int8_t expected = reference_xqrequant8(
            accumulators[index], multipliers[parameter_index],
            shifts[parameter_index], zero_points[index % 4U]);
        int64_t actual = test_xqrequant8(
            accumulators[index],
            pack_params(multipliers[parameter_index], shifts[parameter_index],
                        zero_points[index % 4U]));

        if (actual != (int64_t)expected) {
            return 0;
        }
    }

    for (index = 0; index < 64U; ++index) {
        int64_t accumulator = (int64_t)next_random(&state);
        int32_t multiplier = (int32_t)next_random(&state);
        uint8_t shift = (uint8_t)(next_random(&state) % 64U);
        int8_t zero_point = (int8_t)next_random(&state);
        int8_t expected = reference_xqrequant8(
            accumulator, multiplier, shift, zero_point);
        int64_t actual = test_xqrequant8(
            accumulator, pack_params(multiplier, shift, zero_point));

        if (actual != (int64_t)expected) {
            return 0;
        }
    }

    return 1;
}

int main(void)
{
    if (!test_xqmac8_cases() || !test_xqrequant8_cases()) {
        baremetal_finish(1, "EE645500_FAIL instruction_test\n");
    }

    baremetal_finish(0, "EE645500_PASS instruction_test\n");
}
