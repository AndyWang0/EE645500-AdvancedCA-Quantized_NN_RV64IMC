#ifndef XQ_OPS_H
#define XQ_OPS_H

#include <stdint.h>

/*
 * Emit the project instructions directly with GNU assembler's .insn
 * directive. This keeps the compiler unchanged and describes the
 * read/write operands to the optimizer.
 */
static inline int64_t xqmac8_emit(int64_t accumulator,
                                   uint64_t packed_a,
                                   uint64_t packed_w)
{
    __asm__ volatile (
        ".insn r 0x0b, 0x0, 0x00, %0, %1, %2"
        : "+r" (accumulator)
        : "r" (packed_a), "r" (packed_w));
    return accumulator;
}

static inline int64_t xqrequant8_emit(int64_t accumulator, uint64_t params)
{
    __asm__ volatile (
        ".insn r 0x0b, 0x0, 0x01, %0, %0, %1"
        : "+r" (accumulator)
        : "r" (params));
    return accumulator;
}

static inline uint64_t xq_pack_i8(const int8_t *values, uint32_t count,
                                  uint32_t stride)
{
    uint64_t packed = 0;
    uint32_t lane;

    for (lane = 0; lane < count && lane < 8U; ++lane) {
        packed |= (uint64_t)(uint8_t)values[lane * stride] << (lane * 8U);
    }

    return packed;
}

static inline uint64_t xq_pack_requant_params(int32_t multiplier,
                                              uint8_t shift,
                                              int8_t zero_point)
{
    return (uint64_t)(uint32_t)multiplier |
           ((uint64_t)(shift & 0x3fU) << 32U) |
           ((uint64_t)(uint8_t)zero_point << 38U);
}

#endif
