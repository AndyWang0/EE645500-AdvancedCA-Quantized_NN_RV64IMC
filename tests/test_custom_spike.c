#include <stdint.h>

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
    int64_t output;

    __asm__ volatile (
        ".insn r 0x0b, 0x0, 0x01, %0, %1, %2"
        : "=r" (output)
        : "r" (accumulator), "r" (params));
    return output;
}

__attribute__((noreturn))
static void custom_test_pass(void)
{
    for (;;) {
        __asm__ volatile ("nop" ::: "memory");
    }
}

__attribute__((noreturn))
static void custom_test_fail(void)
{
    for (;;) {
        __asm__ volatile ("c.ebreak");
    }
}

int main(void)
{
    const uint64_t packed_a = 0x0101010101010101ULL;
    const uint64_t packed_w = 0x0807060504030201ULL;
    const uint64_t params = 3ULL | (1ULL << 32) | (0xfeULL << 38);
    int64_t accumulator;
    int64_t output;

    accumulator = test_xqmac8(10, packed_a, packed_w);
    output = test_xqrequant8(accumulator, params);

    if (accumulator != 46 || output != 67) {
        custom_test_fail();
    }

    custom_test_pass();
}
