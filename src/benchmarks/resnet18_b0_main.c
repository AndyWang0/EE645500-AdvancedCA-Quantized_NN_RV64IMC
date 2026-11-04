#include <stdint.h>

#include "resnet18_input.h"

void resnet18_run(void);
void resnet18_load_input(const int8_t *input, uint32_t elements);
uint64_t resnet18_output_checksum(void);

static volatile uint64_t output_checksum;
extern volatile uint64_t tohost;

int main(void)
{
    resnet18_load_input(resnet18_input, sizeof(resnet18_input));
    __asm__ volatile (".globl benchmark_start\nbenchmark_start:");
    resnet18_run();
    __asm__ volatile (".globl benchmark_end\nbenchmark_end:");
    output_checksum = resnet18_output_checksum();
    tohost = 1;
    for (;;)
        __asm__ volatile ("nop" ::: "memory");
    return 0;
}
