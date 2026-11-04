#include <stdint.h>

#include "resnet18_graph.h"
#include "resnet18_input.h"
#include "resnet18_reference_tensors.h"

extern volatile uint64_t tohost;
extern volatile uint64_t fromhost;

void resnet18_run(void);
void resnet18_load_input(const int8_t *input, uint32_t elements);

volatile uint32_t resnet18_validation_status;
volatile uint32_t resnet18_validation_tensor;
volatile uint32_t resnet18_validation_mismatches;
volatile uint32_t resnet18_validation_first_index;
volatile int8_t resnet18_validation_expected;
volatile int8_t resnet18_validation_actual;
static volatile uint64_t validation_syscall[8] __attribute__((aligned(8)));
static char validation_message[128];

#define REFERENCE_CASE(id) \
    case id: reference = resnet18_reference_tensor_##id; break

static const int8_t *reference_tensor(uint32_t tensor)
{
    const int8_t *reference = 0;
    switch (tensor) {
        REFERENCE_CASE(0); REFERENCE_CASE(1); REFERENCE_CASE(2);
        REFERENCE_CASE(3); REFERENCE_CASE(4); REFERENCE_CASE(5);
        REFERENCE_CASE(6); REFERENCE_CASE(7); REFERENCE_CASE(8);
        REFERENCE_CASE(9); REFERENCE_CASE(10); REFERENCE_CASE(32);
        REFERENCE_CASE(33); REFERENCE_CASE(34); REFERENCE_CASE(35);
        REFERENCE_CASE(41); REFERENCE_CASE(42); REFERENCE_CASE(65);
        REFERENCE_CASE(66); REFERENCE_CASE(67); REFERENCE_CASE(68);
        REFERENCE_CASE(69); REFERENCE_CASE(70); REFERENCE_CASE(71);
        REFERENCE_CASE(72); REFERENCE_CASE(73); REFERENCE_CASE(74);
        REFERENCE_CASE(75); REFERENCE_CASE(76); REFERENCE_CASE(77);
        REFERENCE_CASE(78); REFERENCE_CASE(79); REFERENCE_CASE(80);
        REFERENCE_CASE(81); REFERENCE_CASE(82); REFERENCE_CASE(83);
        REFERENCE_CASE(84); REFERENCE_CASE(85);
    default:
        break;
    }
    return reference;
}

static void report_failure(uint32_t tensor, uint32_t mismatches,
                           uint32_t first_index, int8_t expected,
                           int8_t actual)
{
    resnet18_validation_status = 0x1000U + tensor;
    resnet18_validation_tensor = tensor;
    resnet18_validation_mismatches = mismatches;
    resnet18_validation_first_index = first_index;
    resnet18_validation_expected = expected;
    resnet18_validation_actual = actual;
    validation_message[0] = 'R';
    validation_message[1] = 'E';
    validation_message[2] = 'S';
    validation_message[3] = 'N';
    validation_message[4] = 'E';
    validation_message[5] = 'T';
    validation_message[6] = '1';
    validation_message[7] = '8';
    validation_message[8] = '_';
    validation_message[9] = 'F';
    validation_message[10] = 'A';
    validation_message[11] = 'I';
    validation_message[12] = 'L';
    validation_message[13] = ' ';
    {
        uint32_t position = 14;
        uint32_t values[] = {tensor, mismatches, first_index,
                             (uint8_t)expected, (uint8_t)actual};
        const char *names[] = {"tensor", "mismatches", "first_index",
                               "expected", "actual"};
        uint32_t field;
        for (field = 0; field < 5; ++field) {
            uint32_t digit;
            const char *name = names[field];
            while (*name)
                validation_message[position++] = *name++;
            validation_message[position++] = '=';
            validation_message[position++] = '0';
            validation_message[position++] = 'x';
            for (digit = 8; digit-- > 0;)
                validation_message[position++] =
                    "0123456789abcdef"[(values[field] >> (digit * 4)) & 0xf];
            validation_message[position++] = field == 4 ? '\n' : ' ';
        }
        validation_syscall[0] = 64;
        validation_syscall[1] = 1;
        validation_syscall[2] = (uintptr_t)validation_message;
        validation_syscall[3] = position;
        fromhost = 0;
        tohost = (uintptr_t)validation_syscall;
        while (fromhost == 0)
            __asm__ volatile ("nop" ::: "memory");
    }
    tohost = ((uint64_t)(2U + tensor) << 1) | 1ULL;
    for (;;)
        __asm__ volatile ("nop" ::: "memory");
}

static void compare_tensor(uint32_t tensor)
{
    const int8_t *reference = reference_tensor(tensor);
    uint32_t elements = resnet18_tensors[tensor].elements;
    uint32_t index;
    uint32_t mismatches = 0;
    uint32_t first = 0xffffffffU;
    int8_t first_expected = 0;
    int8_t first_actual = 0;

    if (!reference || !resnet18_tensors[tensor].data)
        return;
    for (index = 0; index < elements; ++index) {
        int8_t actual = resnet18_tensors[tensor].data[index];
        if (actual != reference[index]) {
            if (first == 0xffffffffU) {
                first = index;
                first_expected = reference[index];
                first_actual = actual;
            }
            ++mismatches;
        }
    }
    if (mismatches)
        report_failure(tensor, mismatches, first, first_expected, first_actual);
}

int main(void)
{
    uint32_t tensor;
    resnet18_load_input(resnet18_input, sizeof(resnet18_input));
    resnet18_run();
    for (tensor = 0; tensor < resnet18_tensor_count; ++tensor)
        compare_tensor(tensor);

    resnet18_validation_status = 1;
    validation_syscall[0] = 93;
    validation_syscall[1] = 0;
    fromhost = 0;
    tohost = (uintptr_t)validation_syscall;
    for (;;)
        __asm__ volatile ("nop" ::: "memory");
}
