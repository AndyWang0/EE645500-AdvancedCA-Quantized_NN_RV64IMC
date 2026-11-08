#ifndef BAREMETAL_STATUS_H
#define BAREMETAL_STATUS_H

#include <stdint.h>

extern volatile uint64_t tohost;
extern volatile uint64_t fromhost;

/* Spike's HTIF syscall proxy, followed by an explicit exit (no pk required). */
__attribute__((noreturn, noinline, noclone))
static void baremetal_finish(uint32_t code, const char *message)
{
    static volatile uint64_t request[8] __attribute__((aligned(8)));
    uint32_t length = 0;
    while (message[length] != '\0')
        ++length;
    request[0] = 64; /* write */
    request[1] = 1;  /* stdout */
    request[2] = (uintptr_t)message;
    request[3] = length;
    fromhost = 0;
    __asm__ volatile ("fence rw, rw" ::: "memory");
    tohost = (uintptr_t)request;
    while (fromhost == 0)
        __asm__ volatile ("nop" ::: "memory");
    __asm__ volatile ("fence rw, rw" ::: "memory");
    __asm__ volatile (".globl workload_exit\nworkload_exit:" ::: "memory");
    tohost = ((uint64_t)code << 1U) | 1U;
    for (;;)
        __asm__ volatile ("nop" ::: "memory");
}

#endif
