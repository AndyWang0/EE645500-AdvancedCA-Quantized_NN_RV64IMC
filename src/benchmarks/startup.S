.section .text
.globl _start
.type _start, @function

_start:
    la sp, __stack_top
    la t0, tohost
    sd zero, 0(t0)
    la t0, fromhost
    sd zero, 0(t0)
    call main
1:
    j 1b
