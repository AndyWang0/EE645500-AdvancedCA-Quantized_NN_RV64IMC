.section .text
.globl _start
.type _start, @function

.equ STACK_TOP, 0x81000000

_start:
    li sp, STACK_TOP
    la t0, tohost
    sd zero, 0(t0)
    la t0, fromhost
    sd zero, 0(t0)
    call main
1:
    j 1b
