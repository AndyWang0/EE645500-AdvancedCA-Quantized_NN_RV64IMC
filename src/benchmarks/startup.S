.section .text
.globl _start
.type _start, @function

.equ STACK_TOP, 0x81000000

_start:
    li sp, STACK_TOP
    call main
1:
    j 1b
