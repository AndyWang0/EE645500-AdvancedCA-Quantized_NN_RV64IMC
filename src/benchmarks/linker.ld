OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 0x10000000
}

SECTIONS
{
    . = ORIGIN(RAM);

    .text : {
        *(.text.startup)
        *(.text*)
    } > RAM

    .rodata : {
        *(.rodata*)
    } > RAM

    .data : {
        *(.data*)
        *(.sdata*)
        *(.srodata*)
    } > RAM

    .tohost : ALIGN(8) {
        __tohost_start = .;
        tohost = .;
        LONG(0);
        LONG(0);
        fromhost = .;
        LONG(0);
        LONG(0);
        __tohost_end = .;
    } > RAM

    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        __bss_end = .;
    } > RAM

    . = ALIGN(16);
    __stack_top = ORIGIN(RAM) + LENGTH(RAM) - 16;
}
