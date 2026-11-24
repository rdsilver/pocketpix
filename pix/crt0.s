@ Cartridge header and startup. No BIOS calls, no interrupts.
    .section .crt0, "ax"
    .arm
    .global _start
_start:
    b       rom_start
    .fill   156, 1, 0               @ 0x04 boot logo: the console BIOS checks these bytes;
                                    @ left blank here (emulators don't need them)
    .ascii  "POCKET PIX\0\0"         @ 0xA0 title
    .ascii  "PPXE"                  @ 0xAC game code
    .ascii  "00"                    @ 0xB0 maker
    .byte   0x96, 0, 0              @ 0xB2 fixed, unit, device
    .fill   7, 1, 0
    .byte   0                       @ 0xBC version
    .byte   0                       @ 0xBD header checksum (tools/fixrom.py)
    .byte   0, 0
rom_start:
    mov     r0, #0x12               @ IRQ mode stack
    msr     cpsr_c, r0
    ldr     sp, =0x03007FA0
    mov     r0, #0x1F               @ system mode stack
    msr     cpsr_c, r0
    ldr     sp, =0x03007F00
    ldr     r0, =__iwram_lma        @ copy fast-RAM code and data out of ROM
    ldr     r1, =__iwram_start
    ldr     r2, =__iwram_end
1:  cmp     r1, r2
    ldrlo   r3, [r0], #4
    strlo   r3, [r1], #4
    blo     1b
    ldr     r0, =__vcode_lma        @ copy the engine's code into spare video memory (faster than ROM)
    ldr     r1, =__vcode_start
    ldr     r2, =__vcode_end
4:  cmp     r1, r2
    ldrlo   r3, [r0], #4
    strlo   r3, [r1], #4
    blo     4b
    ldr     r1, =__bss_start        @ clear fast-RAM bss (slow-RAM buffers are not cleared)
    ldr     r2, =__bss_end
    mov     r3, #0
2:  cmp     r1, r2
    strlo   r3, [r1], #4
    blo     2b
    ldr     r0, =main
    mov     lr, pc
    bx      r0
3:  b       3b
    .pool
