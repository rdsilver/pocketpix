ENTRY(_start)
MEMORY {
    rom   (rx)  : ORIGIN = 0x08000000, LENGTH = 32M
    ewram (rw)  : ORIGIN = 0x02000000, LENGTH = 256K
    iwram (rwx) : ORIGIN = 0x03000000, LENGTH = 0x7900   /* leaves 1.5 KB of stack below 0x03007F00 (0.8 KB used, measured) */
    /* video memory the mode 4 screen (page 0, 0x06000000-0x06009600) does not use; 32-bit access costs 2 cycles.
     * 0x0600A000-0x06012000 is the engine's word arena (pix.c). Only 16- and 32-bit data may live here. */
    vram0 (rw)  : ORIGIN = 0x06009600, LENGTH = 0x0A00
    vram1 (rw)  : ORIGIN = 0x06010C00, LENGTH = 0x1400   /* shared with the arena's top 5 KB: only for passes that leave it free (VRAM1_WORDS in pix.c) */
    vram2 (rwx) : ORIGIN = 0x06012000, LENGTH = 0x6000
}
SECTIONS {
    .text : {
        KEEP(*(.crt0))
        *(EXCLUDE_FILE(*pix_engine.o) .text EXCLUDE_FILE(*pix_engine.o) .text.*)
        *(.romcode .romcode.*)
        *(.glue_7 .glue_7t .vfp11_veneer .v4_bx)
        *(.rodata .rodata.*)
        . = ALIGN(4);
    } > rom
    .vcode : {
        __vcode_start = .;
        *pix_engine.o(.text .text.*)
        *(.vram.text)
        . = ALIGN(4);
        __vcode_end = .;
    } > vram2 AT> rom
    __vcode_lma = LOADADDR(.vcode);
    .vram2 (NOLOAD) : { *(.bss.vram2 .bss.vram2.*) . = ALIGN(4); } > vram2
    .vram0 (NOLOAD) : { *(.bss.vram0 .bss.vram0.*) . = ALIGN(4); } > vram0
    .vram1 (NOLOAD) : { *(.bss.vram1 .bss.vram1.*) . = ALIGN(4); } > vram1
    .iwram : {
        __iwram_start = .;
        *(.iwram.text .iwram.text.*)
        *(.data .data.*)
        . = ALIGN(4);
        __iwram_end = .;
    } > iwram AT> rom
    __iwram_lma = LOADADDR(.iwram);
    .ewram (NOLOAD) : {
        KEEP(*(.bss.dbg))
        *(.bss.ewram .bss.ewram.*)
        . = ALIGN(4);
    } > ewram
    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > iwram
    /DISCARD/ : { *(.ARM.exidx*) *(.ARM.extab*) *(.comment) *(.note*) }
}
