    .section .rodata.pix_blob, "a"
    .balign 4
    .global pix_blob
pix_blob:
    .incbin "build/pix.bin"
