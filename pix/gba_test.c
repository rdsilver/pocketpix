/* Test ROM: paint one picture through every model with a fixed seed and prompt; leave the pictures, cycle counts
 * and the profile in memory for pix/test_gba.py. */
#include "pix.h"
struct Dbg { uint32_t magic, done, stage, cells, cyc[3], prof[3][PP_N]; uint8_t img[64 * 64]; uint32_t profa[3][11]; };
volatile struct Dbg dbg __attribute__((section(".bss.dbg")));
void *memset(void *d, int c, unsigned long n) { uint8_t *p = d; while (n--) *p++ = (uint8_t)c; return d; }
void *memcpy(void *d, const void *s, unsigned long n) { uint8_t *p = d; const uint8_t *q = s; while (n--) *p++ = *q++; return d; }
static uint32_t now(void) {
    volatile uint16_t *t = (volatile uint16_t *)0x04000100;
    uint16_t h = t[2], l = t[0];
    if (h != t[2]) { h = t[2]; l = t[0]; }
    return ((uint32_t)h << 16) | l;
}
static uint8_t th[32 * 32] __attribute__((section(".bss.ewram")));
#ifndef TEST_BASE
#define TEST_BASE 0
#endif
#ifndef TEST_NM
#define TEST_NM 3
#endif
#ifndef TEST_CELLS
#define TEST_CELLS 100000
#endif
/* Slow RAM with one wait state instead of two (works on GBA and GBA SP; checked with a pattern test and undone
 * if a word reads back wrong, as it may on a Game Boy Micro). Nothing of value is in slow RAM yet. */
static void fast_ewram(void) {
    volatile uint32_t *ctl = (volatile uint32_t *)0x04000800, *ram = (volatile uint32_t *)0x02000000;
    *ctl = 0x0E000020;
    int bad = 0;
    for (int pass = 0; pass < 2 && !bad; pass++) {
        uint32_t k = pass ? 0xA5A5A5A5u : 0x5A5A5A5Au;
        for (uint32_t i = 0; i < 65536; i += 61) ram[i] = (i * 2654435761u) ^ k;
        for (uint32_t i = 0; i < 65536; i += 61) if (ram[i] != ((i * 2654435761u) ^ k)) bad = 1;
    }
    if (bad) *ctl = 0x0D000020;
}
int main(void) {
    *(volatile uint16_t *)0x04000204 = 0x4317;          /* fastest cartridge timing, prefetch on */
    fast_ewram();
    *(volatile uint16_t *)0x04000000 = 0x0080;          /* screen off */
    *(volatile uint16_t *)0x04000106 = 0x0084;
    *(volatile uint16_t *)0x04000102 = 0x0080;          /* one tick per CPU cycle, 32 bits */
    static const uint16_t words[8] = { TEST_WORDS };
    uint32_t rng = 12345;
    dbg.done = 0;
    for (int m = 0; m < TEST_NM; m++) {
        for (int i = 0; i < PP_N; i++) pix_prof[i] = 0;
        dbg.stage = m;
#ifdef PIX_FORCE
        { extern const uint8_t force_pix[]; extern const uint8_t *pix_force; pix_force = force_pix + (m == 0 ? 0 : m == 1 ? 256 : 1280); }     /* (test_gba.py force=1) */
#endif
        if (pix_begin(m ? m : TEST_BASE, words, th, m ? 256 : TEST_GUIDE, m ? 366 : 284, &rng)) { dbg.magic = 0xDEAD0000u + m; dbg.done = 1; for (;;) {} }
        uint32_t t0 = now(), total = 0;
        int n = 0;
        while (n < TEST_CELLS) {
            int more = pix_cell(); n++;
            uint32_t t1 = now(); total += t1 - t0; t0 = t1;      /* a picture takes longer than the 32-bit counter in rare cases; add as we go */
            dbg.cells = n;
            if (!more) break;
        }
        dbg.cyc[m] = total >> 8;                             /* in units of 256 cycles */
        for (int i = 0; i < PP_N; i++) dbg.prof[m][i] = pix_prof[i] >> 8;
#ifdef PIX_PROFA
        { extern uint32_t pix_profa[11]; for (int i = 0; i < 11; i++) { dbg.profa[m][i] = pix_profa[i] >> 8; pix_profa[i] = 0; } }
#endif
        int S = pix_info(m ? m : TEST_BASE, MF_S);
        if (m + 1 < 3) for (int i = 0; i < S * S; i++) th[i] = pix_img[i];
        else for (int i = 0; i < S * S; i++) dbg.img[i] = pix_img[i];
    }
    dbg.magic = 0x50495831; dbg.done = 1;
    for (;;) {}
}
