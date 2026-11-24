/* Pocket Pix: type a prompt, watch the cartridge paint a 64x64 photo. Bitmap mode 4 with the photo palette. */
#include "pix.h"
#include "font.h"
#define REG16(a) (*(volatile uint16_t *)(a))
#define VRAM ((volatile uint16_t *)0x06000000)
#define KEYS (~REG16(0x04000130) & 0x3FF)
enum { K_A = 1, K_B = 2, K_SELECT = 4, K_START = 8, K_RIGHT = 16, K_LEFT = 32, K_UP = 64, K_DOWN = 128, K_R = 256, K_L = 512 };
#define PX 8
#define PY 24
void *memset(void *d, int c, unsigned long n) { uint8_t *p = d; while (n--) *p++ = (uint8_t)c; return d; }
void *memcpy(void *d, const void *s, unsigned long n) { uint8_t *p = d; const uint8_t *q = s; while (n--) *p++ = *q++; return d; }

static void rect(int x, int y, int w, int h, int c) {          /* x and w even */
    uint16_t v = (uint16_t)(c | c << 8);
    for (int j = 0; j < h; j++) { volatile uint16_t *p = VRAM + ((y + j) * 240 + x) / 2; for (int i = 0; i < w / 2; i++) p[i] = v; }
}
static void glyph(int x, int y, int ch, int fg, int bg) {      /* x even */
    const unsigned char *g = FONT[(ch < 32 || ch > 127 ? '?' : ch) - 32];
    for (int j = 0; j < 8; j++) {
        volatile uint16_t *p = VRAM + ((y + j) * 240 + x) / 2;
        int b = g[j];
        for (int i = 0; i < 4; i++, b >>= 2) p[i] = (uint16_t)((b & 1 ? fg : bg) | (b & 2 ? fg : bg) << 8);
    }
}
static int text(int x, int y, const char *s, int fg, int bg) { while (*s) { glyph(x, y, *s++, fg, bg); x += 8; } return x; }
static void textn(int x, int y, const char *s, int n, int fg, int bg) {      /* exactly n cells, padded with spaces */
    for (int i = 0; i < n; i++) { glyph(x + i * 8, y, *s ? *s : ' ', fg, bg); if (*s) s++; }
}
static void num2(char *o, int v) { o[0] = (char)('0' + v / 10 % 10); o[1] = (char)('0' + v % 10); }

static const char *wtext(int id) { return (const char *)(pix_blob + O_WTEXT + ((const uint32_t *)(pix_blob + O_WOFF))[id]); }
static int starts(const char *w, const char *pre, int n) { for (int i = 0; i < n; i++) if (w[i] != pre[i]) return w[i] < pre[i] ? -1 : 1; return 0; }
static int first_match(const char *pre, int n) {              /* first sorted word that is not before the prefix */
    const uint16_t *so = (const uint16_t *)(pix_blob + O_WSORT);
    int lo = 0, hi = PIX_NSORT;
    while (lo < hi) { int m = (lo + hi) / 2; if (starts(wtext(so[m]), pre, n) < 0) lo = m + 1; else hi = m; }
    return lo;
}
static int is_colour(const char *w) {
    static const char *const cs[] = { "white", "black", "red", "blue", "green", "yellow", "orange", "brown", "pink", "purple", "grey", "gray", "silver", "gold", "golden", "dark", "bright", "light", "beige", 0 };
    for (int i = 0; cs[i]; i++) { const char *a = cs[i], *b = w; while (*a && *a == *b) { a++; b++; } if (!*a && !*b) return 1; }
    return 0;
}

static uint16_t words[8]; static int nwords, base_sel;         /* base_sel 0 = objects, 1 = scenes */
static char pre[12]; static int npre, cand, kx, ky;
static volatile int cancel;
static uint16_t prev_keys;
static int pressed(void) { uint16_t k = (uint16_t)KEYS, e = (uint16_t)(k & ~prev_keys); prev_keys = k; return e; }
static void wait_frame(void) { while (REG16(0x04000006) >= 160) {} while (REG16(0x04000006) < 160) {} }
static void poll(void) { if (KEYS & K_B) cancel = 1; }
static uint32_t seconds(void) { uint16_t h = REG16(0x0400010C), l = REG16(0x04000108); if (h != REG16(0x0400010C)) { h = REG16(0x0400010C); l = REG16(0x04000108); } return (((uint32_t)h << 16) | l) >> 14; }

static int ncand(void) {
    if (!npre) return 0;
    const uint16_t *so = (const uint16_t *)(pix_blob + O_WSORT);
    int f = first_match(pre, npre), n = 0;
    while (f + n < PIX_NSORT && n < 200 && !starts(wtext(so[f + n]), pre, npre)) n++;
    return n;
}
static void draw_prompt_line(int y) {
    char buf[29]; int n = 0;
    for (int i = 0; i < nwords && n < 27; i++) { const char *w = wtext(words[i]); if (i && n < 27) buf[n++] = ' '; while (*w && n < 28) buf[n++] = *w++; }
    buf[n] = 0;
    textn(8, y, n ? buf : "(empty: a random photo)", 28, n ? PIX_C_FG : PIX_C_DIM, PIX_C_BG);
}
/* keyboard: two rows of letters and a row of four buttons (kb: 0 add the word, 1 delete, 2 paint, 3 gallery) */
static int kb;
static void draw_entry(void) {
    const uint16_t *so = (const uint16_t *)(pix_blob + O_WSORT);
    draw_prompt_line(30);
    char line[14]; int i;
    for (i = 0; i < npre; i++) line[i] = pre[i];
    line[i++] = '_'; line[i] = 0;
    textn(8, 56, line, 13, PIX_C_ACC, PIX_C_BG);
    int f = npre ? first_match(pre, npre) : PIX_NSORT, shown = 0;
    for (int r = 0; r < 4; r++) {
        int k = f + (cand > 3 ? cand - 3 : 0) + r;
        const char *w = (npre && k < PIX_NSORT && !starts(wtext(so[k]), pre, npre)) ? wtext(so[k]) : "";
        int sel = *w && (k - f) == cand;
        textn(128, 44 + r * 10, w, 13, sel ? PIX_C_BG : PIX_C_FG, sel ? PIX_C_ACC : PIX_C_BG);
        shown += *w != 0;
    }
    if (npre && !shown) textn(128, 44, "(not a word)", 13, PIX_C_DIM, PIX_C_BG);
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 13; c++) { int sel = r == ky && c == kx; glyph(16 + c * 16, 88 + r * 14, 'a' + r * 13 + c, sel ? PIX_C_BG : PIX_C_FG, sel ? PIX_C_ACC : PIX_C_PANEL); }
    static const char *const bt[4] = { " ADD ", " DEL ", " PAINT ", " GALLERY " };     /* ADD is dim when no word is highlighted */
    static const uint8_t bx[4] = { 8, 54, 100, 160 }, bw[4] = { 5, 5, 7, 9 };
    int can = shown && nwords < 8;
    for (int b = 0; b < 4; b++) {
        int s = ky == 2 && kb == b;
        textn(bx[b], 118, bt[b], bw[b], s ? PIX_C_BG : b == 0 && !can ? PIX_C_DIM : b == 2 ? PIX_C_ACC : PIX_C_FG, s ? PIX_C_ACC : PIX_C_PANEL);
    }
    textn(8, 148, base_sel ? "START paint   SELECT scenes" : "START paint   SELECT objects", 28, PIX_C_DIM, PIX_C_BG);
}
static void entry_screen(void) {
    rect(0, 0, 240, 160, PIX_C_BG);
    text(8, 6, "POCKET PIX", PIX_C_ACC, PIX_C_BG);
    text(8, 20, "Prompt so far", PIX_C_DIM, PIX_C_BG);
    text(8, 46, "Next word", PIX_C_DIM, PIX_C_BG);
    rect(8, 84, 224, 46, PIX_C_PANEL);
    text(8, 136, "A press  B delete  L/R pick", PIX_C_DIM, PIX_C_BG);
    draw_entry();
}
static void add_word(void) {                                  /* the highlighted match joins the prompt */
    if (!npre || !ncand() || nwords >= 8) return;
    const uint16_t *so = (const uint16_t *)(pix_blob + O_WSORT);
    uint16_t id = so[first_match(pre, npre) + cand];
    int dup = 0;
    for (int i = 0; i < nwords; i++) dup |= words[i] == id;
    if (!dup) words[nwords++] = id;
    npre = 0; cand = 0;
}
static void del(void) { if (npre) npre--; else if (nwords) nwords--; cand = 0; }
static int gallery(void);
/* returns when the player asks for a picture */
static void entry(void) {
    entry_screen();
    for (;;) {
        wait_frame();
        int e = pressed(), ch = 0;
        if (!e) continue;
        if (e & K_LEFT) { if (ky == 2) kb = (kb + 3) % 4; else kx = (kx + 12) % 13; ch = 1; }
        if (e & K_RIGHT) { if (ky == 2) kb = (kb + 1) % 4; else kx = (kx + 1) % 13; ch = 1; }
        if (e & K_UP) { ky = (ky + 2) % 3; ch = 1; }
        if (e & K_DOWN) { ky = (ky + 1) % 3; ch = 1; }
        if (e & K_A) {
            if (ky < 2) { if (npre < 11) { pre[npre++] = (char)('a' + ky * 13 + kx); cand = 0; } }
            else if (kb == 0) { add_word(); if (!npre) ky = 0; }
            else if (kb == 1) del();
            else if (kb == 2) { add_word(); return; }
            else { if (gallery()) kb = 2; entry_screen(); continue; }     /* with a prompt from the gallery, PAINT is next */
            ch = 1;
        }
        if (e & K_B) { del(); ch = 1; }
        if (e & K_R) { int n = ncand(); if (n) cand = (cand + 1) % n; ch = 1; }
        if (e & K_L) { int n = ncand(); if (n) cand = (cand + n - 1) % n; ch = 1; }
        if (e & K_SELECT) { base_sel ^= 1; ch = 1; }
        if (e & K_START) { add_word(); return; }
        if (ch) draw_entry();
    }
}

static uint8_t th[32 * 32] __attribute__((section(".bss.ewram")));
static void status(const char *s, int stage) {
    char b[13]; int i = 0;
    while (s[i] && i < 12) { b[i] = s[i]; i++; }
    b[i] = 0;
    textn(144, 88, b, 11, PIX_C_FG, PIX_C_BG);
    if (stage >= 0) { char p[] = "pass 1 of 3"; p[5] = (char)('1' + stage); textn(144, 100, p, 11, PIX_C_DIM, PIX_C_BG); }
}
static void show_time(void) {
    uint32_t s = seconds(); char t[6]; t[0] = (char)('0' + s / 60 % 10); t[1] = ':'; num2(t + 2, (int)(s % 60)); t[4] = 0;
    textn(144, 116, t, 5, PIX_C_ACC, PIX_C_BG);
}
/* paint one picture; returns 0 if the player stopped it */
static int paint(uint32_t seed) {
    uint16_t w8[8] = { 0 }; int n = 0;
    for (int pass = 0; pass < 2; pass++)                      /* the subject first, colours after, as in training */
        for (int i = 0; i < nwords; i++) if (is_colour(wtext(words[i])) == pass) w8[n++] = words[i];
    rect(0, 0, 240, 160, PIX_C_BG);
    text(8, 6, "POCKET PIX", PIX_C_ACC, PIX_C_BG);
    rect(PX - 2, PY - 2, 132, 132, PIX_C_PANEL);
    rect(PX, PY, 128, 128, PIX_C_BG);
    for (int i = 0; i < nwords && i < 6; i++) textn(144, 24 + i * 10, wtext(words[i]), 11, PIX_C_FG, PIX_C_BG);
    if (!nwords) text(144, 24, "random", PIX_C_DIM, PIX_C_BG);
    text(144, 146, "B stop", PIX_C_DIM, PIX_C_BG);
    REG16(0x0400010A) = 0; REG16(0x0400010E) = 0;              /* seconds counter: timer 2 at 16,384 Hz, timer 3 counts its overflows */
    REG16(0x04000108) = 0; REG16(0x0400010C) = 0;
    REG16(0x0400010E) = 0x0084; REG16(0x0400010A) = 0x0083;
#ifdef GAL_TEST                                               /* test builds only: a made-up picture at once, numbered in its first byte */
    static int made;
    for (int i = 0; i < 4096; i++) pix_img[i] = (uint8_t)(i * 3 + (i >> 6) * made + made * 16 + (seed & 0));
    pix_img[0] = (uint8_t)made++;
    for (int i = 0; i < 4096; i++) rect(PX + (i % 64) * 2, PY + (i / 64) * 2, 2, 2, pix_img[i]);
    return 1;
#endif
    uint32_t rng = seed | 1;
    cancel = 0;
    static const char *const names[3] = { "layout", "sharpen", "detail" };
    for (int stage = 0; stage < 3; stage++) {
        int m = stage ? stage : (base_sel ? 3 : 0);
        if (pix_begin(m, w8, th, stage ? 256 : 768, stage ? 366 : 284, &rng)) { status("no memory", -1); for (;;) {} }
        int S = (int)pix_info(m, MF_S), NS = (int)pix_info(m, MF_NS), k = 128 / S, t = 0, more;
        const int16_t *perm = pix_perm();
        status(names[stage], stage);
        do {
            more = pix_cell();
            for (int j = 0; j < NS; j++) { int p = perm[t * NS + j]; rect(PX + (p % S) * k, PY + (p / S) * k, k, k, pix_img[p]); }
            t++;
            show_time();
            if (cancel) return 0;
        } while (more);
        for (int i = 0; i < S * S; i++) th[i] = pix_img[i];
    }
    return 1;
}

/* ---- Gallery: up to 7 kept pictures in the first 32 KB of the cartridge's save memory (battery RAM, or a flash
 * chip if that is what answers; byte access only) ----
 * Sectors 0..6 hold one picture each (4096 palette indices). Sector 7 is a diary: a 32-byte heading, then
 * 32-byte notes, each saying "slot N now holds this prompt" or "slot N is empty". The last good note about a
 * slot wins. Notes are only ever added (flash can clear bits without an erase), so a power cut spoils at most
 * the note being written; the sector is erased and rewritten only when all 127 notes are used, and its heading
 * is written last (a power cut in those few moments loses the list, not the other sectors). Every note has
 * its own check value and one for its picture, so blank or garbage flash reads as an empty gallery. */
static const char save_id[] __attribute__((used, aligned(4))) = "SRAM_V113";       /* tells emulators and flash carts the save type: 32 KB battery RAM */
static const char gal_magic[8] = "PPIXGAL\1";
#define FL ((volatile uint8_t *)0x0E000000)
#define LINE REG16(0x04000006)
#define GAL_N 7                                               /* 7 pictures + the diary = 32 KB, the size every save type has */
#define GAL_DIR (GAL_N * 4096)
#define GAL_NOTES 127
struct gal_ent { uint32_t seq; uint16_t w[8], sum; uint8_t used, base, n; };
static struct {
    struct gal_ent e[GAL_N];
    uint32_t seq;                                             /* next sequence number: larger = kept later */
    int chip, fmt, notes, count;                              /* chip: 2 battery RAM, 1 a flash chip with 4 KB sectors, 0 nothing */
    uint8_t order[GAL_N], buf[32 + 32 * GAL_N];               /* order: used slots, newest first */
} gal __attribute__((section(".bss.ewram")));

static uint16_t check16(const volatile uint8_t *p, int n) { uint32_t h = 2166136261u; while (n--) h = (h ^ *p++) * 16777619u; return (uint16_t)(h ^ h >> 16); }
static void fl_cmd(int c) { FL[0x5555] = 0xAA; FL[0x2AAA] = 0x55; FL[0x5555] = (uint8_t)c; }
/* wait until the byte at a reads v; gives up after so many screen lines (73 microseconds each), never hangs */
static int fl_wait(uint32_t a, int v, int lines) {
    int last = LINE;
    for (;;) {
        if (FL[a] == v) return 1;
        int l = LINE;
        if (l != last) { last = l; if (!--lines) break; }
    }
    if (gal.chip == 1) FL[0x5555] = 0xF0;                          /* tell a stuck chip to give up */
    return 0;
}
static void fl_pause(int lines) { int last = LINE; while (lines) { int l = LINE; if (l != last) { last = l; lines--; } } }
static void fl_probe(void) {                                  /* which chip is it? (the first access must be a write) */
    gal.chip = 0;                                             /* slow RAM is not cleared at power-on */
    uint32_t t = GAL_DIR + 31;                                /* plain RAM first: an unused byte of the diary's heading keeps what */
    uint8_t old = FL[t];                                      /* is written to it (a flash chip ignores a plain write) */
    FL[t] = (uint8_t)(old ^ 0x5A);
    if (FL[t] == (uint8_t)(old ^ 0x5A)) { FL[t] = (uint8_t)~old; if (FL[t] == (uint8_t)~old) { FL[t] = old; gal.chip = 2; return; } }
    fl_cmd(0x90); fl_pause(300);                              /* identity mode needs about 20 ms to switch on and off */
    int id = FL[0] | FL[1] << 8;
    fl_cmd(0xF0); fl_pause(300);
    gal.chip = id == 0xD4BF || id == 0x1CC2 || id == 0x1B32 || id == 0x1362 || id == 0x09C2;   /* the last two: 128K chips, first half used */  /* SST, Macronix, Panasonic. Not Atmel (0x3D1F): it writes whole
                                                               * 128-byte pages, which this diary of small notes is not safe on */
}
static int fl_same(uint32_t a, const uint8_t *p, int n, int size) {      /* flash at a holds p[0..n) and then blank up to size */
    for (int i = 0; i < size; i++) if (FL[a + i] != (i < n ? p[i] : 0xFF)) return 0;
    return 1;
}
static int fl_bytes(uint32_t a, const uint8_t *p, int n) {    /* onto blank flash, one byte at a time */
    if (gal.chip == 2) { for (int i = 0; i < n; i++) FL[a + i] = p[i]; return 1; }
    for (int i = 0; i < n; i++) if (p[i] != 0xFF) { fl_cmd(0xA0); FL[a + i] = p[i]; if (!fl_wait(a + i, p[i], 64)) return 0; }
    return 1;
}
/* a whole 4 KB sector: p[0..n) and blank after it; checked by reading back */
static int fl_sector(int sec, const uint8_t *p, int n) {
    uint32_t a = (uint32_t)sec << 12;
    if (gal.chip == 2) { for (int i = 0; i < 4096; i++) FL[a + i] = i < n ? p[i] : 0xFF; return fl_same(a, p, n, 4096); }
    fl_cmd(0x80); FL[0x5555] = 0xAA; FL[0x2AAA] = 0x55; FL[a] = 0x30;      /* erase: up to 2 s on the slowest chips, 3 s allowed */
    if (!fl_wait(a, 0xFF, 41000) || !fl_same(a, p, 0, 4096)) return 0;
    return fl_bytes(a, p, n) && fl_same(a, p, n, 4096);
}
static int fl_add(uint32_t a, const uint8_t *p, int n) { return fl_bytes(a, p, n) && fl_same(a, p, n, n); }      /* n more bytes on blank flash */

static void gal_sort(void) {                                  /* list the used slots, newest first */
    int n = 0;
    for (int s = 0; s < GAL_N; s++) if (gal.e[s].used) {
        int i = n++;
        for (; i > 0 && gal.e[gal.order[i - 1]].seq < gal.e[s].seq; i--) gal.order[i] = gal.order[i - 1];
        gal.order[i] = (uint8_t)s;
    }
    gal.count = n;
}
static void gal_load(void) {                                  /* read the diary; anything that does not check out is ignored */
    for (int s = 0; s < GAL_N; s++) gal.e[s].used = 0;
    gal.seq = 1; gal.fmt = 0; gal.notes = 0; gal.count = 0;
    if (!gal.chip || !fl_same(GAL_DIR, (const uint8_t *)gal_magic, 8, 8)) return;
    gal.fmt = 1;
    for (int r = 0; r < GAL_NOTES; r++) {
        uint8_t b[32]; int blank = 1, ok;
        for (int i = 0; i < 32; i++) { b[i] = FL[GAL_DIR + 32 + r * 32 + i]; blank &= b[i] == 0xFF; }
        if (blank) continue;
        gal.notes = r + 1;                                    /* the next note goes after the last one that is not blank */
        ok = b[0] == 0xA5 && b[1] < GAL_N && b[2] < 2 && b[3] < 2 && b[4] <= 8 && check16(b, 30) == (b[30] | b[31] << 8);
        for (int i = 0; ok && i < b[4]; i++) { int w = b[12 + i * 2] | b[13 + i * 2] << 8; ok = w > 0 && w < PIX_NWORDS; }
        uint32_t seq = b[8] | b[9] << 8 | b[10] << 16 | (uint32_t)b[11] << 24;
        if (!ok || seq >= 0xFFFFFF00u) continue;              /* a number that large is damage: the count must never wrap */
        struct gal_ent *e = &gal.e[b[1]];
        e->seq = seq;
        if (e->seq >= gal.seq) gal.seq = e->seq + 1;
        e->used = b[2]; e->base = b[3]; e->n = b[4]; e->sum = (uint16_t)(b[28] | b[29] << 8);
        for (int i = 0; i < 8; i++) e->w[i] = (uint16_t)(b[12 + i * 2] | b[13 + i * 2] << 8);
    }
    for (int s = 0; s < GAL_N; s++) if (gal.e[s].used && check16(FL + s * 4096, 4096) != gal.e[s].sum) gal.e[s].used = 0;
    gal_sort();
}
static void gal_note(uint8_t *b, int s) {                     /* the 32 bytes that describe slot s */
    const struct gal_ent *e = &gal.e[s];
    b[0] = 0xA5; b[1] = (uint8_t)s; b[2] = e->used; b[3] = e->base; b[4] = e->n; b[5] = b[6] = b[7] = 0;
    for (int i = 0; i < 4; i++) b[8 + i] = (uint8_t)(e->seq >> (8 * i));
    for (int i = 0; i < 8; i++) { b[12 + i * 2] = (uint8_t)e->w[i]; b[13 + i * 2] = (uint8_t)(e->w[i] >> 8); }
    b[28] = (uint8_t)e->sum; b[29] = (uint8_t)(e->sum >> 8);
    uint16_t c = check16(b, 30); b[30] = (uint8_t)c; b[31] = (uint8_t)(c >> 8);
}
/* put what memory says about slot s into the diary: one more note, or a fresh diary if it is full or not there.
 * A note that will not stick is tried once more in the next place and then given up on: the diary is never
 * erased because of a bad write. */
static int gal_write(int s) {
    if (gal.fmt && gal.notes < GAL_NOTES) {
        uint8_t b[32];
        gal_note(b, s);
        for (int t = 0; t < 2 && gal.notes < GAL_NOTES; t++)  /* a note that reads back wrong is skipped later */
            if (fl_add((uint32_t)(GAL_DIR + 32 + 32 * gal.notes++), b, 32)) return 1;
        return 0;
    }
    int n = 32;
    memset(gal.buf, 0xFF, sizeof gal.buf);
    for (int i = 0; i < GAL_N; i++) if (gal.e[i].used) { gal_note(gal.buf + n, i); n += 32; }
    gal.fmt = fl_sector(GAL_N, gal.buf, n) && fl_add(GAL_DIR, (const uint8_t *)gal_magic, 8);     /* the heading last: half a diary reads as none */
    gal.notes = n / 32 - 1;
    return gal.fmt;
}
/* keep the picture just painted: into a free slot, or over the oldest one. Picture first, its note last, so a
 * power cut in between leaves the other pictures alone (the old note then fails its picture check). */
static int gal_keep(void) {
    int s = 0;
    while (s < GAL_N && gal.e[s].used) s++;
    if (s == GAL_N) s = gal.order[GAL_N - 1];
    struct gal_ent *e = &gal.e[s];
    e->used = 0;
    if (fl_sector(s, pix_img, 4096)) {
        e->seq = gal.seq++; e->base = (uint8_t)base_sel; e->n = (uint8_t)nwords; e->sum = check16(pix_img, 4096);
        for (int i = 0; i < 8; i++) e->w[i] = i < nwords ? words[i] : 0;
        e->used = 1;
        if (gal_write(s)) { gal_sort(); return 1; }
    }
    gal_load();                                               /* it did not stick: show what the flash really holds */
    return 0;
}
static void gal_delete(int s) {
    gal.e[s].used = 0; gal.e[s].seq = gal.seq++;
    if (gal_write(s)) gal_sort(); else gal_load();            /* if the note did not stick, show what the flash really holds */
}

#define GX 8
#define GY 4
static char *numz(char *o, int v) { if (v > 9) *o++ = (char)('0' + v / 10); *o++ = (char)('0' + v % 10); return o; }
static void hints(const char *a, const char *b, int c) { textn(8, 138, a, 28, c, PIX_C_BG); textn(8, 149, b, 28, c, PIX_C_BG); }
static void gal_screen(int k) {                               /* the viewer, showing picture k of the newest-first list */
    rect(0, 0, 240, 160, PIX_C_BG);
    if (!gal.count) {
        text(8, 6, "GALLERY", PIX_C_ACC, PIX_C_BG);
        if (!gal.chip) {
            text(8, 44, "Can't save on this cartridge", PIX_C_FG, PIX_C_BG);
            text(8, 60, "(no save memory found)", PIX_C_DIM, PIX_C_BG);
        } else {
            text(8, 44, "No pictures kept yet.", PIX_C_FG, PIX_C_BG);
            text(8, 60, "Paint one, then press START", PIX_C_DIM, PIX_C_BG);
            text(8, 70, "when it is done to keep it", PIX_C_DIM, PIX_C_BG);
            text(8, 80, "here. There is room for 7.", PIX_C_DIM, PIX_C_BG);
        }
        hints("", "B back", PIX_C_DIM);
        return;
    }
    int s = gal.order[k];
    const struct gal_ent *e = &gal.e[s];
    text(144, 4, "GALLERY", PIX_C_ACC, PIX_C_BG);
    rect(GX - 2, GY - 2, 132, 132, PIX_C_PANEL);
    for (int i = 0; i < 4096; i++) {
        int c = FL[s * 4096 + i];
        volatile uint16_t *p = VRAM + ((GY + (i >> 6) * 2) * 240 + GX + (i & 63) * 2) / 2;
        p[0] = p[120] = (uint16_t)(c | c << 8);
    }
    for (int i = 0; i < e->n; i++) textn(144, 18 + i * 10, wtext(e->w[i]), 11, PIX_C_FG, PIX_C_BG);
    if (!e->n) text(144, 18, "random", PIX_C_DIM, PIX_C_BG);
    text(144, 102, e->base ? "scenes" : "objects", PIX_C_DIM, PIX_C_BG);
    char t[10], *o = numz(t, k + 1);
    *o++ = ' '; *o++ = 'o'; *o++ = 'f'; *o++ = ' '; o = numz(o, gal.count); *o = 0;
    text(144, 114, t, PIX_C_ACC, PIX_C_BG);
    hints("LEFT/RIGHT flip    B back", "A use prompt  SELECT delete", PIX_C_DIM);
}
static void settle(void) { for (int i = 0; i < 20; i++) wait_frame(); prev_keys = (uint16_t)KEYS; }     /* a second tap of A right after a "yes" is not a new order */
/* look through the kept pictures; A puts a picture's prompt back on the entry screen (returns 1 then) */
static int gallery(void) {
    int k = 0;
    gal_screen(0);
    for (;;) {
        wait_frame();
        int e = pressed();
        if (e & K_B) return 0;
        if (!gal.count || !e) continue;
        if (e & K_A) {
            const struct gal_ent *g = &gal.e[gal.order[k]];
            nwords = g->n; base_sel = g->base; npre = 0; cand = 0;
            for (int i = 0; i < 8; i++) words[i] = g->w[i];
            return 1;
        }
        if (e & K_SELECT) {
            hints("Delete this picture?", "A yes   B no", PIX_C_FG);
            do { wait_frame(); e = pressed(); } while (!(e & (K_A | K_B)));
            if (e & K_A) {
                hints("deleting...", "do not switch off", PIX_C_DIM);
                gal_delete(gal.order[k]);
                if (k >= gal.count) k = gal.count ? gal.count - 1 : 0;       /* the oldest went: stay at the end of the list */
                gal_screen(k);
                settle();
                continue;
            }
            e = 0;
        }
        if (e & (K_RIGHT | K_R)) k++;
        if (e & (K_LEFT | K_L)) k--;
        if (k >= gal.count) k = 0;
        if (k < 0) k = gal.count - 1;
        gal_screen(k);
    }
}
/* the finished picture stays up: returns 1 to paint the prompt again, 0 to go back. START keeps it, once. */
static void done_keys(const char *top, int c) {
    textn(144, 126, top, 11, c, PIX_C_BG);
    textn(144, 136, "A again", 11, PIX_C_DIM, PIX_C_BG);
    textn(144, 146, "B back", 11, PIX_C_DIM, PIX_C_BG);
}
static int done(uint32_t *frames) {
    const char *keep = gal.chip ? "START keep" : "can't save";
    int kept = 0, msg = 0; char t[12] = "kept ";
    status("done", -1);
    done_keys(keep, PIX_C_DIM);
    for (;;) {
        wait_frame(); (*frames)++;
        int e = pressed();
        if (msg && !--msg) done_keys(kept ? t : keep, PIX_C_DIM);      /* "kept 3/15" stays, dimmed; a failure may be tried again */
        if ((e & K_START) && gal.chip && !kept) {
            if (gal.count == GAL_N) {
                textn(144, 126, "Replace", 11, PIX_C_FG, PIX_C_BG);
                textn(144, 136, "oldest?", 11, PIX_C_FG, PIX_C_BG);
                textn(144, 146, "A yes  B no", 11, PIX_C_ACC, PIX_C_BG);
                do { wait_frame(); (*frames)++; e = pressed(); } while (!(e & (K_A | K_B)));
                if (e & K_B) { done_keys(keep, PIX_C_DIM); msg = 0; continue; }
            }
            done_keys("saving...", PIX_C_DIM);
            kept = gal_keep();
            char *o = numz(t + 5, gal.count);
            *o++ = '/'; o = numz(o, GAL_N); *o = 0;
            done_keys(kept ? t : "save failed", kept ? PIX_C_ACC : PIX_C_WARN);
            msg = 180;                                        /* bright for three seconds */
            settle();
            continue;
        }
        if (e & K_A) return 1;
        if (e & K_B) return 0;
    }
}

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
    REG16(0x04000204) = 0x4317;                               /* fastest cartridge timing, prefetch on */
    fast_ewram();
    const uint16_t *pal = (const uint16_t *)(pix_blob + O_PAL);
    for (int i = 0; i < 256; i++) ((volatile uint16_t *)0x05000000)[i] = pal[i];
    REG16(0x04000000) = 0x0404;                               /* mode 4, background 2 */
    pix_yield = poll;
    fl_probe();                                               /* after fast_ewram: its test wipes slow RAM */
    gal_load();
    uint32_t frames = 0;
    for (;;) {
        entry();
        for (;;) {
            frames += REG16(0x04000006) + 977;
            int ok = paint(frames * 2654435761u + REG16(0x04000108));
            prev_keys = (uint16_t)KEYS;
            if (!ok || !done(&frames)) break;
        }
        while (KEYS & K_B) {}
        prev_keys = (uint16_t)KEYS;
    }
}
