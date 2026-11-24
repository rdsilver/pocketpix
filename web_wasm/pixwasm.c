/* WebAssembly build of the cartridge's picture engine (pix/pix.c, the same C code as the native twin).
   JS copies build/pix.bin to pw_blob(), then for each pass calls pw_begin() and pw_cell() until it returns 0.
   Memory (64 MB, fixed): the engine's own variables at the bottom, the stack after them, the row tables at 1984 KB
   (pix.h PIX_WASM_TBL), each model's keys and values from 2 MB, the saved states (pw_save) from 3 MB, the row
   offset cache from 26 MB (pix.c t3off), the model file from 32 MB, and the buffers JS fills from 62 MB. */
#include <stdint.h>
#include "../pix/pix.h"
#define BLOB_AT  0x2000000u
#define BLOB_MAX 0x1D00000u                       /* 29 MB */
#define BUF_AT   0x3E00000u
#define ARENA_AT 0x0200000u                       /* 5 arenas x 200 KB (pix.c's ARENA_B): one per model, and a spare for model 2 (pw_begin(4)) */
#define ARENA_SZ 204800u
#define SNAP_AT  0x0300000u                       /* per model 257 saved states (after 0 .. 256 cells), each of snapsz(m) bytes */
#define T3C_AT   0x1A00000u                       /* up to the model file: 6 MB (about 3.9 used by the 4 models so far) */
uint8_t *const pix_t3c = (uint8_t *)T3C_AT;
const uint32_t pix_t3c_len = BLOB_AT - T3C_AT;
void pix_t3c_reset(void);
const uint8_t *pix_blob = (const uint8_t *)BLOB_AT;
extern const uint8_t *pix_force, *pix_forcemask;
extern char __heap_base;
uint8_t *pix_arena;
uint32_t pix_save(uint8_t *p, uint32_t max);
void pix_load(const uint8_t *p);
static int model, spare;
/* a saved state of model m: pix_save's header, picture and flat-cell list, and the mixer state of two streams for the
   one-pixel models (guidance), of one for the others (their passes have no room for two in the arenas anyway) */
static uint32_t snapsz(int m) {
    uint32_t L = pix_info(m, MF_L), natt = 0, S = pix_info(m, MF_S), nc = pix_info(m, MF_NC);
    for (uint32_t l = 0; l < L; l++) natt += pix_info(m, MF_ATTMASK) >> l & 1;
    return (16 + S * S + (nc + 15) / 16 * 16 + (pix_info(m, MF_NS) == 1 ? 2 : 1) * (L - natt) * pix_info(m, MF_D) * 24 + 15) & ~15u;
}
static uint8_t *snapbase(int m) { uint32_t a = SNAP_AT; for (int j = 0; j < m; j++) a += 257 * snapsz(j); return (uint8_t *)a; }
static uint8_t *snap(int d) { return snapbase(model) + (uint32_t)d * snapsz(model); }
#define FIMG  ((uint8_t *)(BUF_AT))
#define FMASK ((uint8_t *)(BUF_AT + 4096))
#define TH    ((uint8_t *)(BUF_AT + 8192))
#define WORDS ((uint16_t *)(BUF_AT + 9216))
#define RNG   ((uint32_t *)(BUF_AT + 9248))
#define THDEP ((int16_t *)(BUF_AT + 12288))
/* pix.c's copies made per pass (the mixers' coefficients by columns: at most 8 layers of 192 x 6 words), outside the saved state */
int32_t *const pix_mixcf = (int32_t *)(BUF_AT + 65536);
static void layout_ok(void) { if ((uintptr_t)&__heap_base > PIX_WASM_TBL) __builtin_trap(); }     /* variables and stack must end below the tables */
void *memset(void *d, int c, unsigned long n) { uint8_t *p = d; while (n--) *p++ = (uint8_t)c; return d; }
void *memcpy(void *d, const void *s, unsigned long n) { uint8_t *p = d; const uint8_t *q = s; while (n--) *p++ = *q++; return d; }
void *memmove(void *d, const void *s, unsigned long n) { uint8_t *p = d; const uint8_t *q = s; if (p < q) while (n--) *p++ = *q++; else while (n--) p[n] = q[n]; return d; }
__attribute__((export_name("pw_blob"))) uint8_t *pw_blob(void) { layout_ok(); pix_t3c_reset(); return (uint8_t *)BLOB_AT; }     /* (a new model file: offsets from scratch) */
__attribute__((export_name("pw_blob_max"))) uint32_t pw_blob_max(void) { return BLOB_MAX; }
__attribute__((export_name("pw_words"))) uint16_t *pw_words(void) { return WORDS; }
__attribute__((export_name("pw_thumb"))) uint8_t *pw_thumb(void) { return TH; }
__attribute__((export_name("pw_img"))) uint8_t *pw_img(void) { return pix_img; }
__attribute__((export_name("pw_seed"))) void pw_seed(uint32_t s) { *RNG = s; }
__attribute__((export_name("pw_begin"))) int pw_begin(int m, int guide, int invt) {
    layout_ok();
    if (m < 0 || m > 4 || (uint32_t)snapbase(4) > T3C_AT || ARENA_AT + 5 * ARENA_SZ > SNAP_AT) return -1;
    spare = m == 4; pix_arena = (uint8_t *)(ARENA_AT + (uint32_t)m * ARENA_SZ); model = m = spare ? 2 : m;     /* 4: model 2 in the spare arena, */
    if (pix_begin(m, WORDS, TH, guide, invt, RNG)) return -1;                                              /* so model 2's own keys stay good */
    return spare || pix_save(snap(0), snapsz(m)) ? 0 : -1;     /* (a state after 0 cells: never loaded, checks the size) */
}
__attribute__((export_name("pw_cell"))) int pw_cell(void) { return pix_cell(); }
__attribute__((export_name("pw_size"))) int pw_size(int m) { return pix_info(m, MF_S); }
__attribute__((export_name("pw_ns"))) int pw_ns(int m) { return pix_info(m, MF_NS); }
__attribute__((export_name("pw_perm"))) const int16_t *pw_perm(void) { return pix_perm(); }
__attribute__((export_name("pw_done"))) int pw_done(void) { return pix_cells_done(); }
/* holding pixels: pw_force(1) makes the next passes keep the pixels marked in pw_fmask() at their pw_fimg() values */
__attribute__((export_name("pw_fimg"))) uint8_t *pw_fimg(void) { return FIMG; }
__attribute__((export_name("pw_fmask"))) uint8_t *pw_fmask(void) { return FMASK; }
__attribute__((export_name("pw_force"))) void pw_force(int on) { pix_force = on ? FIMG : 0; pix_forcemask = on ? FMASK : 0; }
/* saved states, one per model and number of cells done: pw_save() keeps the pass's state after the pw_done() cells
   painted so far (0: no room); pw_load(d) after pw_begin of the same model takes up the one after d cells, with this
   pass's thumbnail. Each model keeps its keys and values in its own arena, so the state saved after d cells is good as
   long as no later pass of the model ran fewer than d cells (that one rewrote the keys from where it began). */
__attribute__((export_name("pw_save"))) uint32_t pw_save(void) { return spare ? 0 : pix_save(snap(pix_cells_done()), snapsz(model)); }
__attribute__((export_name("pw_load"))) void pw_load(int d) { if (!spare) pix_load(snap(d)); }
/* per thumbnail pixel of model m: the first cell that reads it (NC when none), so a pass knows how far a saved state
   holds after a change of the thumbnail */
__attribute__((export_name("pw_thdep"))) const int16_t *pw_thdep(int m) {
    int nc = (int)pix_info(m, MF_NC), ns = (int)pix_info(m, MF_NS), win = (int)pix_info(m, MF_WIN);
    const int16_t *tw = (const int16_t *)(pix_blob + pix_info(m, MF_TW)), *tp = (const int16_t *)(pix_blob + pix_info(m, MF_TP));
    for (int i = 0; i < 1024; i++) THDEP[i] = (int16_t)nc;
    if (!pix_info(m, MF_F)) return THDEP;
    for (int t = nc - 1; t >= 0; t--) {
        for (int k = 0; k < win; k++) if (tw[t * win + k] >= 0) THDEP[tw[t * win + k]] = (int16_t)t;
        for (int k = 0; k < ns * 5; k++) if (tp[t * ns * 5 + k] >= 0) THDEP[tp[t * ns * 5 + k]] = (int16_t)t;
    }
    return THDEP;
}
