/* Picture model inference: integers only. Arithmetic follows engine/DESIGN.md in the Koko repository; img/model.py is the model. */
#include "pix.h"
#ifdef __wasm__
#include <wasm_simd128.h>
#endif

#ifdef __arm__
#define IWRAM_CODE __attribute__((section(".iwram.text"), target("arm")))
#define IWRAM_DATA __attribute__((section(".bss.iwram")))
#define EWRAM_DATA __attribute__((section(".bss.ewram")))
#define VRAM0_DATA __attribute__((section(".bss.vram0")))
#define VRAM1_DATA __attribute__((section(".bss.vram1")))
#define VRAM2_DATA __attribute__((section(".bss.vram2")))
#define ROM_CODE __attribute__((section(".romcode")))     /* run once per picture: stays in ROM, leaves video memory to hot code and data */
#ifndef PIX_ARM_GLUE
/* Code outside fast RAM (the C glue in video memory, the ROM code) is Thumb: video memory has a 16-bit bus, so a Thumb
 * instruction is fetched in one cycle and an ARM one in two (fast-RAM code stays ARM: IWRAM_CODE). */
#pragma GCC target("thumb")
#define ARM_GLUE __attribute__((target("arm")))           /* glue that is faster as ARM (64-bit arithmetic) */
#else
#define ARM_GLUE
#endif
#else
#define ARM_GLUE
#define VRAM0_DATA
#define VRAM1_DATA
#define VRAM2_DATA
#define ROM_CODE
#define IWRAM_CODE
#define IWRAM_DATA
#define EWRAM_DATA
#endif
#define NE 81
#define QB 12
#define CHG 48                                   /* groups of 4 inputs per chunk: tables for 192 inputs */
#define P1 257
#define RS(x, n) (((x) + ((int64_t)1 << ((n) - 1))) >> (n))
#define B8(o) ((const int8_t *)(pix_blob + (o)))
#define BU8(o) ((const uint8_t *)(pix_blob + (o)))
#define B16(o) ((const int16_t *)(pix_blob + (o)))
#define BU16(o) ((const uint16_t *)(pix_blob + (o)))
#define B32(o) ((const int32_t *)(pix_blob + (o)))
#define MAXIN (PIX_MAX_HD > PIX_MAX_HH ? PIX_MAX_HD : PIX_MAX_HH)
/* Matrix rows in ROM are "T3" words: three groups of four inputs per 32-bit word (export.py step_words; the row
 * kernels' comment in kernels.s says why). A row of g groups: */
#define T3W(g) (((g) + 2) / 3)                   /* words */
#define ROWB(g) (4 * T3W(g))                     /* bytes */
#ifdef __wasm__
/* (browser) four lanes of (int32_t)rs((int64_t)a * b, sh) for 0 < sh <= 32: the 64-bit products of the 32-bit lanes,
 * rounded, and the low words of their logical shifts (the same words as the arithmetic shift's) */
static inline v128_t mulrs4(v128_t a, v128_t b, int sh) {
    v128_t r = wasm_i64x2_splat((int64_t)1 << (sh - 1));
    v128_t lo = wasm_u64x2_shr(wasm_i64x2_add(wasm_i64x2_extmul_low_i32x4(a, b), r), sh);
    v128_t hi = wasm_u64x2_shr(wasm_i64x2_add(wasm_i64x2_extmul_high_i32x4(a, b), r), sh);
    return wasm_i32x4_shuffle(lo, hi, 0, 2, 4, 6);
}
/* (int32_t)rs((int64_t)x, n) of an int32 x without 64 bits: x >> n, plus 1 when the bit below is set */
#define RS32(x, n) (((x) >> (n)) + ((x) >> ((n) - 1) & 1))
/* byte shuffles that move the 32-bit lanes set in a 4-bit mask to the front (lists made without branches) */
static const uint8_t lane_lut[16][16] = {
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, {0,1,2,3,0,0,0,0,0,0,0,0,0,0,0,0}, {4,5,6,7,0,0,0,0,0,0,0,0,0,0,0,0}, {0,1,2,3,4,5,6,7,0,0,0,0,0,0,0,0},
    {8,9,10,11,0,0,0,0,0,0,0,0,0,0,0,0}, {0,1,2,3,8,9,10,11,0,0,0,0,0,0,0,0}, {4,5,6,7,8,9,10,11,0,0,0,0,0,0,0,0}, {0,1,2,3,4,5,6,7,8,9,10,11,0,0,0,0},
    {12,13,14,15,0,0,0,0,0,0,0,0,0,0,0,0}, {0,1,2,3,12,13,14,15,0,0,0,0,0,0,0,0}, {4,5,6,7,12,13,14,15,0,0,0,0,0,0,0,0}, {0,1,2,3,4,5,6,7,12,13,14,15,0,0,0,0},
    {8,9,10,11,12,13,14,15,0,0,0,0,0,0,0,0}, {0,1,2,3,8,9,10,11,12,13,14,15,0,0,0,0}, {4,5,6,7,8,9,10,11,12,13,14,15,0,0,0,0}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15} };
#endif

uint32_t pix_prof[PP_N] EWRAM_DATA;
void (*pix_yield)(void);
uint8_t pix_img[64 * 64] EWRAM_DATA;
#if defined(__arm__) && defined(PIX_PROF)
static uint32_t tlast VRAM2_DATA;
static inline uint32_t now(void) {
    volatile uint16_t *t = (volatile uint16_t *)0x04000100;
    uint16_t h = t[2], l = t[0];
    if (h != t[2]) { h = t[2]; l = t[0]; }
    return ((uint32_t)h << 16) | l;
}
#define TICK(k) do { uint32_t n_ = now(); pix_prof[k] += n_ - tlast; tlast = n_; } while (0)
#define TICK0() (tlast = now())
#ifdef PIX_PROFA
uint32_t pix_profa[11] EWRAM_DATA; static uint32_t alast VRAM2_DATA;      /* (not in fast RAM: it is full) */
#define ATK(k) do { uint32_t n_ = now(); pix_profa[k] += n_ - alast; alast = n_; } while (0)
#else
#define ATK(k)
#endif
#else
#define ATK(k)
#define TICK(k)
#define TICK0()
#endif
#if defined(__arm__) && defined(PIX_FORCE)
const uint8_t *pix_force __attribute__((section(".bss.vram2")));     /* test ROM: paint a given picture (timing two builds on the same picture) */
#define FORCED(i) (pix_force)
#endif
#ifndef __arm__
const uint8_t *pix_force;
const uint8_t *pix_forcemask;                  /* (twin, browser) with pix_force: only the pixels marked here are forced */
#define FORCED(i) (pix_force && (!pix_forcemask || pix_forcemask[i]))
void (*pix_probe)(int, int, const int32_t *, const int32_t *);
void (*pix_probe2)(int, int, const int32_t *, const int32_t *);
int64_t pix_maxsum;
#endif
#ifdef __wasm__
/* (browser) the twin's test hooks (probes, shadow runs, cell classes) are never set: constants, so their code drops out */
#define pix_probe ((void (*)(int, int, const int32_t *, const int32_t *))0)
#define pix_probe2 pix_probe
#endif

#ifdef __wasm__
/* the browser build: the tables at a fixed address (pixwasm.c's layout, below the offset cache), outside the saved
 * state (they hold nothing from one cell to the next); a constant address goes into the loads' offset fields */
#define tbl ((int32_t *)PIX_WASM_TBL)
#else
static int32_t tbl[CHG * NE] IWRAM_DATA;
#endif
static int32_t x[PIX_MAX_D] IWRAM_DATA, xn[PIX_MAX_D] IWRAM_DATA;
static int32_t proj[256] IWRAM_DATA, hid[MAXIN] VRAM2_DATA, tmp[MAXIN] EWRAM_DATA, mixv[PIX_MAX_DV] VRAM2_DATA;
/* sc lives in sprite memory (OAM, 1 KB at 0x07000000): a 32-bit bus without wait states like fast RAM, and nothing
 * shows sprites while a picture is painted (the display control never turns them on). Words only: byte writes
 * to sprite memory are ignored by the hardware. PIX_SC_OAM 0 puts it into spare video memory instead (one cycle
 * more per word; fast RAM has no room for it since the fused row kernel got its own loops). */
#ifndef PIX_SC_OAM
#define PIX_SC_OAM 1
#endif
#if defined(__arm__) && PIX_SC_OAM
#define sc ((int32_t *)0x07000000)
typedef char pix_oam_check[(PIX_MAX_NC * 4 <= 1024) ? 1 : -1];
#else
static int32_t sc[PIX_MAX_NC] VRAM2_DATA;
#endif
/* The head builds tables for DH inputs only, so while it runs the rest of tbl is idle fast RAM: the head's per-cell
 * sums live there. They are all written after the body's last wider table use (hin: after the rows of MF_HIN; the
 * others: in pix_cell after body()) and are dead before the next cell's body builds tables again. */
#define TAIL0 ((PIX_MAX_DH / 4 * NE + 7) / 8 * 8)
#define cellsum (tbl + TAIL0)                                /* rows of the cell's pixels drawn so far */
#define ycell (tbl + TAIL0 + PIX_MAX_DH)                     /* the same after the head's first matrix, plus the cell's own part */
#define tsum ((int32_t (*)[PIX_MAX_DH])(tbl + TAIL0 + PIX_MAX_DH + PIX_MAX_HH))      /* thumbnail block rows, 4 blocks */
#define hin0 (tbl + TAIL0 + 5 * PIX_MAX_DH + PIX_MAX_HH)     /* stream 0's input to the head */
typedef char pix_tail_check[(TAIL0 + 6 * PIX_MAX_DH + PIX_MAX_HH <= CHG * NE) ? 1 : -1];
/* Fast RAM is scarce, so the head's buffers reuse the body's (the body is finished when the head runs, and starts
 * each cell from scratch): facc = x, u = xn, yacc = sc; sparse_out's result goes to proj. */
#define facc x
#define hu xn
#define yacc sc
#define z proj                                               /* the head's normalised state (proj is free between its uses) */
typedef char pix_alias_check[(PIX_MAX_DH <= PIX_MAX_D && PIX_MAX_HH <= PIX_MAX_NC && PIX_MAX_D <= 256 && PIX_MAX_DK + PIX_MAX_DK / PIX_MAX_NH + 16 <= 256) ? 1 : -1];
static int32_t zsave[PIX_MAX_DH] VRAM0_DATA;
/* The head's second matrix (one T3 row of steps per hidden unit, 11 KB) and its group rows: read from ROM unless pix_begin
 * finds room for a copy in the word arena's unused video memory, where a word costs half as much to read. */
static const uint8_t *h2cols VRAM2_DATA, *outg VRAM2_DATA;
/* vram1 (the top of the word arena, free only with several pixels per cell): five rows of HH words, for the four
 * thumbnail blocks' sums after the first matrix (ytsum) and stream 0's yin. When a pass runs one stream, pix_begin
 * moves three of the five into the idle buffers of stream 1 (PIX_V1SQUEEZE), and the arena's spare words below, where
 * the head's second matrix is copied, grow by 3 KB: enough for the 64 x 64 pass's T3 copy (11 KB). */
#ifndef PIX_V1SQUEEZE
#define PIX_V1SQUEEZE 1
#endif
static int32_t v1buf[5][PIX_MAX_HH] VRAM1_DATA, hin1[PIX_MAX_DH] VRAM0_DATA;
static int32_t *ytsum[4] VRAM2_DATA;                         /* (set by pix_begin) */
/* the head's per-pixel ROM words (final norm gains, group and member multipliers): copied to video memory by pix_begin
 * on the GBA (ldm of 4 words: 10 cycles instead of 16 from ROM) */
static const int32_t *hf_g VRAM2_DATA, *outg_m VRAM2_DATA, *outm_m VRAM2_DATA;
#if defined(__arm__) && !defined(PIX_NO_HCOPY)
static int32_t hcopy[PIX_MAX_DH + 16 + 256] VRAM2_DATA;
#endif
static uint8_t thumb[32 * 32] EWRAM_DATA;
static uint8_t blkof[PIX_MAX_NS] EWRAM_DATA, blkpix[4] EWRAM_DATA;   /* a pixel's thumbnail block within its cell; a block's first pixel (per picture: no divisions per pixel) */
static uint8_t skipped[PIX_MAX_NC] EWRAM_DATA;               /* cells filled without running the model (flat regions) */
int pix_flatskip = 1, pix_nskip, pix_nhskip;
/* (B) approximation, switchable: 1 = cells of 2 x 2 thumbnail blocks (the 64 x 64 pass) are painted straight from the
 * thumbnail when their own four blocks are one colour and at most PIX_FLAT_ND of the eight blocks touching them
 * differ; 0 = the window-tolerance rule (MF_FLATT) for every model, as before. */
#ifndef PIX_FLAT_OWN
#define PIX_FLAT_OWN 0
#endif
/* (B) approximation, switchable: PIX_PCOPY 1 = in cells that run, a pixel whose own thumbnail block and the four blocks
 * beside it are one colour (in the 64 x 64 pass one of those four may differ), and whose drawn neighbours (left, up,
 * up-left, up-right) all have that colour too, takes it without running the head (the body ran, so later cells'
 * attention and mixer history are as before). 0 = off. */
#ifndef PIX_PCOPY
#define PIX_PCOPY 1
#endif
#ifndef __arm__
int pix_pcopy = PIX_PCOPY, pix_pcopynd = 1;                  /* tests: PCOPY, PCOPYND in the twin */
long long pix_npcopy;
#define PIX_PCOPY_ND pix_pcopynd
#else
#define pix_pcopy PIX_PCOPY
#define PIX_PCOPY_ND 1
#endif
#ifndef __arm__
int pix_flatrule = PIX_FLAT_OWN, pix_flatnd = 5;             /* tests: FLATRULE / FLATND in the twin */
#define PIX_FLAT_ND pix_flatnd
#else
#define pix_flatrule PIX_FLAT_OWN
#define PIX_FLAT_ND 5
#endif

/* (B) approximation, switchable (PIX_HEADSKIP, 0 = off): a cell of the 64 x 64 pass that the window rule does not
 * paint flat but whose own four thumbnail blocks are one colour and at most PIX_HS_ND of the eight blocks touching
 * them differ ("own rule") runs only the part of the body that later cells read (every layer up to the last one's
 * keys and values / mixer state) and takes its pixels from the thumbnail. Unlike a skipped cell it stays in later
 * cells' attention, so the other pixels are computed exactly as before. */
#ifndef PIX_HEADSKIP
#define PIX_HEADSKIP 0
#endif
#ifndef PIX_HS_BORDER
#define PIX_HS_BORDER 0
#endif
#ifndef PIX_HS_ND32
#define PIX_HS_ND32 (-1)
#endif
#ifndef PIX_HS_ND
#define PIX_HS_ND 3
#endif
#ifndef __arm__
int pix_headskip = PIX_HEADSKIP, pix_hsnd = PIX_HS_ND, pix_hsb = PIX_HS_BORDER, pix_hsnd32 = PIX_HS_ND32;      /* tests: HEADSKIP / HSND in the twin */
uint8_t pix_cellclass[PIX_MAX_NC], pix_copycol[PIX_MAX_NC * PIX_MAX_NS];   /* tests: bit 0 window rule, bit 1 own rule, bit 2 cell ran the head; each pixel's block colour */
#else
#define pix_headskip PIX_HEADSKIP
#define pix_hsnd PIX_HS_ND
#define pix_hsb PIX_HS_BORDER
#define pix_hsnd32 PIX_HS_ND32
#endif
static int kvonly VRAM2_DATA;

typedef struct { uint8_t *keys; int8_t *vals; uint8_t *vsh; int32_t *mst, *cw; int first; int32_t *hin, *yin; } Stream;
static Stream st[2] VRAM2_DATA;
/* the head's first matrix on the cell's own input, per stream (only models with more than one pixel per cell, so
 * stream 0's can live in the video memory that such passes leave free) */
static int32_t yin1[PIX_MAX_HH] VRAM2_DATA;
/* working memory: bytes (keys, values) in slow RAM; words (mixer state, prompt sums) in spare video memory on the GBA */
#define ARENA_B 204800
#ifdef __wasm__
extern uint8_t *pix_arena;                                   /* (browser) one per model (pixwasm.c): its keys and values outlive the other passes */
#define arena_b pix_arena
#else
static uint8_t arena_b[ARENA_B] EWRAM_DATA __attribute__((aligned(4)));
#endif
#ifdef __arm__
#define arena_w ((int32_t *)0x0600A000)
#define ARENA_W (32 * 1024 / 4)                               /* up to 0x06012000; passes with more than one pixel per cell keep the top 8 KB free */
#define VRAM1_WORDS (0x1400 / 4)                              /* region vram1 in gba.ld: the top 5 KB of the arena's memory */
typedef char pix_vram1_check[(5 * PIX_MAX_HH <= VRAM1_WORDS) ? 1 : -1];
#else
#define VRAM1_WORDS 0
#define ARENA_W (56 * 1024 / 4)
static int32_t arena_w[ARENA_W];
#endif

/* the model being run */
static const uint32_t *M IWRAM_DATA, *LY IWRAM_DATA;
static int S, C, D, L, NH, DK, DV, HD, DH, HH, HK, HV,
    KB, NC, NS, NCW, NATT, WIN, twice, cellno, guide, invt, YQ;
/* (B) approximation, switchable: PIX_TAUL 1 = the body's threshold is per layer (LF_TAU, export.py TAUL: later layers
 * stand larger ones); 0 = the model's plain MF_TAUB in every layer, as before (the twin: TAU_B=-2). */
#ifndef PIX_TAUL
#define PIX_TAUL 1
#endif
static int32_t tau_h, tau_l[PIX_MAX_L] VRAM2_DATA;            /* hidden values up to this are dropped: head, body layers */
static uint32_t p0_h, p0_l[PIX_MAX_L] VRAM2_DATA;             /* smallest p that relu_sq keeps (head, body layers) */
static int gcut;
static int awin;                                             /* attention looks back at most this many cells (0 = all) */
int pix_awin = -1;                                           /* tests: override the model's window */
#ifndef __arm__
int pix_hwin[PIX_MAX_L][PIX_MAX_NH];                         /* tests: a window per attention layer and head (0 = the model's) */
unsigned pix_hdrop, pix_adrop;
int pix_g2;                                                   /* tests: 1/2 = the empty-prompt stream reuses the prompt stream's top 8 (with its own scores / with the same weights) */                               /* tests: heads (bit ai * NH + h) / attention layers (bit ai) left out */
#endif
int pix_gcut16 = 11;                                         /* guidance runs for this many sixteenths of the cells */
int pix_tau_b = -1, pix_tau_h = -1;                          /* tests: override the models' thresholds (pix_tau_b = -2: the model's plain body threshold in every layer) */
const int *pix_taul;                                         /* tests: body thresholds per layer (negative: the model's) */
static uint32_t *rngp;
#define LAY(l, f) (LY[(l) * LF_N + (f)])

/* ---- kernels ---- */
#ifdef __arm__
void tern_build81(int32_t *tbl, const int32_t *x, int ngroups);
void tern_scan(int32_t *out, const uint8_t *steps, const int32_t *tbl, int ngroups, int nrows);       /* T3 rows */
void tern_scan8(int32_t *out, const uint8_t *steps, const int32_t *tbl, int nwords, int nrows);       /* rows of step bytes (keys), from ROM */
void axpy8(int32_t *acc, const int8_t *v, int32_t p, int n);
void mixer_k(int32_t *x, const int32_t *xn, int32_t *state, const int32_t *coef, int n);
#define MIXC(l, mi) B32(LAY(l, LF_MIXC))                 /* layer l's mixer coefficients (the browser build keeps a copy by columns) */
uint64_t sumsq_k(const int32_t *in, int n);
void norm_k(int32_t *out, const int32_t *in, const int32_t *g, int32_t inv4, int n);
void scale_add(int32_t *x, const int32_t *a, int32_t m, int n);
void add16(int32_t *acc, const int16_t *v, int n);
void add32(int32_t *acc, const int32_t *v, int n);
void sum16_3(int32_t *out, const int32_t *in, const int16_t *a, const int16_t *b, const int16_t *c, int n);
void sum16_1w(int32_t *acc, const int32_t *w, const int16_t *a, int n);
void sum16_4u(int32_t *out, int32_t *acc, const int16_t *a, const int16_t *b, const int16_t *c, const int16_t *d, int n);
void axpy8_2(int32_t *out, const int32_t *in, const int8_t *v1, int32_t p1, const int8_t *v2, int32_t p2, int n);
void head_u(int32_t *u, const int32_t *hin, const int32_t *f, const int32_t *ts, int n);
int keytop(const uint8_t *kend, const int32_t *bias, int nrows, int32_t *tv);
typedef struct { const int8_t *v; int32_t p; } MixA;
typedef struct { const int8_t *v; int32_t p, rr, hf; } MixB;
void mixk(int32_t *out, const MixA *a, const MixB *b, int ngroups);
void scale_rs(int32_t *out, const int32_t *in, int32_t m, int n, int shift);
void key_k(const int32_t *pj, const int32_t *kg, const int32_t *rcs, uint8_t *kk, int ngroups);
void copy32(int32_t *dst, const int32_t *src, int n);
void rs4_k(int32_t *x, int n);
void mulrs_k(int32_t *out, const int32_t *a, const int32_t *b, int n, int shift);
#define tern_build tern_build81
#define scan(out, s, tb, ng, nrows) tern_scan(out, s, tb, ng, nrows)
#define scan8(out, s, tb, nbytes, nrows) tern_scan8(out, s, tb, (nbytes) / 4, nrows)
#else
#ifdef __wasm__
/* The browser build: entry lo + 9 hi = (pair of the first two inputs) + (pair of the last two), the same sums.
 * Tables of a multiple of 4 groups are laid out by entry: entry e of group g at word e * ng + g, so that every store
 * is a whole vector (one entry of four groups); the row offsets (t3row) follow. Others, and the keys' tables (scan8
 * reads entry g * NE + key byte), by group as everywhere else: entries 0..7 of each block of 9 as two vectors. */
static void tern_build_g(int32_t *t, const int32_t *v, int ngroups) {
    for (int g = 0; g < ngroups; g++, v += 4, t += NE) {
        int32_t p = v[0], q = v[1], r = v[2], s = v[3];
        v128_t a0 = wasm_i32x4_make(0, p, -p, q), a1 = wasm_i32x4_make(q + p, q - p, -q, p - q);
        int32_t a8 = -q - p, b[9] = { 0, r, -r, s, s + r, s - r, -s, r - s, -s - r };
        for (int hi = 0; hi < 9; hi++) {
            v128_t bb = wasm_i32x4_splat(b[hi]);
            wasm_v128_store(t + hi * 9, wasm_i32x4_add(a0, bb)); wasm_v128_store(t + hi * 9 + 4, wasm_i32x4_add(a1, bb));
            t[hi * 9 + 8] = a8 + b[hi];
        }
    }
}
static void tern_build(int32_t *t, const int32_t *v, int ng) {
    if (ng & 3) { tern_build_g(t, v, ng); return; }
    for (int g = 0; g < ng; g += 4, v += 16) {
        v128_t x0 = wasm_v128_load(v), x1 = wasm_v128_load(v + 4), x2 = wasm_v128_load(v + 8), x3 = wasm_v128_load(v + 12);
        v128_t u0 = wasm_i32x4_shuffle(x0, x1, 0, 4, 1, 5), u1 = wasm_i32x4_shuffle(x2, x3, 0, 4, 1, 5);
        v128_t u2 = wasm_i32x4_shuffle(x0, x1, 2, 6, 3, 7), u3 = wasm_i32x4_shuffle(x2, x3, 2, 6, 3, 7);
        v128_t p = wasm_i32x4_shuffle(u0, u1, 0, 1, 4, 5), q = wasm_i32x4_shuffle(u0, u1, 2, 3, 6, 7);     /* inputs 0..3 of the four groups */
        v128_t r = wasm_i32x4_shuffle(u2, u3, 0, 1, 4, 5), s = wasm_i32x4_shuffle(u2, u3, 2, 3, 6, 7), z = wasm_i32x4_splat(0);
        v128_t a[9] = { z, p, wasm_i32x4_neg(p), q, wasm_i32x4_add(q, p), wasm_i32x4_sub(q, p), wasm_i32x4_neg(q), wasm_i32x4_sub(p, q), wasm_i32x4_sub(wasm_i32x4_neg(q), p) };
        v128_t b[9] = { z, r, wasm_i32x4_neg(r), s, wasm_i32x4_add(s, r), wasm_i32x4_sub(s, r), wasm_i32x4_neg(s), wasm_i32x4_sub(r, s), wasm_i32x4_sub(wasm_i32x4_neg(s), r) };
        for (int hi = 0; hi < 9; hi++) for (int lo = 0; lo < 9; lo++) wasm_v128_store(t + (hi * 9 + lo) * ng + g, wasm_i32x4_add(a[lo], b[hi]));
    }
}
#else
static void tern_build(int32_t *t, const int32_t *v, int ngroups) {
    static const int sg[3] = { 0, 1, -1 };
    for (int g = 0; g < ngroups; g++, v += 4)
        for (int i = 0; i < NE; i++)
            *t++ = sg[i % 3] * v[0] + sg[i / 3 % 3] * v[1] + sg[i / 9 % 3] * v[2] + sg[i / 27] * v[3];
}
#endif
/* rows of step bytes (the keys made at run time): each byte moves the table pointer on to the next group's entry */
static void scan8(int32_t *out, const uint8_t *s, const int32_t *tb, int nbytes, int nrows) {
#ifdef __wasm__
    /* (browser) the keys hold each group's entry number itself (att_key_c), not steps: no chain from byte to byte.
     * Sums modulo 2^32 (what is stored), keys of 8 and 12 groups unrolled. */
    for (int r = 0; r < nrows; r++, s += nbytes) {
        uint32_t acc = 0;
        if (nbytes == 8) { for (int g = 0; g < 8; g++) acc += (uint32_t)tb[g * NE + s[g]]; }
        else if (nbytes == 12) { for (int g = 0; g < 12; g++) acc += (uint32_t)tb[g * NE + s[g]]; }
        else for (int g = 0; g < nbytes; g++) acc += (uint32_t)tb[g * NE + s[g]];
        out[r] = (int32_t)acc;
    }
    return;
#endif
    for (int r = 0; r < nrows; r++) {
        const int32_t *p = tb - 1;
        int64_t acc = 0;
        for (int i = 0; i < nbytes; i++) { p += *s++; acc += *p; }
#ifndef __wasm__
        if (acc > pix_maxsum) pix_maxsum = acc;
        if (-acc > pix_maxsum) pix_maxsum = -acc;
#endif
        out[r] = (int32_t)acc;
    }
}
static void axpy8(int32_t *acc, const int8_t *v, int32_t p, int n) { for (int i = 0; i < n; i++) acc[i] += p * v[i]; }
#ifdef __wasm__
/* (browser) the same sums four values at a time, with the state by columns (s0, s1, s2 and the three running means:
 * n words each; nothing else reads it) and the coefficients by columns too (pix_mixcf, copied by pix_begin) */
extern int32_t *const pix_mixcf;
#define MIXC(l, mi) (pix_mixcf + (mi) * D * 6)
static void mixer_k(int32_t *xx, const int32_t *xnn, int32_t *s, const int32_t *cf, int n) {
    static const int32_t k[3] = { 512, 205, 51 };
    v128_t r = wasm_i64x2_splat(1 << (QB - 1));
    for (int d = 0; d < n; d += 4) {
        v128_t x = wasm_v128_load(xnn + d), v[6], lo = wasm_i64x2_splat(0), hi = lo;
        for (int j = 0; j < 6; j++) v[j] = wasm_v128_load(s + j * n + d);
        for (int i = 0; i < 3; i++) {                        /* the running means take in x first */
            v[3 + i] = wasm_i32x4_add(v[3 + i], mulrs4(wasm_i32x4_sub(x, v[3 + i]), wasm_i32x4_splat(k[i]), 10));
            wasm_v128_store(s + (3 + i) * n + d, v[3 + i]);
        }
        for (int j = 0; j < 6; j++) {
            v128_t c = wasm_v128_load(cf + j * n + d);
            lo = wasm_i64x2_add(lo, wasm_i64x2_extmul_low_i32x4(c, v[j])); hi = wasm_i64x2_add(hi, wasm_i64x2_extmul_high_i32x4(c, v[j]));
        }
        lo = wasm_u64x2_shr(wasm_i64x2_add(lo, r), QB); hi = wasm_u64x2_shr(wasm_i64x2_add(hi, r), QB);
        wasm_v128_store(xx + d, wasm_i32x4_add(wasm_v128_load(xx + d), wasm_i32x4_shuffle(lo, hi, 0, 2, 4, 6)));
        wasm_v128_store(s + 2 * n + d, v[1]); wasm_v128_store(s + n + d, v[0]); wasm_v128_store(s + d, x);
    }
}
#else
#define MIXC(l, mi) B32(LAY(l, LF_MIXC))
static void mixer_k(int32_t *xx, const int32_t *xnn, int32_t *s, const int32_t *cf, int n) {
    static const int32_t k[3] = { 512, 205, 51 };
    for (int d = 0; d < n; d++, s += 6, cf += 6) {
        int64_t y = (int64_t)cf[0] * s[0] + (int64_t)cf[1] * s[1] + (int64_t)cf[2] * s[2];
        for (int i = 0; i < 3; i++) {
            s[3 + i] += (int32_t)RS((int64_t)(xnn[d] - s[3 + i]) * k[i], 10);
            y += (int64_t)cf[3 + i] * s[3 + i];
        }
        s[2] = s[1]; s[1] = s[0]; s[0] = xnn[d];
        xx[d] += (int32_t)RS(y, QB);
    }
}
#endif
static void add16(int32_t *a, const int16_t *v, int n) { for (int i = 0; i < n; i++) a[i] += v[i]; }
static void add32(int32_t *a, const int32_t *v, int n) { for (int i = 0; i < n; i++) a[i] += v[i]; }
#ifdef __wasm__
static void scale_add(int32_t *xx, const int32_t *a, int32_t m, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) wasm_v128_store(xx + i, wasm_i32x4_add(wasm_v128_load(xx + i), mulrs4(wasm_v128_load(a + i), wasm_i32x4_splat(m), 16)));
    for (; i < n; i++) xx[i] += (int32_t)RS((int64_t)a[i] * m, 16);
}
#else
static void scale_add(int32_t *xx, const int32_t *a, int32_t m, int n) { for (int i = 0; i < n; i++) xx[i] += (int32_t)RS((int64_t)a[i] * m, 16); }
#endif
/* the head's per-pixel row sums in fewer passes (out may be in) */
static void sum16_3(int32_t *out, const int32_t *in, const int16_t *a, const int16_t *b, const int16_t *c, int n) { for (int i = 0; i < n; i++) out[i] = in[i] + a[i] + b[i] + c[i]; }
static void sum16_4u(int32_t *out, int32_t *ac, const int16_t *a, const int16_t *b, const int16_t *c, const int16_t *d, int n) { for (int i = 0; i < n; i++) { ac[i] += d[i]; out[i] = ac[i] + a[i] + b[i] + c[i]; } }
static void sum16_1w(int32_t *ac, const int32_t *w, const int16_t *a, int n) { for (int i = 0; i < n; i++) ac[i] += w[i] + a[i]; }
#ifdef __wasm__
/* (browser) sixteen values per pass: one load of each row's bytes (out may be in: each group is read before it is written) */
static void axpy8_2(int32_t *out, const int32_t *in, const int8_t *v1, int32_t p1, const int8_t *v2, int32_t p2, int n) {
    v128_t q1 = wasm_i32x4_splat(p1), q2 = wasm_i32x4_splat(p2);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        v128_t a = wasm_v128_load(v1 + i), b = wasm_v128_load(v2 + i);
        v128_t al = wasm_i16x8_extend_low_i8x16(a), ah = wasm_i16x8_extend_high_i8x16(a), bl = wasm_i16x8_extend_low_i8x16(b), bh = wasm_i16x8_extend_high_i8x16(b);
#define AX(k, x, y) wasm_v128_store(out + i + 4 * k, wasm_i32x4_add(wasm_v128_load(in + i + 4 * k), wasm_i32x4_add(wasm_i32x4_mul(q1, x), wasm_i32x4_mul(q2, y))))
        AX(0, wasm_i32x4_extend_low_i16x8(al), wasm_i32x4_extend_low_i16x8(bl));
        AX(1, wasm_i32x4_extend_high_i16x8(al), wasm_i32x4_extend_high_i16x8(bl));
        AX(2, wasm_i32x4_extend_low_i16x8(ah), wasm_i32x4_extend_low_i16x8(bh));
        AX(3, wasm_i32x4_extend_high_i16x8(ah), wasm_i32x4_extend_high_i16x8(bh));
#undef AX
    }
    for (; i < n; i++) out[i] = in[i] + p1 * v1[i] + p2 * v2[i];
}
#else
static void axpy8_2(int32_t *out, const int32_t *in, const int8_t *v1, int32_t p1, const int8_t *v2, int32_t p2, int n) { for (int i = 0; i < n; i++) out[i] = in[i] + p1 * v1[i] + p2 * v2[i]; }
#endif
#ifdef __wasm__
static void head_u(int32_t *uu, const int32_t *hin, const int32_t *f, const int32_t *ts, int n) { for (int i = 0; i < n; i++) { int32_t v = f[i] + ts[i]; uu[i] = hin[i] + RS32(v, 4); } }
#else
static void head_u(int32_t *uu, const int32_t *hin, const int32_t *f, const int32_t *ts, int n) { for (int i = 0; i < n; i++) uu[i] = hin[i] + (int32_t)RS((int64_t)(f[i] + ts[i]), 4); }
#endif
/* the attention mix of one head, all its cells in one pass (sums are 8 bits above Q12, modulo 2^32):
 * out[d] = (128 + sum over a of p * v[d] + sum over b of ((p * v[d] + hf) >> rr)) >> 8; both lists end with v = 0 */
typedef struct { const int8_t *v; int32_t p; } MixA;
typedef struct { const int8_t *v; int32_t p, rr, hf; } MixB;
static void mixk(int32_t *out, const MixA *a, const MixB *b, int ngroups) {
#ifdef __wasm__
    /* (browser) the same sums, four values at a time */
    for (int g = 0; g < ngroups; g++) {
        v128_t s = wasm_i32x4_splat(128);
#define V4(q) wasm_i32x4_extend_low_i16x8(wasm_i16x8_extend_low_i8x16(wasm_v128_load32_zero(q->v + 4 * g)))
        for (const MixA *q = a; q->v; q++) s = wasm_i32x4_add(s, wasm_i32x4_mul(wasm_i32x4_splat(q->p), V4(q)));
        for (const MixB *q = b; q->v; q++) s = wasm_i32x4_add(s, wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_mul(wasm_i32x4_splat(q->p), V4(q)), wasm_i32x4_splat(q->hf)), q->rr));
#undef V4
        wasm_v128_store(out + 4 * g, wasm_i32x4_shr(s, 8));
    }
    return;
#endif
    for (int d = 0; d < ngroups * 4; d++) {
        uint32_t s = 128;
        for (const MixA *q = a; q->v; q++) s += (uint32_t)q->p * (uint32_t)(int32_t)q->v[d];
        for (const MixB *q = b; q->v; q++) s += (uint32_t)((q->p * q->v[d] + q->hf) >> q->rr);
        out[d] = (int32_t)s >> 8;
    }
}
/* small loops of the per-cell glue, as kernels so that they run in fast RAM */
static void copy32(int32_t *dst, const int32_t *src, int n) { for (int i = 0; i < n; i++) dst[i] = src[i]; }
#ifdef __wasm__
static void rs4_k(int32_t *xx, int n) { for (int i = 0; i < n; i++) xx[i] = RS32(xx[i], 4); }
static void mulrs_k(int32_t *out, const int32_t *a, const int32_t *b, int n, int shift) {
    int i = 0;
    if (shift > 0 && shift <= 32) for (; i + 4 <= n; i += 4) wasm_v128_store(out + i, mulrs4(wasm_v128_load(a + i), wasm_v128_load(b + i), shift));
    for (; i < n; i++) out[i] = (int32_t)RS((int64_t)a[i] * b[i], shift);
}
static void scale_rs(int32_t *out, const int32_t *in, int32_t m, int n, int shift) {
    int i = 0;
    if (shift > 0 && shift <= 32) for (; i + 4 <= n; i += 4) wasm_v128_store(out + i, mulrs4(wasm_v128_load(in + i), wasm_i32x4_splat(m), shift));
    for (; i < n; i++) out[i] = (int32_t)RS((int64_t)in[i] * m, shift);
}
static uint64_t sumsq_k(const int32_t *in, int n) {        /* (the same sum modulo 2^64) */
    v128_t s = wasm_i64x2_splat(0);
    int i = 0;
    for (; i + 4 <= n; i += 4) { v128_t v = wasm_v128_load(in + i); s = wasm_i64x2_add(s, wasm_i64x2_add(wasm_i64x2_extmul_low_i32x4(v, v), wasm_i64x2_extmul_high_i32x4(v, v))); }
    uint64_t ss = (uint64_t)wasm_i64x2_extract_lane(s, 0) + (uint64_t)wasm_i64x2_extract_lane(s, 1);
    for (; i < n; i++) ss += (uint64_t)((int64_t)in[i] * in[i]);
    return ss;
}
#else
static void rs4_k(int32_t *xx, int n) { for (int i = 0; i < n; i++) xx[i] = (int32_t)RS((int64_t)xx[i], 4); }
static void mulrs_k(int32_t *out, const int32_t *a, const int32_t *b, int n, int shift) { for (int i = 0; i < n; i++) out[i] = (int32_t)RS((int64_t)a[i] * b[i], shift); }
static void scale_rs(int32_t *out, const int32_t *in, int32_t m, int n, int shift) { for (int i = 0; i < n; i++) out[i] = (int32_t)RS((int64_t)in[i] * m, shift); }
static uint64_t sumsq_k(const int32_t *in, int n) {
    uint64_t ss = 0;
    for (int i = 0; i < n; i++) ss += (uint64_t)((int64_t)in[i] * in[i]);
    return ss;
}
#endif
#endif
/* where a T3 row's ng groups land in the tables (entry numbers from the tables' start) */
ROM_CODE static void t3pos(const uint8_t *row, int ng, int32_t *pos) {
    const uint32_t *s = (const uint32_t *)row;
    int nw = T3W(ng), k = ng - 3 * (nw - 1), p = -1, g = 0;
    for (int j = 0; j < nw; j++, k = 3) {
        uint32_t w = s[j];
        if (k >= 2) pos[g++] = p + (int32_t)(w >> 11 & 0xFF);
        p += (int32_t)(w >> 24); pos[g++] = p;
        if (k == 3) { p += (int32_t)(w & 0xFF); pos[g++] = p; }
    }
}
#ifdef __wasm__
/* The browser build reads T3 rows as table offsets: one uint16 per group, the byte offset of its entry from the tables'
 * start, so that the row loops load an offset and an entry per group and follow no steps. Each call's rows (address,
 * groups, rows) are turned into offsets once and kept in a cache outside the engine's state (pixwasm.c: pix_t3c), which
 * saved states therefore do not touch: it holds nothing but what the model file gives. */
extern uint8_t *const pix_t3c;
extern const uint32_t pix_t3c_len;
typedef struct { const uint8_t *s; uint32_t shape; const uint16_t *o; } T3C;
#define T3C_SLOTS 4096                                       /* hash table at the cache's start, a spare row block, then the offsets */
#define T3C_SPARE (T3C_SLOTS * sizeof(T3C) + 16)
#define T3C_SPARE_ROWS 4096
#define T3C_DATA (T3C_SPARE + T3C_SPARE_ROWS * CHG * 2)
void pix_t3c_reset(void) { for (uint32_t i = 0; i < T3C_SPARE / 4; i++) ((uint32_t *)pix_t3c)[i] = 0; }
static T3C *t3c_find(const uint8_t *s, uint32_t shape) {      /* the slot of (s, shape), or the free slot for it (o = 0) */
    T3C *h = (T3C *)pix_t3c;
    for (uint32_t i = ((uint32_t)s * 2654435761u) >> 20;; i = (i + 1) & (T3C_SLOTS - 1))
        if ((h[i].s == s && h[i].shape == shape) || !h[i].s) return h + i;
}
static void *t3c_new(T3C *e, const uint8_t *s, uint32_t shape, uint32_t bytes) {      /* room for a new entry, or 0: full */
    uint32_t *used = (uint32_t *)((T3C *)pix_t3c + T3C_SLOTS), need = (bytes + 15) & ~15u;
    if (used[1] >= T3C_SLOTS / 2 || T3C_DATA + used[0] + need > pix_t3c_len) return 0;
    void *o = pix_t3c + T3C_DATA + used[0];
    used[0] += need; used[1]++;
    e->s = s; e->shape = shape; e->o = o;
    return o;
}
/* a T3 row's offsets in the tables tern_build makes: laid out by entry for a multiple of 4 groups (see there) */
static void t3row(const uint8_t *row, int ng, uint16_t *o) {
    int32_t pos[CHG];
    t3pos(row, ng, pos);
    for (int g = 0; g < ng; g++) o[g] = (uint16_t)(4 * ((ng & 3) ? pos[g] : (pos[g] - g * NE) * ng + g));
}
static const uint16_t *t3off(const uint8_t *s, int ng, int nrows) {      /* (ng up to CHG) */
    uint32_t shape = (uint32_t)ng << 16 | (uint32_t)nrows;
    T3C *e = t3c_find(s, shape);
    if (e->s) return e->o;
    uint16_t *o = t3c_new(e, s, shape, (uint32_t)nrows * ng * 2);
    if (!o) {                                                /* full (not with the models so far): made again each time */
        if (nrows > T3C_SPARE_ROWS) __builtin_trap();
        o = (uint16_t *)(pix_t3c + T3C_SPARE);
    }
    for (int r = 0; r < nrows; r++) t3row(s + (uint32_t)r * ROWB(ng), ng, o + r * ng);
    return o;
}
#define TE(t, o) (*(const int32_t *)((const uint8_t *)(t) + (o)))      /* the entry at byte offset o */
/* (a row's offsets: an opaque pointer, so that the compiler does not make it a running one, whose loads lose their
 * constant offsets) */
#define OROW(o, r, ng) ({ const uint16_t *q_ = (o) + (r) * (ng); __asm__("" : "+r"(q_)); q_; })
/* (in unrolled rows, after every four groups: keeps the compiler from loading all of a row's entries first, which
 * leaves the engine more values than registers) */
#define LOADS_HERE() __asm__("" ::: "memory")
#endif
/* out[r] = sum of the table entries of row r's groups: the twin's row scan; on the GBA the kernels' fallback for row
 * lengths without an unrolled loop (tern_scan jumps here; no model so far has such rows) */
#ifdef __arm__
ROM_CODE void scan_c(int32_t *out, const uint8_t *s, const int32_t *tb, int ng, int nrows)
#else
static void scan(int32_t *out, const uint8_t *s, const int32_t *tb, int ng, int nrows)
#endif
{
#ifdef __wasm__
    /* the browser build: four sums of the entries at the rows' offsets (t3off) */
    const uint16_t *o = t3off(s, ng, nrows);
    for (int r = 0; r < nrows; r++) {
        const uint16_t *q = OROW(o, r, ng); uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0; int g = 0;
        for (; g + 3 < ng; g += 4) { a0 += (uint32_t)TE(tb, q[g]); a1 += (uint32_t)TE(tb, q[g + 1]); a2 += (uint32_t)TE(tb, q[g + 2]); a3 += (uint32_t)TE(tb, q[g + 3]); }
        for (; g < ng; g++) a0 += (uint32_t)TE(tb, q[g]);
        out[r] = (int32_t)((a0 + a1) + (a2 + a3));
    }
    return;
#endif
    int32_t pos[CHG];
    for (int r = 0; r < nrows; r++, s += ROWB(ng)) {
        t3pos(s, ng, pos);
#ifdef __arm__
        int32_t acc = 0;
        for (int g = 0; g < ng; g++) acc += tb[pos[g]];
        out[r] = acc;
#else
        int64_t acc = 0;
        for (int g = 0; g < ng; g++) acc += tb[pos[g]];
        if (acc > pix_maxsum) pix_maxsum = acc;
        if (-acc > pix_maxsum) pix_maxsum = -acc;
        out[r] = (int32_t)acc;
#endif
    }
}
/* attention: the up to 8 best of sc[j] + bias[n - 1 - j] over j0 <= j < n, leaving out cells marked in skip (may be 0).
 * tv = 9 pairs (score, j): pair 0 is a stop for the kernel's insertion walk (the largest score there is), pairs
 * 1..8 the best, best first (equal scores: lower j first). Returns how many.
 * Walks j downwards so the bias row is read forwards.
 * On the GBA keytop (kernels.s) does the key scan and this in one pass for keys of 2 or 3 words of steps, and the
 * scores are never stored; this version is the reference (the twin) and serves other key lengths from ROM. */
__attribute__((noinline)) ROM_CODE static int topsel(const int32_t *s, const int32_t *bias, int j0, int n, int32_t *tv, const uint8_t *skip) {
    int k = 0;
#ifdef __wasm__
    /* (browser) the same 8 in the same order, kept in vectors: scores va (best 4) and vb, rows ja and jb; empty places
     * hold the lowest score, which every insertion passes (as topsel's walk ends at k). An insertion keeps the places
     * with a higher score than v and moves the rest down one, without branches. Rows are first listed per block of 16
     * (highest first, without branches: lane_lut) against the 8th best at the block's start, which only rises. */
    v128_t va = wasm_i32x4_splat(-0x7FFFFFFF - 1), vb = va, ja = wasm_i32x4_splat(0), jb = ja;
    int j = n - 1, cand[20];
#define TOP1(j) { int32_t v = s[j] + bias[n - 1 - (j)]; \
        if (!(k == 8 && v < wasm_i32x4_extract_lane(vb, 3)) && !(skip && skip[j])) { \
            v128_t x = wasm_i32x4_splat(v), y = wasm_i32x4_splat(j); \
            v128_t ma = wasm_i32x4_gt(va, x), mb = wasm_i32x4_gt(vb, x); \
            v128_t pa = wasm_i32x4_shuffle(ma, ma, 4, 0, 1, 2), pb = wasm_i32x4_shuffle(ma, mb, 3, 4, 5, 6); \
            pa = wasm_v128_or(pa, wasm_i32x4_make(-1, 0, 0, 0)); \
            v128_t sa = wasm_i32x4_shuffle(va, x, 4, 0, 1, 2), sb = wasm_i32x4_shuffle(va, vb, 3, 4, 5, 6); \
            v128_t ta = wasm_i32x4_shuffle(ja, y, 4, 0, 1, 2), tb_ = wasm_i32x4_shuffle(ja, jb, 3, 4, 5, 6); \
            vb = wasm_v128_bitselect(vb, wasm_v128_bitselect(x, sb, pb), mb); jb = wasm_v128_bitselect(jb, wasm_v128_bitselect(y, tb_, pb), mb); \
            va = wasm_v128_bitselect(va, wasm_v128_bitselect(x, sa, pa), ma); ja = wasm_v128_bitselect(ja, wasm_v128_bitselect(y, ta, pa), ma); \
            if (k < 8) k++; } }
    for (; j >= j0 && k < 8; j--) TOP1(j);
    while (j - 15 >= j0) {
        v128_t lim = wasm_i32x4_splat(wasm_i32x4_extract_lane(vb, 3)), ix = wasm_i32x4_make(j, j - 1, j - 2, j - 3);
        int nc = 0;
        for (int g = 0; g < 4; g++, j -= 4, ix = wasm_i32x4_sub(ix, wasm_i32x4_splat(4))) {
            v128_t sv = wasm_v128_load(s + j - 3), v = wasm_i32x4_add(wasm_i32x4_shuffle(sv, sv, 3, 2, 1, 0), wasm_v128_load(bias + n - 1 - j));
            uint32_t b = wasm_i32x4_bitmask(wasm_i32x4_ge(v, lim));
            wasm_v128_store(cand + nc, wasm_i8x16_swizzle(ix, wasm_v128_load(lane_lut[b])));
            nc += __builtin_popcount(b);
        }
        for (int c = 0; c < nc; c++) TOP1(cand[c]);
    }
    for (; j >= j0; j--) TOP1(j);
#undef TOP1
    int32_t o[16];
    wasm_v128_store(o, va); wasm_v128_store(o + 4, vb); wasm_v128_store(o + 8, ja); wasm_v128_store(o + 12, jb);
    for (int i = 0; i < k; i++) { tv[2 + 2 * i] = o[i]; tv[3 + 2 * i] = o[8 + i]; }
    return k;
#endif
    for (int j = n - 1; j >= j0; j--) {
        int32_t v = s[j] + bias[n - 1 - j];
        if (k == 8 && v < tv[16]) continue;
        if (skip && skip[j]) continue;
        int p = k < 8 ? ++k : 8;
        while (p > 1 && tv[2 * p - 2] <= v) { tv[2 * p] = tv[2 * p - 2]; tv[2 * p + 1] = tv[2 * p - 1]; p--; }
        tv[2 * p] = v; tv[2 * p + 1] = j;
    }
    return k;
}

/* The matrix after a squared ReLU, where most inputs are zero, is run transposed through the same 81-entry tables:
 * each active input adds its value to one table entry per 4 output rows (its column's step bytes), then the tables
 * are folded back into the outputs. */
static int32_t aval[MAXIN] VRAM2_DATA;
#define acc proj
static uint16_t aidx[MAXIN] VRAM2_DATA;
#ifdef __arm__
void scatter32(int32_t *tbl, const uint8_t *cols, const int32_t *vals, const uint16_t *idx, int n);
void scatter48(int32_t *tbl, const uint8_t *cols, const int32_t *vals, const uint16_t *idx, int n);
void qbuild16(int32_t *tbl, const int32_t *x, int ngroups, int32_t m);
static inline void tern_scatter(int32_t *t, const uint8_t *cols, const int32_t *vals, const uint16_t *idx, int n, int ng) {
    if (n > 0) { if (ng == 32) scatter32(t, cols, vals, idx, n); else scatter48(t, cols, vals, idx, n); }      /* (pix_begin checks: D and DH are 128 or 192) */
}
void tern_unbuild(int32_t *out, const int32_t *tbl, int ngroups, int32_t m, const int32_t *base);
void zero32(int32_t *p, int nwords8);
int relu_sq(int32_t *vals, uint16_t *idx, const int32_t *in, int n, int32_t m, int shift, int32_t in0);
void packk(int32_t *v, const int32_t *d, int n, int k);
/* The MLP's first matrix with the squared ReLU applied to each row sum as it is ready (rows of 8 or 12 words): the
 * sums are never stored. The tables' first group is built with -in0 added to every entry (each row takes exactly one
 * entry of each group), so "kept" is the sign of the row sum, or of each half with two streams (-in0 in both halves).
 * One stream (tern_scan_x, rows of 8 or 12 words, with the early exit W1X below): vals / idx get the active units as
 * relu_sq would. Two streams (tern_scan_relu, rows of 8 words): the lists and sums of relu_sq2.
 * (The kernels copy what they need to their stack and leave the sums.) */
typedef struct { int32_t m, shift, sh32, round, in0, rows; uint32_t idxc, difoff; int32_t two; uint32_t sum0, sum1; } ScanR;
int tern_scan_relu(int32_t *vals, const uint8_t *steps, const int32_t *tbl, int nwords, ScanR *r);
int tern_scan_x(int32_t *vals, const uint8_t *steps, const int32_t *tbl, int nwords, ScanR *r);
void tern_build81b(int32_t *tbl, const int32_t *x, int ngroups, int32_t bias);
void tern_build81x(int32_t *tbl, const int32_t *x, int ngroups, const int32_t *bias);
static int32_t w1xb[PIX_MAX_L][8] EWRAM_DATA;                /* tern_scan_x's limits as table biases, per layer (pix_begin) */
static int32_t w1xb2[PIX_MAX_L][8] EWRAM_DATA;               /* the same for two streams (tern_scan_relu, PIX_W1XS0) */
static ScanR scanr[2][PIX_MAX_L] EWRAM_DATA;                 /* per layer, for one stream and for two: constant through a picture */
void scale_add2(int32_t *x0, int32_t *x1, const int32_t *in, int32_t m, int n, int k);
#else
static void t3pos(const uint8_t *row, int ng, int32_t *pos);
static void tern_scatter(int32_t *tb, const uint8_t *cols, const int32_t *vals, const uint16_t *idx, int n, int ng) {
#ifdef __wasm__
    int nw = T3W(ng), k0 = ng - 3 * (nw - 1);
    for (int i = 0; i < n; i++) {
        const uint32_t *w = (const uint32_t *)(cols + (uint32_t)idx[i] * ROWB(ng)); int32_t *p = tb - 1, v = vals[i];
        for (int j = 0, k = k0; j < nw; j++, k = 3) {
            uint32_t x = w[j];
            if (k >= 2) p[x >> 11 & 0xFF] += v;
            p += x >> 24; *p += v;
            if (k == 3) { p += x & 0xFF; *p += v; }
        }
    }
    return;
#endif
    int32_t pos[CHG];
    for (int k = 0; k < n; k++) {
        t3pos(cols + (uint32_t)idx[k] * ROWB(ng), ng, pos);
        for (int i = 0; i < ng; i++) tb[pos[i]] += vals[k];
    }
}
static void tern_unbuild(int32_t *out, const int32_t *t, int ngroups, int32_t m) {
#ifdef __wasm__
    /* the same sums as below, four at a time: lane k of sv[i] is entry i's sign for output k (the browser build) */
    static v128_t sv[NE]; static int ready;
    if (!ready) { static const int sg[3] = { 0, 1, -1 }; for (int i = 0; i < NE; i++) sv[i] = wasm_i32x4_make(sg[i % 3], sg[i / 3 % 3], sg[i / 9 % 3], sg[i / 27]); ready = 1; }
    for (int g = 0; g < ngroups; g++, t += NE, out += 4) {
        v128_t acc = wasm_i32x4_splat(0);
        for (int i = 0; i < NE; i++) acc = wasm_i32x4_add(acc, wasm_i32x4_mul(sv[i], wasm_i32x4_splat(t[i])));
        int32_t a = wasm_i32x4_extract_lane(acc, 0), b = wasm_i32x4_extract_lane(acc, 1), c = wasm_i32x4_extract_lane(acc, 2), d = wasm_i32x4_extract_lane(acc, 3);
#else
    static const int sg[3] = { 0, 1, -1 };
    for (int g = 0; g < ngroups; g++, t += NE, out += 4) {
        int32_t a = 0, b = 0, c = 0, d = 0;
        for (int i = 0; i < NE; i++) { a += sg[i % 3] * t[i]; b += sg[i / 3 % 3] * t[i]; c += sg[i / 9 % 3] * t[i]; d += sg[i / 27] * t[i]; }
#endif
        if (!m) { out[0] = a; out[1] = b; out[2] = c; out[3] = d; continue; }
        out[0] += (int32_t)RS((int64_t)a * m, 16); out[1] += (int32_t)RS((int64_t)b * m, 16);
        out[2] += (int32_t)RS((int64_t)c * m, 16); out[3] += (int32_t)RS((int64_t)d * m, 16);
    }
}
static void zero32(int32_t *p, int n8) { for (int i = 0; i < n8 * 8; i++) p[i] = 0; }
#endif
#ifdef __wasm__
/* The browser build's sparse_out has tables of its own (stbl: after tbl, left clear after each call), laid out by entry
 * as tern_build's (entry e of group g at word e * ng + g), so that one vector holds entry e of four groups and the
 * fold-back is adds and subtracts of whole vectors (the signs are constants once the loop over the 81 entries is
 * unrolled). Each input's column (a T3 row) is turned into offsets (t3row) the first time it is active and kept in the
 * cache: per matrix, a flag byte per input, then the inputs' offsets. */
#define stbl (tbl + CHG * NE)
static uint8_t *t3cols(const uint8_t *cols, int ng) {
    uint32_t shape = (uint32_t)ng << 16 | 0xFFFF;
    T3C *e = t3c_find(cols, shape);
    uint8_t *f = (uint8_t *)e->o;
    if (!e->s && (f = t3c_new(e, cols, shape, MAXIN + MAXIN * ng * 2))) for (int k = 0; k < MAXIN; k++) f[k] = 0;
    return f;
}
/* (0: not done, the caller does it the plain way) */
static int sparse_w(int32_t *out, const uint8_t *cols, const int32_t *vals, int n, int ng, int32_t m) {
    if ((ng & 3) || ng > CHG) return 0;
    uint8_t *f = t3cols(cols, ng);
    if (!f) return 0;
    const uint16_t *o0 = (const uint16_t *)(f + MAXIN);
    for (int k = 0; k < n; k++) if (!f[aidx[k]]) { t3row(cols + aidx[k] * ROWB(ng), ng, (uint16_t *)o0 + aidx[k] * ng); f[aidx[k]] = 1; }
    uint8_t *t = (uint8_t *)stbl;
    for (int k = 0; k < n; k++) {
        const uint16_t *o = OROW(o0, aidx[k], ng); int32_t v = vals[k];
        for (int g = 0; g < ng; g += 4) { *(int32_t *)(t + o[g]) += v; *(int32_t *)(t + o[g + 1]) += v; *(int32_t *)(t + o[g + 2]) += v; *(int32_t *)(t + o[g + 3]) += v; }
    }
    static const int sd[3] = { 0, 1, -1 };
    const v128_t zero = wasm_i32x4_splat(0);
    for (int g = 0; g < ng; g += 4, out += 16) {            /* four groups: c[k] lane j = output k of group g + j */
        v128_t c[4] = { zero, zero, zero, zero };
        _Pragma("clang loop unroll(full)")
        for (int e = 0; e < NE; e++) {
            v128_t *p = (v128_t *)(stbl + e * ng + g), x = wasm_v128_load(p);
            wasm_v128_store(p, zero);                        /* (left clear, as zero32 would) */
            const int d[4] = { sd[e % 3], sd[e / 3 % 3], sd[e / 9 % 3], sd[e / 27] };
            for (int k = 0; k < 4; k++) c[k] = d[k] > 0 ? wasm_i32x4_add(c[k], x) : d[k] < 0 ? wasm_i32x4_sub(c[k], x) : c[k];
        }
        v128_t u0 = wasm_i32x4_shuffle(c[0], c[1], 0, 4, 1, 5), u1 = wasm_i32x4_shuffle(c[2], c[3], 0, 4, 1, 5);
        v128_t u2 = wasm_i32x4_shuffle(c[0], c[1], 2, 6, 3, 7), u3 = wasm_i32x4_shuffle(c[2], c[3], 2, 6, 3, 7);
        v128_t r[4] = { wasm_i32x4_shuffle(u0, u1, 0, 1, 4, 5), wasm_i32x4_shuffle(u0, u1, 2, 3, 6, 7),
                        wasm_i32x4_shuffle(u2, u3, 0, 1, 4, 5), wasm_i32x4_shuffle(u2, u3, 2, 3, 6, 7) };
        if (!m) { for (int j = 0; j < 4; j++) wasm_v128_store(out + 4 * j, r[j]); continue; }
        int32_t a[16];
        for (int j = 0; j < 4; j++) wasm_v128_store(a + 4 * j, r[j]);
        for (int i = 0; i < 16; i++) out[i] += (int32_t)RS((int64_t)a[i] * m, 16);
    }
    return 1;
}
#endif
/* out[0..nout) += rs(m * (transposed matrix (cols: one T3 row of nout / 4 groups per input) times the n active inputs
 * (vals, aidx)), 16): the fold-back scales and adds on the way. With m = 0 the plain sums are stored instead.
 * (Three kernel calls: the glue itself need not be in fast RAM.)
 * base (PIX_KEEPTBL): the caller knows that the tables still hold tern_build of base[0..nout) (plus any constant per
 * group, such as tern_build81b's threshold). The fold-back is linear and turns that table into 54 * base, so the
 * scatter goes on top of it and the fold-back takes 54 * base off again, instead of clearing the tables first.
 * All sums are modulo 2^32 either way, so the result is the same to the bit. The twin clears. */
#define PIX_KEEPTBL 1
__attribute__((noinline)) static void sparse_out(int32_t *out, const uint8_t *cols, const int32_t *vals, int n, int nout, int32_t m, const int32_t *base) {
#ifdef __arm__
    if (!PIX_KEEPTBL) base = 0;
    if (!base) zero32(tbl, (nout / 4 * NE + 7) / 8);
    tern_scatter(tbl, cols, vals, aidx, n, nout / 4);
    tern_unbuild(out, tbl, nout / 4, m, base);
#else
    (void)base;
#ifdef __wasm__
    if (sparse_w(out, cols, vals, n, nout / 4, m)) return;
#endif
    zero32(tbl, (nout / 4 * NE + 7) / 8);
    tern_scatter(tbl, cols, vals, aidx, n, nout / 4);
    tern_unbuild(out, tbl, nout / 4, m);
#endif
}
#ifndef __arm__
long long pix_nact;
#ifndef __wasm__
/* hidden units: p = rs(in * m, 16); kept when positive and rs(p * p, shift) is above tau (small ones change nothing visible) */
static int relu_sq(int32_t *vals, uint16_t *idx, const int32_t *in, int n, int32_t m, int shift, int32_t tau) {
    int k = 0;
    for (int i = 0; i < n; i++) {
        int64_t p = RS((int64_t)in[i] * m, 16);
        if (p <= 0) continue;
        int32_t v = (int32_t)RS(p * p, shift);
        if (v > tau) { vals[k] = v; idx[k++] = (uint16_t)i; pix_nact++; }
    }
    return k;
}
#endif
static void packk(int32_t *v, const int32_t *d, int n, int k) { for (int i = 0; i < n; i++) v[i] = (int32_t)(((uint32_t)v[i] << k) + (uint32_t)d[i]); }
/* in = a * 2^k + d with d in [-2^(k-1), 2^(k-1)):  x0 += rs(a * m, 16),  x1 += rs((a + d) * m, 16) */
static void scale_add2(int32_t *x0, int32_t *x1, const int32_t *in, int32_t m, int n, int k) {
#ifdef __wasm__
    if (!(n & 3)) {                                          /* (browser) four lanes at a time */
        v128_t vm = wasm_i32x4_splat(m), h = wasm_i32x4_splat(1 << (k - 1));
        for (int i = 0; i < n; i += 4) {
            v128_t v = wasm_v128_load(in + i), a = wasm_i32x4_shr(wasm_i32x4_add(v, h), k), d = wasm_i32x4_sub(v, wasm_i32x4_shl(a, k));
            wasm_v128_store(x0 + i, wasm_i32x4_add(wasm_v128_load(x0 + i), mulrs4(a, vm, 16)));
            wasm_v128_store(x1 + i, wasm_i32x4_add(wasm_v128_load(x1 + i), mulrs4(wasm_i32x4_add(a, d), vm, 16)));
        }
        return;
    }
#endif
    for (int i = 0; i < n; i++) {
        int32_t a = (in[i] + (1 << (k - 1))) >> k, d = in[i] - (int32_t)((uint32_t)a << k);
        x0[i] += (int32_t)RS((int64_t)a * m, 16); x1[i] += (int32_t)RS((int64_t)(a + d) * m, 16);
    }
}
#endif

#ifdef __arm__
uint32_t isqrt64(uint64_t v);
#else
static uint32_t isqrt64(uint64_t v) {
#ifdef __wasm__
    /* (browser) the same floor(sqrt(v)): the float root, then made exact */
    uint64_t q = (uint64_t)__builtin_sqrt((double)v);
    if (q > 0xFFFFFFFFu) q = 0xFFFFFFFFu;
    while (q * q > v) q--;
    while (q < 0xFFFFFFFFu && (q + 1) * (q + 1) <= v) q++;
    return (uint32_t)q;
#endif
    uint64_t r = 0, b = (uint64_t)1 << 62;
    while (b > v) b >>= 2;
    while (b) {
        if (v >= r + b) { v -= r + b; r = (r >> 1) + b; } else r >>= 1;
        b >>= 2;
    }
    return (uint32_t)r;
}
#endif
/* A hidden unit is kept when rs(p * p, shift) > tau with p = rs(in * m, 16). That is p >= p0, the smallest p whose
 * square reaches (2 tau + 1) << (shift - 1) (relu_p0, once per picture), and that is in >= ceil(((p0 << 16) - 32768) / m)
 * (relu_in0, once per call; m is positive): the kernels decide with one compare per input and multiply only the kept. */
ROM_CODE static uint32_t relu_p0(int shift, int32_t tau) {
    uint64_t T = (uint64_t)(2 * (int64_t)tau + 1) << (shift - 1);
    return isqrt64(T - 1) + 1;
}
#ifdef __arm__
uint32_t udivr(uint32_t alo, uint32_t ahi, uint32_t r);        /* kernels.s: division-free floor(A / r) (see div_shl) */
#endif
static int32_t relu_in0(uint32_t p0, int32_t m) {
    if (m <= 0) return 0x7FFFFFFF;
#if defined(__arm__) && !defined(PIX_DIVSHL_OLD)
    if (p0 < 32768 && m >= 64 && m < (1 << 24)) return (int32_t)udivr(((p0 << 16) - 32768) + (uint32_t)m - 1, 0, (uint32_t)m);     /* below 2^31 */
#endif
    if (p0 < 32768) return (int32_t)((((p0 << 16) - 32768) + (uint32_t)m - 1) / (uint32_t)m);
    uint64_t q = ((((uint64_t)p0 << 16) - 32768) + (uint32_t)m - 1) / (uint32_t)m;
    return q > 0x7FFFFFFF ? 0x7FFFFFFF : (int32_t)q;
}
#ifdef __wasm__
/* (browser) relu_sq as in the twin, the same units: every unit it keeps has in >= in0 (relu_in0), so four inputs are
 * tested per compare and only those that pass get the twin's own test */
static int32_t relu_in0t(int shift, int32_t tau, int32_t m) {
    static int32_t lsh, ltau; static uint32_t lp0;          /* (relu_p0 of the last shift and tau) */
    if (m <= 0 || tau < 0) return -0x7FFFFFFF - 1;           /* (every input gets the test) */
    if (shift != lsh || tau != ltau || !lp0) { lp0 = relu_p0(shift, tau); lsh = shift; ltau = tau; }
    return relu_in0(lp0, m);
}
/* the numbers of the words of in[0..n) with one of their halves (lo: the low halves; else whole words) at least
 * in0, into list[0..), without branches (lane_lut; list has room for n + 3); returns how many. Words past the last
 * four are all taken. */
static int pass_list(int32_t *list, const int32_t *in, int n, int32_t in0, int lo) {
    v128_t t = wasm_i32x4_splat(in0), ix = wasm_i32x4_make(0, 1, 2, 3);
    int nc = 0, i = 0;
    for (; i + 4 <= n; i += 4, ix = wasm_i32x4_add(ix, wasm_i32x4_splat(4))) {
        v128_t w = wasm_v128_load(in + i), ok;
        if (lo) { v128_t l = wasm_i32x4_shr(wasm_i32x4_shl(w, 16), 16); ok = wasm_v128_or(wasm_i32x4_ge(l, t), wasm_i32x4_ge(wasm_i32x4_shr(wasm_i32x4_sub(w, l), 16), t)); }
        else ok = wasm_i32x4_ge(w, t);
        uint32_t b = wasm_i32x4_bitmask(ok);
        wasm_v128_store(list + nc, wasm_i8x16_swizzle(ix, wasm_v128_load(lane_lut[b])));
        nc += __builtin_popcount(b);
    }
    for (; i < n; i++) list[nc++] = i;
    return nc;
}
static int relu_sq(int32_t *vals, uint16_t *idx, const int32_t *in, int n, int32_t m, int shift, int32_t tau) {
    /* first the inputs that pass in >= in0 (their numbers into vals), then the twin's test on those (vals[k], k <= c,
     * is written after vals[c] is read) */
    int nc = pass_list(vals, in, n, relu_in0t(shift, tau, m), 0), k = 0;
    for (int c = 0; c < nc; c++) {
        int j = vals[c];
        int64_t p = RS((int64_t)in[j] * m, 16);
        if (p > 0) { int32_t v = (int32_t)RS(p * p, shift); if (v > tau) { vals[k] = v; idx[k++] = (uint16_t)j; } }
    }
    return k;
}
#endif
#ifdef __arm__
#define RT(p0, tau, m) relu_in0(p0, m)
#else
#define RT(p0, tau, m) (tau)
#endif
/* ---- The first MLP matrix with early exit ----
 * A row of the first matrix whose partial sum, at one of seven check points (28, 24, ... 4 groups of four inputs
 * still to come), is below that point's limit is dropped there: its hidden unit counts as inactive. The limits
 * (LF_W1X, seven per layer) come from calibration runs (tools/w1cal.py): they save as many lookups as possible while
 * the rows that end up active from below a limit carry a set, small share of the layer's hidden energy.
 * Two streams (LF_W1X2): the tables' high half holds stream 0 + stream 1, and a row is dropped for both streams
 * when that sum is below the limit; rows that reach the end are taken apart and judged per stream as before.
 * (Exactly: with the packed sum so far w, stream 1's part in the low half, the test is
 * (int32_t)(w + ((uint32_t)-limit << 16) - in0) < 0: that is what the kernel's tables make of it.)
 * This is an approximation; W1X = 0 turns it off (both builds then do exactly what they did before). */
#ifndef W1X
#define W1X 1
#endif
#ifndef W1X2
#define W1X2 0                                               /* 1: also with two streams (guided cells). Twin only, for study: the GBA kernel for it was not merged */
#endif
#define W1X_NCK 7
/* (B) approximation, switchable: PIX_W1XS0 1 = guided cells (two streams) take the early exit too, decided by stream 0's
 * half of the packed row sum against the one-stream limits (scaled by 2^-XSH): a row dropped for the prompt stream is
 * dropped for both. Measured with tools/gkl.py on top of PIX_G2B (see DESIGN.md). 0 = no exit with two streams. */
#ifndef PIX_W1XS0
#define PIX_W1XS0 1
#endif
#if defined(__arm__) && W1X2
#error "W1X2 has no GBA kernel (see DESIGN.md)"
#endif
#ifndef __arm__
int pix_w1x = W1X, pix_w1x2 = W1X2, pix_w1xs0 = PIX_W1XS0;
long long pix_w1x_look[4];
#ifndef __wasm__                                            /* (the browser build has scan_exit_relu) */
static void scan_exit(int32_t *out, const uint8_t *s, const int32_t *tb, int ng, int nrows, const int32_t *lim) {
    int32_t pos[CHG];
    for (int r = 0; r < nrows; r++, s += ROWB(ng)) {
        int64_t acc = 0;
        int g;
        t3pos(s, ng, pos);
        for (g = 0; g < ng; g++) {
            acc += tb[pos[g]];
            int rem = ng - 1 - g;                            /* groups still to come */
            if (rem && rem <= 28 && !(rem & 3) && acc < lim[7 - rem / 4]) break;
        }
        pix_w1x_look[0] += g < ng ? g + 1 : ng; pix_w1x_look[1] += ng;
        out[r] = g < ng ? -0x40000000 : (int32_t)acc;
    }
}
#else
/* The browser build: scan_exit and relu_sq of its rows in one pass (as the GBA's tern_scan_x), over the rows' offsets
 * (t3off) and unrolled for 32 or 48 groups. No branches per row: every row follows all its groups (the exit checks
 * would be mispredicted often, and a row out at a check point stays out), and each row's hidden unit is written to
 * aval / aidx at k, which moves on when it is kept. Out, aval, aidx and the count are relu_sq's of scan_exit's rows
 * (dropped rows included). */
static inline __attribute__((always_inline)) int scan_exit_relu_n(int32_t *out, const uint16_t *o, const int32_t *tb, const int ng, int nrows, const int32_t *lim, int32_t m, int shift, int32_t tau) {
    int k = 0;
    for (int r = 0; r < nrows; r++) {
        const uint16_t *q = OROW(o, r, ng); int64_t acc = 0; int out_ = 0;
        _Pragma("clang loop unroll(full)")
        for (int g = 0; g < ng; g++) { acc += TE(tb, q[g]); int rem = ng - 1 - g; if (rem && rem <= 28 && !(rem & 3)) out_ |= acc < lim[7 - rem / 4]; if ((g & 3) == 3) LOADS_HERE(); }
        int32_t h = out_ ? -0x40000000 : (int32_t)acc;
        int64_t p = RS((int64_t)h * m, 16);
        int32_t v = (int32_t)RS((int64_t)((uint64_t)p * (uint64_t)p), shift);      /* (relu_sq's p * p, also where it is not used) */
        out[r] = h; aval[k] = v; aidx[k] = (uint16_t)r; k += (p > 0) & (v > tau);
    }
    pix_nact += k;
    return k;
}
static int scan_exit_relu(int32_t *out, const uint8_t *s, const int32_t *tb, int ng, int nrows, const int32_t *lim, int32_t m, int shift, int32_t tau) {
    const uint16_t *o = t3off(s, ng, nrows);
    if (ng == 48) return scan_exit_relu_n(out, o, tb, 48, nrows, lim, m, shift, tau);
    return scan_exit_relu_n(out, o, tb, 32, nrows, lim, m, shift, tau);      /* (the caller checks: D is 128 or 192) */
}
#endif
/* two streams: tb = the tables of the packed inputs with stream 0 + stream 1 in the high half; out = packed rows as
 * the plain scan leaves them (stream 0 high, stream 1 low), dropped rows below zero in both halves */
static void scan_exit2(int32_t *out, const uint8_t *s, const int32_t *tb, int ng, int nrows, const int32_t *lim, int32_t in0) {
    int32_t pos[CHG];
    for (int r = 0; r < nrows; r++, s += ROWB(ng)) {
        uint32_t acc = 0;
        int g;
        t3pos(s, ng, pos);
#ifdef __wasm__
        { uint16_t o[CHG]; t3row(s, ng, o); for (g = 0; g < ng; g++) pos[g] = o[g] / 4; }     /* (the browser build's table layout) */
#endif
        for (g = 0; g < ng; g++) {
            acc += (uint32_t)tb[pos[g]];
            int rem = ng - 1 - g;
            if (rem && rem <= 28 && !(rem & 3) && (int32_t)(acc + ((uint32_t)-lim[7 - rem / 4] << 16) - (uint32_t)in0) < 0) break;
        }
        pix_w1x_look[2] += g < ng ? g + 1 : ng; pix_w1x_look[3] += ng;
        if (g < ng) { out[r] = -1; continue; }
        int32_t lo = (int16_t)acc, hs = (int32_t)(acc - (uint32_t)lo) >> 16, hi = (int16_t)(hs - lo);
        out[r] = (int32_t)(((uint32_t)hi << 16) + (uint32_t)lo);
    }
}
/* two streams, decided by stream 0 alone (PIX_W1XS0): tb = the plain packed tables (stream 0 high, stream 1 low), lim =
 * the one-stream limits in the packed scale; a row is dropped for both streams when the word with -lim << 16 - in0 added
 * is negative at a check point (what the GBA kernel's table biases make of it) */
#ifdef __wasm__
/* The browser build: scan_exit3 and relu_sq2 of its rows in one pass, over the rows' offsets (t3off) and unrolled for
 * 32 or 48 groups, without branches per row (as scan_exit_relu); the limits with in0 are worked out once per call.
 * in (= out, in place), dif, idx, sums and the count are relu_sq2's of scan_exit3's rows (dropped rows included; the
 * twin's study hooks pix_g3 and pix_lanemax, which nothing sets or reads here, aside). */
static inline __attribute__((always_inline)) int scan_exit3_relu_n(int32_t *out, const uint16_t *o, const int32_t *tb, const int ng, int nrows, const int32_t *lim, int32_t in0,
                                                                   int32_t *dif, uint16_t *idx, int32_t m, int shift, int32_t rt, uint32_t *sums) {
    uint32_t th[W1X_NCK];
    uint64_t sv = 0, sd = 0;
    int k = 0, nk = 0;
    for (int k_ = 0; k_ < W1X_NCK; k_++) th[k_] = ((uint32_t)lim[k_] << 16) + (uint32_t)in0;
    for (int r = 0; r < nrows; r++) {
        const uint16_t *q = OROW(o, r, ng); uint32_t acc = 0; int out_ = 0;
        _Pragma("clang loop unroll(full)")
        for (int g = 0; g < ng; g++) {
            acc += (uint32_t)TE(tb, q[g]);
            int rem = ng - 1 - g;
            if (rem && rem <= 28 && !(rem & 3)) out_ |= (int32_t)(acc - th[7 - rem / 4]) < 0;
            if ((g & 3) == 3) LOADS_HERE();
        }
        int32_t h = out_ ? -1 : (int32_t)acc, lo = (int16_t)h, hi = (h - lo) >> 16, v[2];
        for (int s = 0; s < 2; s++) {
            int64_t p = RS((int64_t)(s ? lo : hi) * m, 16);
            int32_t w = (int32_t)RS((int64_t)((uint64_t)p * (uint64_t)p), shift);      /* (relu_sq2's p * p, also where it is not used) */
            int keep = (p > 0) & (w > rt);
            v[s] = keep ? w : 0; nk += keep;
        }
        out[k] = v[0]; dif[k] = v[1] - v[0]; idx[k] = (uint16_t)r; k += (v[0] != 0) | (v[1] != 0);
        sv += (uint32_t)v[0]; sd += (uint32_t)(v[1] < v[0] ? v[0] - v[1] : v[1] - v[0]);
    }
    pix_nact += nk;
    sums[0] = sv > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)sv; sums[1] = sd > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)sd;
    return k;
}
static int scan_exit3_relu(int32_t *out, const uint8_t *s, const int32_t *tb, int ng, int nrows, const int32_t *lim, int32_t in0, int32_t m, int shift, int32_t rt, uint32_t *sums) {
    const uint16_t *o = t3off(s, ng, nrows);
    if (ng == 48) return scan_exit3_relu_n(out, o, tb, 48, nrows, lim, in0, aval, aidx, m, shift, rt, sums);
    return scan_exit3_relu_n(out, o, tb, 32, nrows, lim, in0, aval, aidx, m, shift, rt, sums);      /* (the caller checks: D is 128) */
}
#else
static void scan_exit3(int32_t *out, const uint8_t *s, const int32_t *tb, int ng, int nrows, const int32_t *lim, int32_t in0) {
    int32_t pos[CHG];
    for (int r = 0; r < nrows; r++, s += ROWB(ng)) {
        uint32_t acc = 0;
        int g;
        t3pos(s, ng, pos);
        for (g = 0; g < ng; g++) {
            acc += (uint32_t)tb[pos[g]];
            int rem = ng - 1 - g;
            if (rem && rem <= 28 && !(rem & 3) && (int32_t)(acc - ((uint32_t)lim[7 - rem / 4] << 16) - (uint32_t)in0) < 0) break;
        }
        pix_w1x_look[2] += g < ng ? g + 1 : ng; pix_w1x_look[3] += ng;
        out[r] = g < ng ? -1 : (int32_t)acc;
    }
}
#endif
#endif

/* out[r] = sum of +-in over a ternary matrix stored in chunks of 192 inputs; returns the address after the matrix */
ROM_CODE static const uint8_t *matsum(int32_t *out, const uint8_t *s, const int32_t *in, int nin, int nrows) {
    for (int a = 0; a < nin; a += CHG * 4) {
        int n = nin - a < CHG * 4 ? nin - a : CHG * 4;
        tern_build(tbl, in + a, n / 4);
        if (!a) scan(out, s, tbl, n / 4, nrows);
        else { scan(tmp, s, tbl, n / 4, nrows); for (int r = 0; r < nrows; r++) out[r] += tmp[r]; }
        s += nrows * ROWB(n / 4);
    }
    return s;
}
/* exact 64-bit divisions done with 32-bit ones (the general 64-bit routine is very slow on this CPU) */
#define D3(x) ((uint32_t)(((uint64_t)(x) * 0xAAAAAAABu) >> 33))
ARM_GLUE static uint64_t div_small(uint64_t v, uint32_t n) {           /* v / n for n = 2^a * k, k below 256 */
    uint32_t hi = (uint32_t)(v >> 32), lo = (uint32_t)v;
    int a = 0;
    if (!(n & 15)) { n >>= 4; a = 4; }
    if (!(n & 3)) { n >>= 2; a += 2; }
    while (!(n & 1)) { n >>= 1; a++; }
    if (a) { lo = (lo >> a) | (hi << (32 - a)); hi >>= a; }  /* one shift for the power of two (a is below 32) */
    if (n == 1) return ((uint64_t)hi << 32) | lo;
    if (n == 3) {                                             /* x / 3 = (x * 0xAAAAAAAB) >> 33, exact for 32 bits: no division */
        uint32_t q1 = D3(hi), m = ((hi - q1 * 3) << 16) | (lo >> 16), q2 = D3(m), w = ((m - q2 * 3) << 16) | (lo & 0xFFFF);
        return ((uint64_t)q1 << 32) + ((uint64_t)q2 << 16) + D3(w);
    }
    uint32_t q1 = hi / n, m = ((hi - q1 * n) << 16) | (lo >> 16), q2 = m / n, w = ((m - q2 * n) << 16) | (lo & 0xFFFF);
    return ((uint64_t)q1 << 32) + ((uint64_t)q2 << 16) + w / n;
}
#ifdef __arm__
extern const uint32_t pix_rcp[32768];                     /* kernels.s (ROM): floor(2^55 / (h * 256 + 128)), h = 32768 .. 65535 */
#elif !defined(__wasm__)
static uint32_t pix_rcp[32768];
#endif
/* floor(A / r) without a division, for A = ahi:alo at most 2^40, r in [2^6, 2^24) and a quotient below 2^31: r is
 * normalised to rn in [2^23, 2^24), pix_rcp gives 2^55 / rn to 2^-16 from its top 16 bits, the quotient comes from
 * A >> 9, then one correction from the remainder and a final +-1. udivr in kernels.s on the GBA; this C copy (the twin)
 * does the same steps. Tested against the division for every r in [64, 2^24) with four A each (among them 2^40 and the
 * largest A with a quotient below 2^31) and for 10^8 random (A, r): never more than one +-1 step. */
#ifdef __arm__
uint32_t udivr(uint32_t alo, uint32_t ahi, uint32_t r);
#elif defined(__wasm__)
/* (browser) the same floor(A / r) by one float division: A and r are exact doubles and the quotient is rounded to
 * nearest, so it is floor(A / r) or, just below an integer, one more (then one step back). No table: pix_rcp and its
 * 128 KB leave the saved state. */
static uint32_t udivr(uint32_t alo, uint32_t ahi, uint32_t r) {
    uint64_t a = ((uint64_t)ahi << 32) | alo, q = (uint64_t)((double)a / (double)r);
    if (q * r > a) q--;
    return (uint32_t)q;
}
#else
static uint32_t udivr(uint32_t alo, uint32_t ahi, uint32_t r) {
    int k = 0; uint32_t rn = r;
    if (rn < 1u << 8) { rn <<= 16; k = 16; }
    if (rn < 1u << 16) { rn <<= 8; k += 8; }
    if (rn < 1u << 20) { rn <<= 4; k += 4; }
    if (rn < 1u << 22) { rn <<= 2; k += 2; }
    if (rn < 1u << 23) { rn <<= 1; k += 1; }
    if (!pix_rcp[0]) for (uint32_t h = 0; h < 32768; h++) pix_rcp[h] = (uint32_t)(((uint64_t)1 << 55) / ((uint64_t)(32768 + h) * 256 + 128));
    uint32_t y = pix_rcp[(rn >> 8) - 32768];
    uint32_t q = (uint32_t)(((uint64_t)((alo >> 9) | (ahi << 23)) * y) >> (46 - k));
    uint64_t a = ((uint64_t)ahi << 32) | alo;
    int32_t rm = (int32_t)((int64_t)(a - (uint64_t)q * r) >> 1);
    q += (uint32_t)(int32_t)(((int64_t)rm * (int32_t)(y >> 1)) >> (53 - k));
    int64_t rem = (int64_t)(a - (uint64_t)q * r);
    if (rem < 0) q--; else if (rem >= (int64_t)r) q++;
    return q;
}
#endif
static uint64_t div_shl(uint32_t c, int sh, uint32_t r) {     /* (c << sh) / r for c << (sh - 8) below 2^32 and r below 2^24 */
#ifndef PIX_DIVSHL_OLD
    if (r < (1u << 24)) {
        if (sh == 40 && c == 1 && r > 512) return udivr(0, 256, r);                        /* below 2^31 */
        if (sh == 20 && r >= 512) return udivr(c << 20, c >> 12, r);                      /* below 2^31 (c below 2^20) */
        if (sh == 20 && r >= 64) {                                                        /* two steps of 32 and 8 bits */
            uint32_t a = c << 12, q = udivr(a, 0, r);
            return ((uint64_t)q << 8) + udivr((a - q * r) << 8, 0, r);
        }
    }
#endif
    if (r >> 24) return ((uint64_t)c << sh) / r;
    uint32_t q, rem;
    if (sh == 40 && c == 1) { q = 0xFFFFFFFFu / r; rem = 0xFFFFFFFFu - q * r + 1; if (rem == r) { q++; rem = 0; } }
    else { uint32_t a = c << (sh - 8); q = a / r; rem = a - q * r; }
    return ((uint64_t)q << 8) + (rem << 8) / r;
}
#ifdef __arm__
#define GMAX 37000
static int norm_fast VRAM2_DATA;                             /* this model's norm gains are all at most GMAX in size: norm_k's products fit */
#endif
static void norm(int32_t *out, const int32_t *in, const int32_t *g, int n) {
    uint32_t r = isqrt64(div_small(sumsq_k(in, n), (uint32_t)n) + 168);
    if (r < 16) r = 16;
    int64_t inv = (int64_t)div_shl(1, 40, r);
#ifdef __arm__
    /* r above 2^13: inv is below 2^27, so rs(in * inv, 28) is the high word of in * (inv << 4) + 2^31. |in| is below
     * sqrt(n) (r + 1), so t = rs(in * inv, 28) stays below 4096 sqrt(192) (1 + 2^-13) + 1 < 56,800 in size, and with
     * gains up to GMAX (checked per model in pix_begin) t * g + 2^11 fits 32 bits. Nearly every norm is like this
     * (all but 12 of 78,061 on the test pictures); the rest take the loop below. */
    if (norm_fast && r > 8192 && n <= 192) { norm_k(out, in, g, (int32_t)inv << 4, n); return; }
#endif
#ifdef __wasm__
    /* (browser) r above 512: inv is below 2^31, and with every |in| below 2^26 (so the sum of squares did not wrap)
     * |in| is below sqrt(n) (r + 1) and t = rs(in * inv, 28) below 4096 sqrt(n) (1 + 1 / r) + 1: 32 bits for both
     * products, four lanes at a time */
    if (r > 512 && !(n & 3)) {
        v128_t big = wasm_i32x4_splat(0);
        for (int i = 0; i < n; i += 4) big = wasm_v128_or(big, wasm_i32x4_abs(wasm_v128_load(in + i)));
        if (!wasm_v128_any_true(wasm_v128_and(big, wasm_i32x4_splat(~((1 << 26) - 1))))) {
            v128_t iv = wasm_i32x4_splat((int32_t)inv);
            for (int i = 0; i < n; i += 4) wasm_v128_store(out + i, mulrs4(mulrs4(wasm_v128_load(in + i), iv, 28), wasm_v128_load(g + i), QB));
            return;
        }
    }
#endif
    for (int i = 0; i < n; i++) {
        int64_t t = RS((int64_t)in[i] * inv, 28);
        out[i] = (int32_t)RS(t * g[i], QB);
    }
}
/* rotation by position: cosines and sines of this cell's angles, worked out once per cell (rope_prep) */
static int32_t rcs[PIX_MAX_DK / PIX_MAX_NH + 16] IWRAM_DATA;      /* cosine, sine, cosine, sine, ... */
static void rope_prep(uint32_t pos) {
    const uint32_t *inc = (const uint32_t *)B32(M[MF_ROPE_INC]);
    const int16_t *sn = B16(O_SIN);
    for (int i = 0; i < HK / 2; i++) {
        uint32_t idx = ((pos * inc[i] + (1u << 19)) >> 20) & 4095;
        rcs[2 * i + 1] = sn[idx]; rcs[2 * i] = sn[(idx + 1024) & 4095];
    }
}
#ifdef __arm__
void rope_k(int32_t *v, const int32_t *rot, int nh, int npairs, uint64_t *ss);
#define rope(v, ss) rope_k(v, rcs, NH, HK / 2, ss)
#else
/* DK values = NH heads of HK, rotated in pairs; ss[h] = the sum of squares of head h's rotated values (the query's
 * norm wants it: the kernel adds the squares up while it has the values in registers) */
static void rope(int32_t *v, uint64_t *ss) {
    int32_t *v0 = v;
#ifdef __wasm__
    /* (browser) two pairs at a time: lanes (a, b, a', b') times (co, co, co', co') plus (b, a, b', a') times
     * (-si, si, -si', si'), in 64 bits */
    if (!(HK & 3)) {
        for (int h = 0; h < NH; h++)
            for (int i = 0; i < HK / 2; i += 2, v += 4) {
                v128_t x = wasm_v128_load(v), r = wasm_v128_load(rcs + 2 * i), xs = wasm_i32x4_shuffle(x, x, 1, 0, 3, 2);
                v128_t co = wasm_i32x4_shuffle(r, r, 0, 0, 2, 2), si = wasm_i32x4_mul(wasm_i32x4_shuffle(r, r, 1, 1, 3, 3), wasm_i32x4_make(-1, 1, -1, 1));
                v128_t rd = wasm_i64x2_splat(1 << 13);
                v128_t lo = wasm_i64x2_add(wasm_i64x2_add(wasm_i64x2_extmul_low_i32x4(x, co), wasm_i64x2_extmul_low_i32x4(xs, si)), rd);
                v128_t hi = wasm_i64x2_add(wasm_i64x2_add(wasm_i64x2_extmul_high_i32x4(x, co), wasm_i64x2_extmul_high_i32x4(xs, si)), rd);
                wasm_v128_store(v, wasm_i32x4_shuffle(wasm_u64x2_shr(lo, 14), wasm_u64x2_shr(hi, 14), 0, 2, 4, 6));
            }
        for (int h = 0; h < NH; h++) ss[h] = sumsq_k(v0 + h * HK, HK);
        return;
    }
#endif
    for (int h = 0; h < NH; h++)
        for (int i = 0; i < HK / 2; i++, v += 2) {
            int32_t si = rcs[2 * i + 1], co = rcs[2 * i];
            int64_t a = v[0], b = v[1];
            v[0] = (int32_t)RS(a * co - b * si, 14);
            v[1] = (int32_t)RS(a * si + b * co, 14);
        }
    for (int h = 0; h < NH; h++) ss[h] = sumsq_k(v0 + h * HK, HK);
}
#endif
/* acc += row of an int8 table times its Q16 step */
static inline void addrow(int32_t *acc, uint32_t tab, uint32_t steps, int row, int n) {
    int32_t s = B32(steps)[row];
    if (s) axpy8(acc, B8(tab) + (uint32_t)row * (uint32_t)n, s, n);
}

/* ---- one or two streams at once ----
 * With prompt guidance the body runs twice per cell (prompt, empty prompt). The table kernels are linear, so both
 * inputs go through them together: each table entry holds stream 0's sum in its high half and stream 1's in its low
 * half (inputs first lose XSH bits so the sums fit 16 bits). One build and one scan serve both streams. */
#define XSH 6
#ifndef __arm__
extern int pix_lanemax;                                      /* largest half-sum seen when two streams share a table (must stay below 32768) */
#endif
static int32_t x1[2 * PIX_MAX_D] VRAM2_DATA, proj1[256] VRAM0_DATA, mixv1[PIX_MAX_DV] VRAM0_DATA;
#define xn1 (x1 + PIX_MAX_D)
typedef char pix_v1_check[(2 * PIX_MAX_D >= PIX_MAX_HH && 256 >= PIX_MAX_HH) ? 1 : -1];        /* x1 + xn1 and proj1 can each hold a row of HH words */
#ifdef __arm__
/* Stream 1's buffers are placed per picture (pix_begin): a model with one pixel per cell, no thumbnail and tables of
 * at most PIX_MAX_DH inputs leaves ycell and tsum (see the tail of tbl above) idle, and stream 1 runs in that fast
 * RAM instead of video memory. PIX_S1_FAST 0 keeps it in video memory. */
#ifndef PIX_S1_FAST
#define PIX_S1_FAST 1
#endif
static int32_t *XS[2] VRAM2_DATA, *XNS[2] VRAM2_DATA, *PJ[2] VRAM2_DATA, *MV[2] VRAM2_DATA;
typedef char pix_s1_check[(2 * PIX_MAX_DH + 256 + PIX_MAX_DV + PIX_MAX_DH <= PIX_MAX_HH + 4 * PIX_MAX_DH) ? 1 : -1];
#else
static int32_t *const XS[2] = { x, x1 }, *const XNS[2] = { xn, xn1 }, *const PJ[2] = { proj, proj1 };
static int32_t *const MV[2] = { mixv, mixv1 };
#endif
#ifdef __arm__
void pack2(int32_t *out, const int32_t *a, const int32_t *b, int n);
void unpack2(int32_t *a, int32_t *b, const int32_t *in, int n);
#else
static void pack2(int32_t *out, const int32_t *a, const int32_t *b, int n) {
    for (int i = 0; i < n; i++) out[i] = (int32_t)((uint32_t)((a[i] + (1 << (XSH - 1))) >> XSH) << 16) + ((b[i] + (1 << (XSH - 1))) >> XSH);
}
static void unpack2(int32_t *a, int32_t *b, const int32_t *in, int n) {
#ifdef __wasm__
    for (int i = 0; i < n; i++) { int32_t lo = (int16_t)in[i]; b[i] = lo; a[i] = (in[i] - lo) >> 16; }      /* (browser: no pix_lanemax) */
    return;
#endif
    for (int i = 0; i < n; i++) { int32_t lo = (int16_t)in[i]; b[i] = lo; a[i] = (in[i] - lo) >> 16;
        int m = lo < 0 ? -lo : lo, h = a[i] < 0 ? -a[i] : a[i]; if (m > pix_lanemax) pix_lanemax = m; if (h > pix_lanemax) pix_lanemax = h; }
}
#endif
/* The same for two streams packed in one word (see tb: stream 0 in the high half, stream 1 in the low half), for
 * the shared second matrix (mlp_out2). For every input kept in either stream: in[k] = v0 (in place: k never passes
 * the input being read), dif[k] = v1 - v0, idx[k] = input number, a stream that does not keep it counting as 0.
 * sums[0] = sum of v0, sums[1] = sum of |v1 - v0| (both stop at 2^32 - 1). Returns how many. */
#ifndef __arm__
int pix_g3;                                                   /* (study hook, env G3) */
int pix_lanemax;
long long pix_shared[2];                                     /* calls done in one pass over the tables / split in two */
#endif
/* (rt: tau here, the smallest kept input on the GBA, where tern_scan_relu does this for the models so far) */
#ifdef __wasm__
/* (browser) the same lists and sums; a word is looked at only when one of its halves reaches in0 (see relu_sq): their
 * numbers go to dif first. (k never passes the candidate c being read, nor c the word.) */
static int relu_sq2(int32_t *in, int32_t *dif, uint16_t *idx, int n, int32_t m, int shift, int32_t rt, uint32_t *sums) {
    uint64_t sv = 0, sd = 0;
    int k = 0, i = 0;
#define RSQ2(i) { int32_t lo = (int16_t)in[i], hi = (in[i] - lo) >> 16, v[2] = { 0, 0 }; \
        for (int s = 0; s < 2; s++) { int64_t p = RS((int64_t)(s ? lo : hi) * m, 16); \
            if (p > 0) { int32_t w = (int32_t)RS(p * p, shift); if (w > rt) v[s] = w; } } \
        if (v[0] || v[1]) { in[k] = v[0]; dif[k] = v[1] - v[0]; idx[k++] = (uint16_t)(i); \
            sv += (uint32_t)v[0]; sd += (uint32_t)(v[1] < v[0] ? v[0] - v[1] : v[1] - v[0]); } }
    int nc = pass_list(dif, in, n, relu_in0t(shift, rt, m), 1);
    for (int c = 0; c < nc; c++) { i = dif[c]; RSQ2(i); }
#undef RSQ2
    sums[0] = sv > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)sv; sums[1] = sd > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)sd;
    return k;
}
#else
__attribute__((noinline)) ROM_CODE static int relu_sq2(int32_t *in, int32_t *dif, uint16_t *idx, int n, int32_t m, int shift, int32_t rt, uint32_t *sums) {
    uint64_t sv = 0, sd = 0;
    int k = 0;
    for (int i = 0; i < n; i++) {
        int32_t lo = (int16_t)in[i], hi = (in[i] - lo) >> 16, v[2] = { 0, 0 };
#ifndef __arm__
        int ml = lo < 0 ? -lo : lo, mh = hi < 0 ? -hi : hi; if (ml > pix_lanemax) pix_lanemax = ml; if (mh > pix_lanemax) pix_lanemax = mh;
#endif
        for (int s = 0; s < 2; s++) {
#ifdef __arm__
            if ((s ? lo : hi) < rt) continue;
            int64_t p = RS((int64_t)(s ? lo : hi) * m, 16);
            v[s] = (int32_t)RS(p * p, shift);
#else
            int64_t p = RS((int64_t)(s ? lo : hi) * m, 16);
            if (p <= 0) continue;
            int32_t w = (int32_t)RS(p * p, shift);
            if (w > rt) { v[s] = w; pix_nact++; }
#endif
        }
#ifndef __arm__
        if (pix_g3 && !v[0]) v[1] = 0;                      /* (study G3=1: stream 1 keeps only the units stream 0 keeps; worse) */
#endif
        if (!v[0] && !v[1]) continue;
        in[k] = v[0]; dif[k] = v[1] - v[0]; idx[k++] = (uint16_t)i;
        sv += (uint32_t)v[0]; sd += (uint32_t)(v[1] < v[0] ? v[0] - v[1] : v[1] - v[0]);
    }
    sums[0] = sv > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)sv; sums[1] = sd > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)sd;
    return k;
}
#endif
/* The second MLP matrix for both streams in one pass over the tables. Most hidden units that are active are active
 * in both streams with similar values, so each table entry holds stream 0's sum above bit k and the sum of
 * (stream 1 - stream 0) below it. k comes from the values: sums[1] bounds every |difference sum| and sums[0] every
 * |stream 0 sum|, so when both fit one word the result is exact; when they do not, the same units go through the
 * tables once per stream. na, sums: from tern_scan_relu, or na < 0 and hid = the packed first-matrix sums (see tb). */
__attribute__((noinline)) ROM_CODE static void mlp_out2(const uint8_t *cols, int hdl, int32_t m1, int shift, int32_t m2, int na, uint32_t *sums, const int32_t *base, int l) {
    int k = 1;
    uint32_t sm[2];
    if (na < 0) {                                            /* (not done by the row kernel) */
        int32_t rt = RT(p0_l[l], tau_l[l], m1);
#ifdef __arm__
        if (rt > 32767) return;                              /* no half of a packed sum reaches it: nothing to add */
#endif
        na = relu_sq2(hid, aval, aidx, hdl, m1, shift, rt, sums = sm);
    }
    if (!na) return;
    while (k < 31 && sums[1] >> (k - 1)) k++;
    if (k < 31 && !(sums[0] >> (31 - k))) {
#ifndef __arm__
        pix_shared[0]++;
#endif
        packk(hid, aval, na, k);
        sparse_out(acc, cols, hid, na, D, 0, base);
        scale_add2(XS[0], XS[1], acc, m2, D, k);
        return;
    }
#ifndef __arm__
    pix_shared[1]++;
#endif
    if (!m2) return;
    sparse_out(XS[0], cols, hid, na, D, m2, base);
    packk(hid, aval, na, 0);                                 /* stream 1's values */
    sparse_out(XS[1], cols, hid, na, D, m2, 0);
}
/* build the tables for n inputs of one stream (b = 0) or two; returns how many bits the row sums are short of Q12 */
static int tb(const int32_t *a, const int32_t *b, int n) {
    if (!b) { tern_build(tbl, a, n / 4); return 0; }
    int32_t *pk = n <= PIX_MAX_NC ? sc : tmp;                 /* sc is free outside att_mix and is fast memory */
    pack2(pk, a, b, n); tern_build(tbl, pk, n / 4);
    return XSH;
}
__attribute__((noinline)) static void rows(int32_t *o0, int32_t *o1, const uint8_t *steps, int nbytes, int nrows) {
    if (!o1) { scan(o0, steps, tbl, nbytes, nrows); return; }
    scan(o0, steps, tbl, nbytes, nrows); unpack2(o0, o1, o0, nrows);     /* unpacked in place: o0 is the faster memory */
}

/* Key rows that are all zero, or whose gain is zero, give a zero key value whatever the input: pix_begin lists the
 * runs of rows that are worth scanning (per attention layer), the rest are set to zero. */
#define KSEGS 16
static uint16_t kseg[PIX_MAX_L][2 * KSEGS] EWRAM_DATA;
static uint8_t ksegn[PIX_MAX_L] EWRAM_DATA;
static void krows(int32_t *o0, int32_t *o1, const uint8_t *steps, int ai) {
    const uint16_t *sg = kseg[ai];
    int nb = D / 4, r = 0;
    for (int i = 0; i < ksegn[ai]; i++, sg += 2) {
        while (r < sg[0]) o0[r++] = 0;
        scan(o0 + r, steps + (uint32_t)r * ROWB(nb), tbl, nb, sg[1]);
        r += sg[1];
    }
    while (r < DK) o0[r++] = 0;
    if (o1) unpack2(o0, o1, o0, DK);
}
ROM_CODE static void kseg_prep(void) {
    int ai = 0, nb = D / 4;
    for (int l = 0; l < L; l++) {
        if (!(M[MF_ATTMASK] >> l & 1)) continue;
        const uint8_t *k = BU8(LAY(l, LF_K));
        const int32_t *kg = B32(LAY(l, LF_KG));
        int n = 0, open = 0;
        for (int r = 0; r < DK && n >= 0; r++) {
            int live = kg[r] != 0;
            if (live) {                                      /* a zero row takes entry 0 of every group: its words are known */
                const uint32_t *w = (const uint32_t *)(k + (uint32_t)r * ROWB(nb));
                int nw = T3W(nb), f = nb - 3 * (nw - 1), j = 1;
                live = w[0] != (f == 3 ? 82u << 24 | 1u << 11 | 81u : f == 2 ? 82u << 24 | 1u << 11 : 1u << 24);
                while (!live && j < nw) live = w[j++] != (162u << 24 | 81u << 11 | 81u);
            }
            if (live && !open) { if (n == KSEGS) n = -1; else { kseg[ai][2 * n] = (uint16_t)r; kseg[ai][2 * n + 1] = 0; n++; } }
            if (live && n > 0) kseg[ai][2 * n - 1]++;
            open = live;
        }
        if (n < 0) { n = 1; kseg[ai][0] = 0; kseg[ai][1] = (uint16_t)DK; }      /* too many runs: scan everything */
        ksegn[ai++] = (uint8_t)n;
    }
}
/* attention, in three per-stream parts around the shared projections */
/* the general form (keys with zeros); the models so far have sign-only keys, which key_k does on the GBA */
__attribute__((noinline)) ROM_CODE static void att_key_c(Stream *sm, int l, int ai, int t, int32_t *pj) {
    uint8_t *keys = sm->keys + (uint32_t)ai * NH * NC * KB;
    const int32_t *kg = B32(LAY(l, LF_KG));
#ifdef __wasm__
    mulrs_k(pj, pj, kg, DK, QB);
#else
    for (int i = 0; i < DK; i++) pj[i] = (int32_t)RS((int64_t)pj[i] * kg[i], QB);
#endif
    uint64_t ssq[PIX_MAX_NH];
    rope(pj, ssq);
    for (int h = 0; h < NH; h++) {
        uint8_t *kk = keys + ((uint32_t)h * NC + t) * KB;
        int prev = 0;
        for (int g = 0; g < KB; g++) {
            int idx = 0;
            if (g * 4 < HK) {
                const int32_t *p = pj + h * HK + g * 4;
                idx = (p[0] >= 0 ? 1 : 2) + 3 * (p[1] >= 0 ? 1 : 2) + 9 * (p[2] >= 0 ? 1 : 2) + 27 * (p[3] >= 0 ? 1 : 2);
            }
#ifdef __wasm__
            kk[g] = (uint8_t)idx; continue;                  /* (browser: entry numbers, see scan8) */
#endif
            kk[g] = (uint8_t)(g ? NE + idx - prev : idx + 1);
            prev = idx;
        }
    }
}
static void att_key(Stream *sm, int l, int ai, int t, int32_t *pj) {
#ifdef __arm__
    if (KB * 4 == HK) {                                      /* sign-only keys: gain, rotation and packing in one pass */
        uint8_t *keys = sm->keys + (uint32_t)ai * NH * NC * KB;
        const int32_t *kg = B32(LAY(l, LF_KG));
        for (int h = 0; h < NH; h++) key_k(pj + h * HK, kg + h * HK, rcs, keys + ((uint32_t)h * NC + t) * KB, KB);
        return;
    }
#endif
    att_key_c(sm, l, ai, t, pj);
}
#ifdef __arm__
int att_val_k(int8_t *vt, const int32_t *pj, int32_t m, int shift, int n);
#endif
static void att_val(Stream *sm, int l, int ai, int t, int32_t *pj, int sh) {
    const int32_t *att = B32(LAY(l, LF_ATT));
    int8_t *vt = sm->vals + (uint32_t)ai * NC * DV + (uint32_t)t * DV;
#ifdef __arm__
    /* one kernel: the scaling is monotonic, so the largest magnitude comes from the largest or the smallest input,
     * and each value is then scaled, shifted and stored in one pass (pj is left as it was) */
    sm->vsh[ai * NC + t] = (uint8_t)att_val_k(vt, pj, att[0], 16 - sh, DV);
#else
    int32_t mx = 0;
    scale_rs(pj, pj, att[0], DV, 16 - sh);
    for (int i = 0; i < DV; i++) {
        int32_t a = pj[i] < 0 ? -pj[i] : pj[i];
        if (a > mx) mx = a;
    }
    int s2 = 0;
    while ((mx >> s2) > 127) s2++;
#ifdef __wasm__
    for (int i = 0; i < DV; i++) {                           /* (browser: the same rounding in 32 bits) */
        int32_t m = s2 ? RS32(pj[i], s2) : pj[i];
        vt[i] = (int8_t)(m > 127 ? 127 : m < -127 ? -127 : m);
    }
#else
    for (int i = 0; i < DV; i++) {
        int32_t m = s2 ? (int32_t)RS((int64_t)pj[i], s2) : pj[i];
        vt[i] = (int8_t)(m > 127 ? 127 : m < -127 ? -127 : m);
    }
#endif
    sm->vsh[ai * NC + t] = (uint8_t)s2;
#endif
}
/* the general query table: scaled values through memory (keys that are not sign-only, or a very large scale) */
__attribute__((noinline)) ROM_CODE static void qbuild(const int32_t *pq, int64_t m) {
#ifdef __wasm__
    /* (browser) sign-only keys read 16 of a group's 81 entries (digits 1 and 2, att_key_c): only those are built */
    if (KB * 4 == HK) {
        static const uint8_t lo[4] = { 4, 5, 7, 8 }, hi[4] = { 36, 45, 63, 72 };
        for (int g = 0; g < KB; g++, pq += 4) {
            int32_t v[4], *t = tbl + g * NE;
            for (int i = 0; i < 4; i++) v[i] = (int32_t)RS((int64_t)pq[i] * m, 24);
            int32_t a[4] = { v[0] + v[1], -v[0] + v[1], v[0] - v[1], -v[0] - v[1] }, b[4] = { v[2] + v[3], -v[2] + v[3], v[2] - v[3], -v[2] - v[3] };
            for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) t[hi[y] + lo[x]] = a[x] + b[y];
        }
        return;
    }
#endif
    int32_t *q = proj + PIX_MAX_DK;                          /* this head's query, padded with zeros to whole words of steps (spare end of proj) */
    for (int i = 0; i < KB * 4; i++) q[i] = i < HK ? (int32_t)RS((int64_t)pq[i] * m, 24) : 0;
#ifdef __wasm__
    tern_build_g(tbl, q, KB); return;                        /* (by group: scan8 reads entry g * NE + key byte) */
#endif
    tern_build(tbl, q, KB);
}
/* one head's key scan and top 8 the plain way: scores to sc, then topsel */
__attribute__((noinline)) ROM_CODE static int keysel(const uint8_t *kh, const int32_t *bias, int j0, int n, int32_t *tv) {
    scan8(sc + j0, kh + (uint32_t)j0 * KB, tbl, KB, n - j0);
    return topsel(sc, bias, j0, n, tv, pix_nskip ? skipped : 0);
}
#ifdef __arm__
/* One head's key scan, distance bias and top 8: kscan(end of the head's keys of cells 0..n-1, bias row, rows, tv)
 * is keytop, or keysel4 for keys it does not take. What is the same for all heads of a cell is in pix_ktp, set
 * once per attention call (and read by the kernel: fewer arguments, less glue in fast RAM).
 * Two-word sign-only keys (the 16x16 models): keytop works out the most a row's second word of steps can add (from
 * the query table that is built) and drops a row after its first word when even that cannot lift it to the 8th best
 * so far. Exact: such a row would not have been inserted. */
struct { const uint8_t *skip; int32_t j0m1, nwords; const int32_t *tblm4; } pix_ktp VRAM2_DATA;
typedef int (*KScan)(const uint8_t *kend, const int32_t *bias, int nrows, int32_t *tv);
static KScan kscan VRAM2_DATA;
ROM_CODE static int keysel4(const uint8_t *kend, const int32_t *bias, int nrows, int32_t *tv) {
    int j0 = pix_ktp.j0m1 + 1, n = j0 + nrows;
    return keysel(kend - (uint32_t)n * KB, bias, j0, n, tv);
}
#endif
__attribute__((noinline)) IWRAM_CODE static void att_mix(Stream *sm, int l, int ai, int t, int32_t *pj, int32_t *mv) {
    const int32_t *att = B32(LAY(l, LF_ATT));
    uint8_t *keys = sm->keys + (uint32_t)ai * NH * NC * KB;
    int8_t *vals = sm->vals + (uint32_t)ai * NC * DV;
    const uint8_t *vsh = sm->vsh + ai * NC;
    uint64_t ssq[PIX_MAX_NH];
    rope(pj, ssq); ATK(6);
    const int32_t *db = B32(LAY(l, LF_DBIAS));
    int n = t + 1, j0 = awin && n > awin ? n - awin : 0;
    for (int h = 0; h < NH; h++) {
#ifndef __arm__
        if (pix_hdrop >> (ai * NH + h) & 1) { for (int i = 0; i < HV; i++) mv[h * HV + i] = 0; continue; }
        j0 = awin && n > awin ? n - awin : 0;
        if (pix_hwin[ai][h] && n > pix_hwin[ai][h]) j0 = n - pix_hwin[ai][h];
#endif
        uint32_t r = isqrt64(div_small(ssq[h], (uint32_t)HK) + 168);
        if (r < 1) r = 1;
        int64_t m = (int64_t)div_shl((uint32_t)att[2 + h], 20, r);
#ifdef __arm__
        /* sign-only keys read 16 of the 81 entries: qbuild16 scales the query on its way into them */
        if (KB * 4 == HK && m < ((int64_t)1 << 31)) qbuild16(tbl, pj + h * HK, KB, (int32_t)m); else
#endif
        qbuild(pj + h * HK, m);
        ATK(7);
        int32_t tv[19];
        const uint8_t *kh = keys + (uint32_t)h * NC * KB;
#ifdef __arm__
        int k = kscan(kh + (uint32_t)n * KB, db + h * NC, n - j0, tv);
#else
        int k = keysel(kh, db + h * NC, j0, n, tv);
#endif
        ATK(8);
        uint32_t w[8], sum = 0;
        for (int i = 0; i < k; i++) {
            uint32_t d = (uint32_t)(tv[2] - tv[2 + 2 * i]) >> (QB - PIX_EXP_BITS);
            w[i] = d < PIX_EXP_N ? BU16(O_EXP)[d] : 0;
            sum += w[i];
        }
#ifndef PIX_DIVSHL_OLD
        uint32_t R = sum >= 64 && sum < (1u << 24) ? udivr(0x80000000u, 0, sum) : 0x80000000u / sum;      /* (the top weight alone is 65,535) */
#else
        uint32_t R = 0x80000000u / sum;
#endif
        /* sum of weight (Q15) x value with exponent s2, kept 8 bits above Q12. Values stored with 7 or more bits
         * dropped are shifted up, which goes into the weight (ea); the others are rounded one by one (eb). Zero
         * weights add nothing. mixk does all the cells in one pass. (The scores are in tv now: the lists use sc.) */
#ifdef __arm__
        MixA *ea = (MixA *)sc, *pa = ea;
        MixB *eb = (MixB *)(sc + 18), *pb = eb;
#else
        MixA ea[9], *pa = ea;
        MixB eb[9], *pb = eb;
#endif
        for (int i = 0; i < k; i++) {
            int32_t p = (int32_t)((w[i] * R) >> 16);         /* w * R is at most 2^31 */
            if (!p) continue;
            int j = tv[3 + 2 * i], s2 = vsh[j];
            const int8_t *vv = vals + (uint32_t)j * DV + h * HV;
            if (s2 >= 7) { pa->v = vv; pa->p = (int32_t)((uint32_t)p << (s2 - 7)); pa++; }
            else { pb->v = vv; pb->p = p; pb->rr = 7 - s2; pb->hf = 1 << (6 - s2); pb++; }
        }
        pa->v = 0; pb->v = 0;
        mixk(mv + h * HV, ea, eb, HV / 4);
    }
}
/* (B) approximation, switchable: PIX_G2B 1 = in guided cells the empty-prompt stream takes the prompt stream's
 * attention picks and weights (its own values): no keys, query or key scan for stream 1. Teacher-forced with guidance
 * (tools/gkl.py) this is closer to the PyTorch model than separate picks (KL 0.033 -> 0.024, see DESIGN.md);
 * 0 = each stream picks its own top 8, as before (the twin: G2B=0). */
#ifndef PIX_G2B
#define PIX_G2B 1
#endif
#ifndef __arm__
int pix_g2b = PIX_G2B;
#else
#define pix_g2b PIX_G2B
#endif
/* att_mix for both streams of a guided cell under PIX_G2B (not in fast RAM: only the guided cells run it) */
__attribute__((noinline)) static void att_mix2(int l, int ai, int t, int32_t *pj) {
    const int32_t *att = B32(LAY(l, LF_ATT));
    uint8_t *keys = st[0].keys + (uint32_t)ai * NH * NC * KB;
    uint64_t ssq[PIX_MAX_NH];
    rope(pj, ssq);
    const int32_t *db = B32(LAY(l, LF_DBIAS));
    int n = t + 1, j0 = awin && n > awin ? n - awin : 0;
    for (int h = 0; h < NH; h++) {
        uint32_t r = isqrt64(div_small(ssq[h], (uint32_t)HK) + 168);
        if (r < 1) r = 1;
        int64_t m = (int64_t)div_shl((uint32_t)att[2 + h], 20, r);
#ifdef __arm__
        if (KB * 4 == HK && m < ((int64_t)1 << 31)) qbuild16(tbl, pj + h * HK, KB, (int32_t)m); else
#endif
        qbuild(pj + h * HK, m);
        int32_t tv[19];
        const uint8_t *kh = keys + (uint32_t)h * NC * KB;
#ifdef __arm__
        int k = kscan(kh + (uint32_t)n * KB, db + h * NC, n - j0, tv);
#else
        int k = keysel(kh, db + h * NC, j0, n, tv);
#endif
        uint32_t w[8], sum = 0;
        for (int i = 0; i < k; i++) {
            uint32_t d = (uint32_t)(tv[2] - tv[2 + 2 * i]) >> (QB - PIX_EXP_BITS);
            w[i] = d < PIX_EXP_N ? BU16(O_EXP)[d] : 0;
            sum += w[i];
        }
        uint32_t R = 0x80000000u / sum;
        for (int s = 0; s < 2; s++) {                        /* the same weights, each stream's own values */
            const int8_t *vals = st[s].vals + (uint32_t)ai * NC * DV;
            const uint8_t *vsh = st[s].vsh + ai * NC;
#ifdef __arm__
            MixA *ea = (MixA *)sc, *pa = ea;
            MixB *eb = (MixB *)(sc + 18), *pb = eb;
#else
            MixA ea[9], *pa = ea;
            MixB eb[9], *pb = eb;
#endif
            for (int i = 0; i < k; i++) {
                int32_t p = (int32_t)((w[i] * R) >> 16);
                if (!p) continue;
                int j = tv[3 + 2 * i], s2 = vsh[j];
                const int8_t *vv = vals + (uint32_t)j * DV + h * HV;
                if (s2 >= 7) { pa->v = vv; pa->p = (int32_t)((uint32_t)p << (s2 - 7)); pa++; }
                else { pb->v = vv; pb->p = p; pb->rr = 7 - s2; pb->hf = 1 << (6 - s2); pb++; }
            }
            pa->v = 0; pb->v = 0;
            mixk(MV[s] + h * HV, ea, eb, HV / 4);
        }
    }
}
static void attention(int ns, int l, int ai, int t) {
    const int32_t *att = B32(LAY(l, LF_ATT));
    int32_t *p1 = ns > 1 ? PJ[1] : 0;
#ifdef PIX_PROFA
    alast = now();
#endif
    int sh = tb(XNS[0], ns > 1 ? XNS[1] : 0, D); ATK(0);
    krows(PJ[0], p1, BU8(LAY(l, LF_K)), ai); ATK(1);
    int g2 = ns > 1 && pix_g2b;                              /* stream 1's keys are never read (PIX_G2B) */
    for (int s = 0; s < (g2 ? 1 : ns); s++) att_key(&st[s], l, ai, t, PJ[s]);
    ATK(2);
    rows(PJ[0], p1, BU8(LAY(l, LF_V)), D / 4, DV); ATK(3);
    for (int s = 0; s < ns; s++) att_val(&st[s], l, ai, t, PJ[s], sh);
    ATK(4);
    rows(PJ[0], p1, BU8(LAY(l, LF_Q)), D / 4, DK); ATK(5);
#ifdef __arm__
    pix_ktp.skip = pix_nskip ? skipped : 0; pix_ktp.j0m1 = (awin && t + 1 > awin ? t + 1 - awin : 0) - 1; pix_ktp.nwords = KB / 4; pix_ktp.tblm4 = (const int32_t *)((uintptr_t)tbl - 4);
#endif
    if (g2) att_mix2(l, ai, t, PJ[0]);
    else for (int s = 0; s < ns; s++) att_mix(&st[s], l, ai, t, PJ[s], MV[s]);
    ATK(9);
    sh = tb(MV[0], ns > 1 ? MV[1] : 0, DV);
    rows(PJ[0], p1, BU8(LAY(l, LF_O)), DV / 4, D);
#ifndef __arm__
    if (pix_adrop >> ai & 1) return;
#endif
    for (int s = 0; s < ns; s++) scale_add(XS[s], PJ[s], att[1] << sh, D);
    ATK(10);
}

/* the part of an attention layer that later cells read: this cell's keys and values (PIX_HEADSKIP), one stream */
__attribute__((noinline)) ROM_CODE static void att_kv(int l, int ai, int t) {
    int sh = tb(XNS[0], 0, D);
    krows(PJ[0], 0, BU8(LAY(l, LF_K)), ai);
    att_key(&st[0], l, ai, t, PJ[0]);
    rows(PJ[0], 0, BU8(LAY(l, LF_V)), D / 4, DV);
    att_val(&st[0], l, ai, t, PJ[0], sh);
}

#ifdef __wasm__
/* (browser) a cell whose pixels are all held: of the last layer only what later cells read, for every stream (its
 * keys and values / mixer state); set by pix_cell around body */
static int kvheld;
static void kv_last(int ns, int l, int ai, int mi, int t) {
    if (!(M[MF_ATTMASK] >> l & 1)) { for (int s = 0; s < ns; s++) mixer_k(XS[s], XNS[s], st[s].mst + mi * D * 6, MIXC(l, mi), D); return; }
    int32_t *p1 = ns > 1 ? PJ[1] : 0;
    int sh = tb(XNS[0], p1 ? XNS[1] : 0, D);              /* (the first half of attention()) */
    krows(PJ[0], p1, BU8(LAY(l, LF_K)), ai);
    int g2 = ns > 1 && pix_g2b;
    for (int s = 0; s < (g2 ? 1 : ns); s++) att_key(&st[s], l, ai, t, PJ[s]);
    rows(PJ[0], p1, BU8(LAY(l, LF_V)), D / 4, DV);
    for (int s = 0; s < ns; s++) att_val(&st[s], l, ai, t, PJ[s], sh);
}
#endif
/* an MLP of routed experts (none of the models so far has one: kept in ROM), one stream at a time */
__attribute__((noinline)) ROM_CODE static void routed(int ns, int l, int t, int pg) {
    int hs = (int)M[MF_HS];
    for (int s = 0; s < ns; s++) {
        int ex = BU8(LAY(l, LF_ROUTE))[t * 16 + (pg >= 0 ? pg : (st[s].first & 15))];
        int32_t m1 = B32(LAY(l, LF_W1M))[ex], m2 = B32(LAY(l, LF_W2M))[ex];
        matsum(hid, BU8(LAY(l, LF_W1)) + (uint32_t)ex * HD * ROWB(D / 4), XNS[s], D, HD);
        for (int i = 0; i < HD; i++) {
            int64_t p = RS((int64_t)hid[i] * m1, 16);
            hid[i] = p > 0 ? (int32_t)RS(p * p, QB + hs) : 0;
        }
        matsum(proj, BU8(LAY(l, LF_W2)) + (uint32_t)ex * D * (HD / (CHG * 4) * ROWB(CHG) + ROWB(HD % (CHG * 4) / 4)), hid, HD, D);
        for (int i = 0; i < D; i++) XS[s][i] += (int32_t)RS((int64_t)proj[i] * m2, 16);
    }
}
/* With several pixels per cell the head's second matrix finds the tables as the pixel before left them (a build of z,
 * which is still in place), or as the body's last matsum did: hbase says of what (0: not known, clear them). */
static const int32_t *hbase VRAM2_DATA;
/* the body for cell t, for the prompt stream alone (ns = 1) or with the empty-prompt stream (ns = 2):
 * leaves each stream's input to the head (Q12, DH wide) in its hin */
__attribute__((always_inline)) static inline void body(int ns, int t) {
    int n = S / C;
    const int16_t *perm = B16(M[MF_PERM]);
    TICK0();
    for (int s = 0; s < ns; s++) {
        int32_t *xs = XS[s];
        if (s) { for (int i = 0; i < D; i++) xs[i] = XS[0][i] - st[0].cw[i] + st[s].cw[i]; continue; }     /* same table rows, other prompt */
        copy32(xs, st[s].cw, D);
        addrow(xs, M[MF_CELLPOS], M[MF_CELLPOS_ST], t, D);
        for (int j = 0; j < NS; j++) {                       /* two rows per pass */
            int rp = j * P1 + (t > 0 ? pix_img[perm[(t - 1) * NS + j]] : 256), ru = j * P1 + (t >= n ? pix_img[perm[(t - n) * NS + j]] : 256);
            axpy8_2(xs, xs, B8(M[MF_PREV]) + (uint32_t)rp * D, B32(M[MF_PREV_ST])[rp], B8(M[MF_UP]) + (uint32_t)ru * D, B32(M[MF_UP_ST])[ru], D);
        }
        if (M[MF_F]) {
            const int16_t *tw = B16(M[MF_TW]) + t * WIN;
            const int8_t *tt = B8(M[MF_TH]); const int32_t *tst = B32(M[MF_TH_ST]);
            int k = 0;
            for (; k + 1 < WIN; k += 2) {
                int r0 = k * P1 + (tw[k] < 0 ? 256 : thumb[tw[k]]), r1 = (k + 1) * P1 + (tw[k + 1] < 0 ? 256 : thumb[tw[k + 1]]);
                axpy8_2(xs, xs, tt + (uint32_t)r0 * D, tst[r0], tt + (uint32_t)r1 * D, tst[r1], D);
            }
            if (k < WIN) addrow(xs, M[MF_TH], M[MF_TH_ST], k * P1 + (tw[k] < 0 ? 256 : thumb[tw[k]]), D);
        }
    }
    for (int s = 0; s < ns; s++) rs4_k(XS[s], D);
    int pg = t > 0 ? pix_img[perm[(t - 1) * NS + NS - 1]] >> 4 : -1;
    int ai = 0, mi = 0;
    rope_prep((uint32_t)t);
    TICK(PP_INPUT);
    for (int l = 0; l < L; l++) {
        for (int s = 0; s < ns; s++) {
            int32_t *xs = XS[s];
            if (l && M[MF_CLSL]) add32(xs, st[s].cw + l * D, D);
            norm(XNS[s], xs, B32(LAY(l, LF_N1)), D);
        }
        TICK(PP_NORM);
#ifdef __wasm__
        if (kvheld && l == L - 1) { kv_last(ns, l, ai, mi, t); return; }
#endif
        if (kvonly && l == L - 1) {                          /* (PIX_HEADSKIP) nothing after this is read by later cells */
            if (M[MF_ATTMASK] >> l & 1) att_kv(l, ai, t);
            else mixer_k(XS[0], XNS[0], st[0].mst + mi * D * 6, MIXC(l, mi), D);
            TICK(PP_ATT);
            return;
        }
        if (M[MF_ATTMASK] >> l & 1) { attention(ns, l, ai++, t); TICK(PP_ATT); }
        else { for (int s = 0; s < ns; s++) mixer_k(XS[s], XNS[s], st[s].mst + mi * D * 6, MIXC(l, mi), D); mi++; TICK(PP_MIXER); }
        if (pix_yield) pix_yield();
        for (int s = 0; s < ns; s++) norm(XNS[s], XS[s], B32(LAY(l, LF_N2)), D);
        TICK(PP_NORM);
        int hs = (int)M[MF_HS];
        if (LAY(l, LF_HDL)) {                                /* dense MLP: only its live hidden units are stored */
            int hdl = (int)LAY(l, LF_HDL);
            int32_t m1 = B32(LAY(l, LF_W1M))[0], m2 = B32(LAY(l, LF_W2M))[0];
            int sh = ns > 1 ? XSH : 0, na = -1;
            uint32_t *sums = 0;
#ifdef __wasm__
            uint32_t sm2[2];                                 /* (scan_exit3_relu's sums) */
#endif
            const int32_t *kept = 0;                         /* what the tables are a build of when the second matrix starts */
#ifdef __arm__
            ScanR *r = &scanr[ns > 1][l];                    /* set up by pix_begin when the fused row kernel covers this layer */
            if (r->rows) {
                if (ns > 1) {
                    pack2(sc, XNS[0], XNS[1], D); tern_build81x(tbl, sc, D / 4 - 4 * W1X_NCK, w1xb2[l]);     /* (-in0 in both halves, and the limits) */
                    na = tern_scan_relu(hid, BU8(LAY(l, LF_W1)), tbl, D / 16, r);
                } else {                                     /* early exit: the limits are in the tables (w1xb, see tern_scan_x) */
                    tern_build81x(tbl, XNS[0], D / 4 - 4 * W1X_NCK, w1xb[l]);
                    na = tern_scan_x(aval, BU8(LAY(l, LF_W1)), tbl, D / 16, r);
                }
                sums = &r->sum0; kept = ns > 1 ? sc : XNS[0];
            }
#endif
#ifdef PIX_W1PROBE
            { extern void (*pix_w1probe)(int, int, const int32_t *, int); if (pix_w1probe) for (int s = 0; s < ns; s++) pix_w1probe(l, ns > 1 ? s + 1 : 0, XNS[s], D); }
#endif
            if (na < 0) {
#ifndef __arm__
                /* (the same conditions as the GBA's row kernels: see pix_begin) */
                int x1 = pix_w1x && LAY(l, LF_W1X) && (D == 128 || D == 192) && QB + hs >= 2 && QB + hs <= 31 && relu_in0(p0_l[l], m1) <= (1 << 20);
                int x2 = pix_w1x && pix_w1x2 && LAY(l, LF_W1X2) && (D == 128 || D == 192) && relu_in0(p0_l[l], m1 << XSH) <= 4096;
                if (x2) for (int j = 0; j < W1X_NCK; j++) { int32_t t = B32(LAY(l, LF_W1X2))[j]; if (t > 4096 || t < -4096) x2 = 0; }
                if (ns == 1 && x1) {
                    tb(XNS[0], 0, D);
#ifdef __wasm__
                    na = scan_exit_relu(hid, BU8(LAY(l, LF_W1)), tbl, D / 4, hdl, B32(LAY(l, LF_W1X)), m1 << sh, QB + hs, RT(p0_l[l], tau_l[l], m1 << sh));
#else
                    scan_exit(hid, BU8(LAY(l, LF_W1)), tbl, D / 4, hdl, B32(LAY(l, LF_W1X)));
#endif
                } else if (ns > 1 && x1 && pix_w1xs0 && D == 128 && relu_in0(p0_l[l], m1 << XSH) <= 4096) {      /* (the GBA's two-stream row kernel: 8 words) */
                    int32_t lim[W1X_NCK];
                    int ok = 1;
                    for (int j = 0; j < W1X_NCK; j++) { lim[j] = (B32(LAY(l, LF_W1X))[j] + (1 << (XSH - 1))) >> XSH; if (lim[j] > 4096 || lim[j] < -4096) ok = 0; }
                    if (!ok) for (int j = 0; j < W1X_NCK; j++) lim[j] = -16384;
                    tb(XNS[0], XNS[1], D);
#ifdef __wasm__
                    na = scan_exit3_relu(hid, BU8(LAY(l, LF_W1)), tbl, D / 4, hdl, lim, relu_in0(p0_l[l], m1 << XSH), m1 << sh, QB + hs, RT(p0_l[l], tau_l[l], m1 << sh), sums = sm2);
#else
                    scan_exit3(hid, BU8(LAY(l, LF_W1)), tbl, D / 4, hdl, lim, relu_in0(p0_l[l], m1 << XSH));
#endif
                } else if (ns > 1 && x2) {                   /* (study only: the GBA has no kernel for this) */
                    pack2(sc, XNS[0], XNS[1], D);
                    for (int i = 0; i < D; i++) sc[i] = (int32_t)((uint32_t)sc[i] + ((uint32_t)sc[i] << 16));
                    tern_build(tbl, sc, D / 4);
                    scan_exit2(hid, BU8(LAY(l, LF_W1)), tbl, D / 4, hdl, B32(LAY(l, LF_W1X2)), relu_in0(p0_l[l], m1 << XSH));
                } else
#endif
                {
                tb(XNS[0], ns > 1 ? XNS[1] : 0, D);          /* with two streams the rows stay packed: relu_sq2 takes them apart */
                scan(hid, BU8(LAY(l, LF_W1)), tbl, D / 4, hdl);
                }
            }
            TICK(PP_W1);
            if (ns > 1) mlp_out2(BU8(LAY(l, LF_W2)), hdl, m1 << sh, QB + hs, m2, na, sums, kept, l);
            else {
                if (na < 0) na = relu_sq(aval, aidx, hid, hdl, m1 << sh, QB + hs, RT(p0_l[l], tau_l[l], m1 << sh));
                if (m2) sparse_out(XS[0], BU8(LAY(l, LF_W2)), aval, na, D, m2, kept);
            }
        } else {
            routed(ns, l, t, pg);
            TICK(PP_W1);
        }
        TICK(PP_W2);
        if (pix_yield) pix_yield();
    }
    for (int s = 0; s < ns; s++) norm(XNS[s], XS[s], B32(M[MF_NF]), D);
    TICK(PP_NORM);
    int sh = tb(XNS[0], ns > 1 ? XNS[1] : 0, D);
    rows(PJ[0], ns > 1 ? PJ[1] : 0, BU8(M[MF_HIN]), D / 4, DH);
    int32_t hm = (int32_t)M[MF_HIN_M];
    for (int s = 0; s < ns; s++) scale_rs(st[s].hin, PJ[s], hm, DH, 16 - sh);
    if (NS == 1) { TICK(PP_HIN); return; }                    /* one pixel per cell: the head runs its first matrix itself */
    /* the head's first matrix on this cell's part of its input, once per cell (the table rows have it built in) */
    const int32_t *gn = B32(M[MF_HN]);
    int32_t h1m = (int32_t)M[MF_H1M]; int ysh = 16 + QB - YQ;
    for (int s = 0; s < ns; s++) {
        mulrs_k(z, st[s].hin, gn, DH, QB);
        matsum(hid, BU8(M[MF_H1]), z, DH, HH);
        scale_rs(st[s].yin, hid, h1m, HH, ysh);
    }
    hbase = DH <= CHG * 4 ? z : 0;
    TICK(PP_HIN);
}

/* head for one pixel of one stream: leaves the table of the normalised head state built and 16 group scores in g.
 * facc: this pixel's table rows summed (Q16); yacc: the same rows after the first matrix (Q of MF_YQ). */
static void head(const Stream *sm, int32_t *g, const int32_t *ts) {
    if (ts) head_u(hu, sm->hin, facc, ts, DH);               /* ts: this pixel's thumbnail block rows, not yet in facc */
    else if (NS == 1) head_u(hu, sm->hin, facc, cellsum, DH);        /* one pixel per cell: cellsum stays all zeros */
#ifdef __wasm__
    else for (int i = 0; i < DH; i++) hu[i] = sm->hin[i] + RS32(facc[i], 4);
#else
    else for (int i = 0; i < DH; i++) hu[i] = sm->hin[i] + (int32_t)RS((int64_t)facc[i], 4);
#endif
    uint32_t r = isqrt64(div_small(sumsq_k(hu, DH), (uint32_t)DH) + 168);
    if (r < 16) r = 16;
    int32_t minv = (int32_t)(div_shl(1, 40, r) >> YQ), m2 = (int32_t)M[MF_H2M];
    int hs = (int)M[MF_HSH];
    const int32_t *hv = hid;
    if (NS == 1) {
        const int32_t *gn = B32(M[MF_HN]);
        mulrs_k(z, hu, gn, DH, QB);
        matsum(hid, BU8(M[MF_H1]), z, DH, HH);
        int32_t h1m = (int32_t)M[MF_H1M]; int ysh = 16 + QB - YQ;
        scale_rs(hid, hid, h1m, HH, ysh);
    } else if (sm == st) hv = yacc;                           /* stream 0's yin is in yacc already (pix_cell) */
    else for (int i = 0; i < HH; i++) hid[i] = yacc[i] + sm->yin[i] - st[0].yin[i];
    int na = relu_sq(aval, aidx, hv, HH, minv, QB + hs, RT(p0_h, tau_h, minv));
    TICK(PP_H1);
    if (m2) sparse_out(hu, h2cols, aval, na, DH, m2, NS == 1 ? (DH <= CHG * 4 ? z : 0) : hbase);
    norm(z, hu, hf_g, DH);
    TICK(PP_H2);
    tern_build(tbl, z, DH / 4); hbase = z;
    scan(g, outg, tbl, DH / 4, 16);
    mulrs_k(g, g, outg_m, 16, 16);
    TICK(PP_OUT);
}
IWRAM_CODE static void members(int gi, int32_t *m) {                    /* 16 colour scores of group gi from the table that is built */
    scan(m, BU8(M[MF_OUTM]) + (uint32_t)gi * 16 * ROWB(DH / 4), tbl, DH / 4, 16);
    mulrs_k(m, m, outm_m + gi * 16, 16, 16);
}
IWRAM_CODE static int pick(const int32_t *s) {                          /* sample one of 16 by Q12 score at the set temperature */
    int32_t mx = s[0];
    for (int i = 1; i < 16; i++) if (s[i] > mx) mx = s[i];
    uint32_t w[16], sum = 0;
    for (int i = 0; i < 16; i++) {
        uint32_t d = (uint32_t)(((int64_t)(mx - s[i]) * invt) >> 8) >> (QB - PIX_EXP_BITS);
        w[i] = d < PIX_EXP_N ? BU16(O_EXP)[d] : 0;
        sum += w[i];
    }
    uint32_t r = *rngp;
    r ^= r << 13; r ^= r >> 17; r ^= r << 5; *rngp = r;
    uint32_t v = (uint32_t)(((uint64_t)r * sum) >> 32);
    for (int i = 0; i < 16; i++) { if (v < w[i]) return i; v -= w[i]; }
    return 15;
}

ROM_CODE uint32_t pix_info(int m, int f) { return ((const uint32_t *)(pix_blob + O_MODELS))[m * MF_N + f]; }
ROM_CODE const int16_t *pix_perm(void) { return B16(M[MF_PERM]); }
ROM_CODE int pix_cells_done(void) { return cellno; }

ROM_CODE int pix_begin(int m, const uint16_t *words, const uint8_t *th, int guide_q8, int inv_temp_q8, uint32_t *rng) {
    M = (const uint32_t *)(pix_blob + O_MODELS) + m * MF_N;
    LY = (const uint32_t *)(pix_blob + M[MF_LAYERS]);
    S = M[MF_S]; C = M[MF_C]; D = M[MF_D]; L = M[MF_L]; NH = M[MF_NH]; DK = M[MF_DK]; DV = M[MF_DV]; HD = M[MF_HD]; DH = M[MF_DH]; HH = M[MF_HH];
    HK = DK / NH; HV = DV / NH; KB = M[MF_KB]; NC = M[MF_NC]; NS = M[MF_NS]; WIN = M[MF_WIN];
    tau_h = pix_tau_h >= 0 ? pix_tau_h : (int32_t)M[MF_TAUH]; p0_h = relu_p0(QB + (int)M[MF_HSH], tau_h);
    for (int l = 0; l < L; l++) {                            /* the body's thresholds are per layer (PIX_TAUL; MF_TAUB is the model's plain one) */
        tau_l[l] = pix_tau_b >= 0 ? pix_tau_b : pix_tau_b == -2 || !PIX_TAUL ? (int32_t)M[MF_TAUB] : (int32_t)LAY(l, LF_TAU);
#ifndef __arm__
        if (pix_taul && pix_taul[l] >= 0) tau_l[l] = pix_taul[l];
#endif
        p0_l[l] = relu_p0(QB + (int)M[MF_HS], tau_l[l]);
    }
#ifdef __arm__
    for (int l = 0; l < L; l++) for (int two = 0; two < 2; two++) {
        ScanR *r = &scanr[two][l];
        int hdl = (int)LAY(l, LF_HDL), sft = QB + (int)M[MF_HS];
        r->rows = 0;
        if (!hdl || (D != 128 && D != 192) || (two && D != 128)) continue;      /* the fused kernels take rows of 8 or 12 words (two streams: 8) */
        if (sft < 2 || sft > 31) continue;                   /* (its shifts are immediates) */
        int32_t m = B32(LAY(l, LF_W1M))[0] << (two ? XSH : 0), in0 = relu_in0(p0_l[l], m);
        if (in0 > (two ? 4096 : 1 << 20)) continue;          /* row sums minus in0 must stay far inside their words (see DESIGN.md) */
        int32_t *vals = two ? hid : aval;
        r->m = m; r->shift = sft; r->sh32 = 32 - sft; r->round = 1 << (sft - 1); r->in0 = in0; r->rows = hdl;
        if (!two) {                                          /* the early exit's limits (W1X); without them, limits that never trigger */
            const int32_t *lim = W1X && LAY(l, LF_W1X) ? B32(LAY(l, LF_W1X)) : 0;
            for (int j = 0; j <= W1X_NCK; j++) w1xb[l][j] = (j ? lim ? lim[j - 1] : -(1 << 30) : 0) - (j < W1X_NCK ? lim ? lim[j] : -(1 << 30) : in0);
        } else {                                             /* two streams (PIX_W1XS0): stream 0's limits in the high half, scaled like the packed inputs */
            int32_t lp[W1X_NCK], use = W1X && PIX_W1XS0 && LAY(l, LF_W1X) && scanr[0][l].rows;
            for (int j = 0; j < W1X_NCK; j++) {
                lp[j] = use ? (B32(LAY(l, LF_W1X))[j] + (1 << (XSH - 1))) >> XSH : -16384;
                if (lp[j] > 4096 || lp[j] < -4096) use = 0;
            }
            for (int j = 0; j < W1X_NCK && !use; j++) lp[j] = -16384;      /* (never below: the high half is at most about 5,000) */
            for (int j = 0; j <= W1X_NCK; j++)
                w1xb2[l][j] = (int32_t)((uint32_t)((j ? lp[j - 1] : 0) - (j < W1X_NCK ? lp[j] : in0)) << 16) - (j ? 0 : in0);
        }
        r->idxc = (uint32_t)aidx - ((uint32_t)vals >> 1); r->difoff = (uint32_t)((uint8_t *)aval - (uint8_t *)hid); r->two = two;
    }
    {   /* the fused kernel's shift by QB + HS is written into its instructions (kernels.s, relu_sites) */
        extern uint32_t *const relu_sites[9];
        int sft = QB + (int)M[MF_HS];
        if (sft >= 2 && sft <= 31) {
            int k = sft - 1;                                 /* rounding term 2^k as an immediate: 1 or 2, rotated right by an even count */
            uint32_t imm = k & 1 ? (uint32_t)(((33 - k) / 2 & 15) << 8) | 2 : (uint32_t)(((32 - k) / 2 & 15) << 8) | 1;
            for (int i = 0; i < 3; i++) {
                *relu_sites[i] = (*relu_sites[i] & ~0xFFFu) | imm;
                *relu_sites[3 + i] = (*relu_sites[3 + i] & ~0xF80u) | (uint32_t)sft << 7;
                *relu_sites[6 + i] = (*relu_sites[6 + i] & ~0xF80u) | (uint32_t)(32 - sft) << 7;
            }
        }
    }
    {
        int32_t *t1 = ycell;                                 /* ycell and tsum: PIX_MAX_HH + 4 * PIX_MAX_DH words */
        int fast1 = PIX_S1_FAST && NS == 1 && !M[MF_F] && D <= PIX_MAX_DH && DV <= PIX_MAX_DV && DK <= 256;
        XS[0] = x; XNS[0] = xn; PJ[0] = proj; MV[0] = mixv;
        XS[1] = fast1 ? t1 : x1; XNS[1] = fast1 ? t1 + PIX_MAX_DH : xn1; PJ[1] = fast1 ? t1 + 2 * PIX_MAX_DH : proj1;
        MV[1] = fast1 ? t1 + 2 * PIX_MAX_DH + 256 : mixv1;
        st[1].hin = fast1 ? t1 + 2 * PIX_MAX_DH + 256 + PIX_MAX_DV : hin1;
    }
    norm_fast = 1;
    for (int k = 0; k < 2 * L + 2; k++) {
        const int32_t *g = B32(k < 2 * L ? LAY(k >> 1, k & 1 ? LF_N2 : LF_N1) : k == 2 * L ? M[MF_NF] : M[MF_HF]);
        for (int i = (k == 2 * L + 1 ? DH : D) - 1; i >= 0; i--) if (g[i] > GMAX || g[i] < -GMAX) norm_fast = 0;
    }
#endif
    NATT = 0;
    for (int l = 0; l < L; l++) NATT += M[MF_ATTMASK] >> l & 1;
#ifdef __arm__
    /* keytop: keys of 3 words, or of 2 words with its early row exit, which wants sign-only keys of 32 values and adds
     * up 16 query values of at most 2 sqrt(HK) * gain / 16 each (gains below 2^20 keep that far inside 32 bits) */
    kscan = KB == 12 || (KB == 8 && HK == 32) ? keytop : keysel4;
    if (KB == 8) for (int l = 0; l < L; l++) if (M[MF_ATTMASK] >> l & 1) for (int h = 0; h < NH; h++) if ((uint32_t)B32(LAY(l, LF_ATT))[2 + h] >> 20) kscan = keysel4;
#endif
    NCW = L * D;
    int any = 0;
    if (words) for (int i = 0; i < 8; i++) any |= words[i];
    twice = guide_q8 != 256 && any;
    guide = guide_q8; invt = inv_temp_q8; rngp = rng; cellno = 0;
    gcut = NC * pix_gcut16 / 16;
    awin = pix_awin >= 0 ? pix_awin : (int)M[MF_AWIN];
    uint32_t nb = (uint32_t)NATT * NH * NC * KB + (uint32_t)NATT * NC * DV + (uint32_t)NATT * NC, nw = (uint32_t)(L - NATT) * D * 6 + NCW;
    if ((D | DH) & 15) return -1;                            /* the kernels take blocks of 8 values */
#ifdef __arm__
    if ((D != 128 && D != 192) || (DH != 128 && DH != 192)) return -1;       /* the transposed matrices have kernels for 32 and 48 groups only */
#endif
    YQ = (int)(M[MF_YQ] & 255);                              /* bit 8: the IN_Y rows carry the slot rows (export.py), which pix_cell expects */
    if (NS > 1 && !(M[MF_YQ] >> 8 & 1)) return -1;
    if (nb * (twice + 1) > ARENA_B || nw * (twice + 1) > ARENA_W || (NS > 1 && nw * (twice + 1) > ARENA_W - 2048)) return -1;
    for (int s = 0; s <= twice; s++) {
        Stream *sm = &st[s];
        uint8_t *b = arena_b + s * nb;
        sm->keys = b; sm->vals = (int8_t *)(b + (uint32_t)NATT * NH * NC * KB); sm->vsh = b + (uint32_t)NATT * NH * NC * KB + (uint32_t)NATT * NC * DV;
        sm->mst = arena_w + s * nw; sm->cw = sm->mst + (L - NATT) * D * 6;
        for (uint32_t i = 0; i < nw; i++) sm->mst[i] = 0;
        sm->first = 0; sm->yin = s ? yin1 : v1buf[2];
#ifdef __arm__
        if (!s) sm->hin = hin0;                              /* (stream 1's was placed above) */
#else
        sm->hin = s ? hin1 : hin0;
#endif
       
        if (s == 0 && words) {                               /* stream 0 carries the prompt, stream 1 the empty prompt */
            sm->first = words[0];
            for (int k = 0; k < 8; k++) {
                if (!words[k]) continue;
                addrow(sm->cw, M[MF_CLS], M[MF_CLS_ST], words[k], D);
                if (M[MF_CLSL]) for (int l = 1; l < L; l++) addrow(sm->cw + l * D, LAY(l, LF_CLSL), LAY(l, LF_CLSL_ST), words[k], D);
            }
        }
        for (int i = D; i < NCW; i++) sm->cw[i] = (int32_t)RS((int64_t)sm->cw[i], 4);      /* later layers' prompt sums, ready to add */
    }
    h2cols = BU8(M[MF_H2]); outg = BU8(M[MF_OUTG]);
    ytsum[0] = v1buf[3]; ytsum[1] = v1buf[4]; ytsum[2] = v1buf[0]; ytsum[3] = v1buf[1];
    int v1words = VRAM1_WORDS;                               /* what this pass uses of vram1, from its top */
    (void)v1words;
    if (PIX_V1SQUEEZE && NS > 1 && !twice) {                 /* one stream: stream 1's x / xn, proj and yin are idle */
        ytsum[2] = yin1; ytsum[3] = proj1; st[0].yin = x1; v1words = 2 * PIX_MAX_HH;
    }
    hf_g = B32(M[MF_HF]); outg_m = B32(M[MF_OUTG_M]); outm_m = B32(M[MF_OUTM_M]);
#if defined(__arm__) && !defined(PIX_NO_HCOPY)
    if (DH <= PIX_MAX_DH) {
        copy32(hcopy, hf_g, DH); copy32(hcopy + PIX_MAX_DH, outg_m, 16); copy32(hcopy + PIX_MAX_DH + 16, outm_m, 256);
        hf_g = hcopy; outg_m = hcopy + PIX_MAX_DH; outm_m = hcopy + PIX_MAX_DH + 16;
    }
#endif
#ifdef __arm__
    {   /* (exact) copies in the arena's spare words, when this pass leaves enough: passes with several pixels per cell
         * keep the arena's top for vram1 (gba.ld), the guided base pass fills the arena */
        uint32_t used = nw * (uint32_t)(twice + 1), lim = NS > 1 ? ARENA_W - (uint32_t)v1words : ARENA_W, n = (uint32_t)HH * T3W(DH / 4);
#ifdef PIX_NO_H2COPY
        lim = 0;
#endif
        if (used + n <= lim) { copy32(arena_w + used, B32(M[MF_H2]), (int)n); h2cols = (const uint8_t *)(arena_w + used); used += n; }
        n = 16 * (uint32_t)T3W(DH / 4);
        if (used + n <= lim) { copy32(arena_w + used, B32(M[MF_OUTG]), (int)n); outg = (const uint8_t *)(arena_w + used); used += n; }
    }
#endif
    if (M[MF_F]) for (int i = 0; i < (S / (int)M[MF_F]) * (S / (int)M[MF_F]); i++) thumb[i] = th[i];
    if (M[MF_F]) {
        int Fc = (int)M[MF_F], bw = C / Fc > 1 ? C / Fc : 1;
        for (int j = 0; j < NS; j++) blkof[j] = (uint8_t)(((j / C) / Fc) * bw + (j % C) / Fc);
        for (int b = 0; b < bw * bw && b < 4; b++) blkpix[b] = (uint8_t)((b / bw) * Fc * C + (b % bw) * Fc);
    }
    for (int i = 0; i < NC; i++) skipped[i] = 0;
    pix_nskip = 0; pix_nhskip = 0;
    kseg_prep();
#ifdef __wasm__
    for (int l = 0, mi = 0; l < L; l++) if (!(M[MF_ATTMASK] >> l & 1)) {        /* the mixers' coefficients by columns (mixer_k) */
        const int32_t *c = B32(LAY(l, LF_MIXC));
        for (int d = 0; d < D; d++) for (int j = 0; j < 6; j++) pix_mixcf[mi * D * 6 + j * D + d] = c[d * 6 + j];
        mi++;
    }
#endif
    for (int i = 0; i < S * S; i++) pix_img[i] = 0;
    return 0;
}
#ifdef __wasm__
/* (browser) what a pass carries from one cell to the next besides its keys and values (those stay in the pass's own
 * arena, row t written by cell t only): the counters, the picture so far, the flat cells and the mixer state. The rest
 * is set by pix_begin or rewritten by every cell before it is read. pix_save writes it to p (0: more than max bytes),
 * pix_load reads it back after pix_begin of the same pass. */
static void cp16(void *d, const void *s, uint32_t n) { for (uint32_t i = 0; i < n; i += 16) wasm_v128_store((uint8_t *)d + i, wasm_v128_load((const uint8_t *)s + i)); }
static uint32_t pix_mstb(void) { return (uint32_t)(L - NATT) * D * 6 * 4; }        /* one stream's mixer state, bytes */
uint32_t pix_save(uint8_t *p, uint32_t max) {
    uint32_t n = 16 + (uint32_t)(S * S) + ((uint32_t)NC + 15) / 16 * 16, nm = pix_mstb();
    if (n + (uint32_t)(twice + 1) * nm > max) return 0;
    int32_t *h = (int32_t *)p;
    h[0] = cellno; h[1] = twice; h[2] = pix_nskip; h[3] = pix_nhskip;
    cp16(p + 16, pix_img, (uint32_t)(S * S)); cp16(p + 16 + S * S, skipped, ((uint32_t)NC + 15) / 16 * 16);
    for (int s = 0; s <= twice; s++, n += nm) cp16(p + n, st[s].mst, nm);
    return n;
}
void pix_load(const uint8_t *p) {
    const int32_t *h = (const int32_t *)p;
    uint32_t n = 16 + (uint32_t)(S * S) + ((uint32_t)NC + 15) / 16 * 16, nm = pix_mstb();
    cellno = h[0]; twice = h[1]; pix_nskip = h[2]; pix_nhskip = h[3];
    cp16(pix_img, p + 16, (uint32_t)(S * S)); cp16(skipped, p + 16 + S * S, ((uint32_t)NC + 15) / 16 * 16);
    for (int s = 0; s <= twice; s++, n += nm) cp16(st[s].mst, p + n, nm);
}
#endif

#ifndef __arm__
int pix_shadow;                                              /* tests: also run the model on cells that are painted flat (probe sees them), changing nothing */
int pix_in_shadow;
#define in_shadow pix_in_shadow
#ifdef __wasm__
#undef in_shadow
#define in_shadow 0
#define pix_shadow 0
#else
static void shadow_cell(int t) {
    static int32_t keep[ARENA_W]; static uint8_t img[64 * 64];
    uint32_t r = *rngp; int tw = twice, ns = pix_nskip;
    for (int i = 0; i < ARENA_W; i++) keep[i] = arena_w[i];
    for (int i = 0; i < S * S; i++) img[i] = pix_img[i];
    in_shadow = 1; cellno = t; pix_cell(); in_shadow = 0;
    for (int i = 0; i < ARENA_W; i++) arena_w[i] = keep[i];
    for (int i = 0; i < S * S; i++) pix_img[i] = img[i];
    *rngp = r; twice = tw; pix_nskip = ns; cellno = t;
}
#endif
#endif
/* the own rule (see PIX_FLAT_OWN / PIX_HEADSKIP): cell t's own four thumbnail blocks are one colour and at most nd of
 * the eight blocks that touch them differ from it (cells of 2 x 2 blocks) */
ROM_CODE static int own_flat(int t, int nd_max) {
    const uint16_t *pal = BU16(O_PAL);
    const int16_t *tp0 = B16(M[MF_TP]) + t * NS * 5;
    int own[4], nd = 0;
    for (int b = 0; b < 4; b++) own[b] = tp0[blkpix[b] * 5];
    int c = pal[thumb[own[0]]];
    for (int b = 1; b < 4; b++) if (pal[thumb[own[b]]] != c) return 0;
    for (int b = 0; b < 4; b++)
        for (int q = 1; q < 5; q++) {
            int i = tp0[blkpix[b] * 5 + q];
            if (i < 0 || i == own[0] || i == own[1] || i == own[2] || i == own[3]) continue;
            if (pal[thumb[i]] != c && ++nd > nd_max) return 0;
        }
    return 1;
}
/* (PIX_HEADSKIP) cell t: keys, values and mixer state only; pixels from the thumbnail */
__attribute__((noinline)) ROM_CODE static int cell_kv(int t, const int16_t *perm) {
    kvonly = 1; body(1, t); kvonly = 0;
#ifndef __arm__
    if (pix_probe && !pix_shadow) { int32_t g[16], all[256]; for (int i = 0; i < 16; i++) g[i] = -0x7FFFFFFF; for (int i = 0; i < 256; i++) all[i] = 0; for (int j = 0; j < NS; j++) pix_probe(t, j, g, all); }
#endif
    const int16_t *tp0 = B16(M[MF_TP]) + t * NS * 5;
    for (int j = 0; j < NS; j++) pix_img[perm[j]] = thumb[tp0[j * 5]];
#if !defined(__arm__) || defined(PIX_FORCE)
    if (pix_force) for (int j = 0; j < NS; j++) if (FORCED(perm[j])) pix_img[perm[j]] = pix_force[perm[j]];
#endif
    pix_nhskip++; cellno = t + 1;
    return NC - cellno;
}
/* (PIX_HEADSKIP) every painted pixel bordering cell t (the causal neighbours of its pixels that lie outside it) has
 * the colour of the cell's own blocks */
ROM_CODE static int border_same(int t, const int16_t *perm, const int16_t *nb) {
    int c = thumb[B16(M[MF_TP])[(t * NS + blkpix[0]) * 5]];
    for (int k = 0; k < NS * 4; k++) {
        int i = nb[k], in = 0;
        if (i < 0) continue;
        for (int j = 0; j < NS; j++) if (perm[j] == i) in = 1;
        if (!in && pix_img[i] != c) return 0;
    }
    return 1;
}
/* (PIX_HEADSKIP) may cell t be a body-only cell? cells of 2 x 2 blocks: the own rule; cells of one block (pix_hsnd32 >= 0):
 * at most pix_hsnd32 blocks of the window differ in colour from the cell's own block */
ROM_CODE static int hs_rule(int t, const int16_t *perm, const int16_t *nb) {
    if (NS <= 1) return 0;
    int r = C / (int)M[MF_F];
    if (r == 2) return own_flat(t, pix_hsnd) && (!pix_hsb || border_same(t, perm, nb));
    if (r != 1 || pix_hsnd32 < 0) return 0;
    const uint16_t *pal = BU16(O_PAL);
    const int16_t *tw = B16(M[MF_TW]) + t * WIN;
    int c = pal[thumb[B16(M[MF_TP])[t * NS * 5]]], nd = 0;
    for (int k = 0; k < WIN; k++) if (tw[k] >= 0 && pal[thumb[tw[k]]] != c && ++nd > pix_hsnd32) return 0;
    return !pix_hsb || border_same(t, perm, nb);
}
/* PIX_PCOPY: pixel j of cell t: its own block and the (up to four) blocks beside it one colour (in the 64 x 64 pass at
 * most PIX_PCOPY_ND of those four may differ), and every drawn neighbour pixel that colour too */
static int pcopy_ok(int t, int j) {
    const uint16_t *pal = BU16(O_PAL);
    const int16_t *tp = B16(M[MF_TP]) + (t * NS + j) * 5, *nb = B16(M[MF_NBIX]) + (t * NS + j) * 4;
    int c = pal[thumb[tp[0]]];
    int nd = 0, ndmax = C / (int)M[MF_F] == 2 ? PIX_PCOPY_ND : 0;  /* (the 64 x 64 pass may have one block beside it differ) */
    for (int q = 1; q < 5; q++) if (tp[q] >= 0 && pal[thumb[tp[q]]] != c && ++nd > ndmax) return 0;
    for (int q = 0; q < 4; q++) if (nb[q] >= 0 && pal[pix_img[nb[q]]] != c) return 0;
    return 1;
}
int pix_cell(void) {
    int t = cellno;
    if (t >= NC) return 0;
    const int16_t *perm = B16(M[MF_PERM]) + t * NS, *nb = B16(M[MF_NBIX]) + t * NS * 4;
    if (twice && t >= gcut) twice = 0;                      /* past here guidance no longer changes the picture visibly */
    int hskip = 0;
#if !defined(__arm__) && !defined(__wasm__)
    if (!in_shadow) pix_cellclass[t] = 0;
#endif
#ifndef __arm__
    if (!in_shadow)
#endif
    if (M[MF_F] && pix_flatskip && t > 0) {                  /* paint the cell straight from the thumbnail? */
        const uint16_t *pal = BU16(O_PAL);
        int flat = 1;
        if (pix_flatrule && NS > 1 && C / (int)M[MF_F] == 2) {
            /* Rule for cells of 2 x 2 thumbnail blocks (PIX_FLAT_OWN, see DESIGN.md) */
            flat = own_flat(t, PIX_FLAT_ND);
        } else {                                             /* every thumbnail block around the cell within tol of the first */
            const int16_t *tw = B16(M[MF_TW]) + t * WIN;
            int c = -1, tol = (int)M[MF_FLATT];              /* tol: largest difference per colour channel, in 5-bit steps */
            for (int k = 0; k < WIN && flat; k++) if (tw[k] >= 0) {
                int v = thumb[tw[k]];
                if (c < 0) c = v;
                else if (v != c) {
                    int a = pal[v], b = pal[c], d0 = (a & 31) - (b & 31), d1 = (a >> 5 & 31) - (b >> 5 & 31), d2 = (a >> 10) - (b >> 10);
                    if (d0 < -tol || d0 > tol || d1 < -tol || d1 > tol || d2 < -tol || d2 > tol) flat = 0;
                }
            }
        }
        hskip = !flat && pix_headskip && !pix_flatrule && !twice && hs_rule(t, perm, nb);
#if !defined(__arm__) && !defined(__wasm__)
        {   const int16_t *tq = B16(M[MF_TP]) + t * NS * 5;
            for (int j = 0; j < NS; j++) pix_copycol[t * NS + j] = thumb[tq[j * 5]];
            pix_cellclass[t] = (uint8_t)((flat ? 1 : 0) | (hs_rule(t, perm, nb) ? 2 : 0)); }
        if ((flat || hskip) && pix_shadow) { shadow_cell(t); if (flat) flat = 2; }
#endif
        if (flat) {
#ifndef __arm__
            if (pix_probe && flat == 1) { int32_t g[16], all[256]; const int16_t *tq = B16(M[MF_TP]) + t * NS * 5; for (int i = 0; i < 16; i++) g[i] = -0x7FFFFFFF; for (int i = 0; i < 256; i++) all[i] = 0;
                for (int j = 0; j < NS; j++) { all[0] = thumb[tq[j * 5]]; all[1] = 0; pix_probe(t, j, g, all); } }     /* (the copied colour and the kind, for tools/pcopy.py) */
#endif
            const int16_t *tp0 = B16(M[MF_TP]) + t * NS * 5;
            for (int j = 0; j < NS; j++) pix_img[perm[j]] = thumb[tp0[j * 5]];      /* each pixel takes its own block's colour */
#if !defined(__arm__) || defined(PIX_FORCE)
            if (pix_force) for (int j = 0; j < NS; j++) if (FORCED(perm[j])) pix_img[perm[j]] = pix_force[perm[j]];
#endif
            skipped[t] = 1; pix_nskip++;
            cellno = t + 1;
            return NC - cellno;
        }
    }
    if (hskip) return cell_kv(t, perm);
#if !defined(__arm__) && !defined(__wasm__)
    if (M[MF_F] && !in_shadow) pix_cellclass[t] |= 4;
#endif
#ifdef __wasm__
    kvheld = pix_force != 0;                                 /* (browser) every pixel of the cell held? */
    for (int j = 0; j < NS && kvheld; j++) if (!FORCED(perm[j])) kvheld = 0;
#endif
    body(twice + 1, t);
#ifdef __wasm__
    if (kvheld) {                                            /* nothing for the head: the pixels, and the draws a drawn pixel makes */
        kvheld = 0;
        for (int j = 0; j < NS; j++) {
            if (!(pix_pcopy && M[MF_F] && NS > 1 && pcopy_ok(t, j))) {        /* (a copied pixel makes none) */
                uint32_t r = *rngp; r ^= r << 13; r ^= r >> 17; r ^= r << 5; r ^= r << 13; r ^= r >> 17; r ^= r << 5; *rngp = r; }
            pix_img[perm[j]] = pix_force[perm[j]];
        }
        cellno = t + 1;
        return NC - cellno;
    }
#endif
    if (DH & 7) for (int i = 0; i < DH; i++) cellsum[i] = 0; else zero32(cellsum, DH / 8);
    if (NS > 1) { copy32(ycell, st[0].yin, HH); add16(ycell, B16(M[MF_SLOT_Y]), HH); }     /* starts from the cell's own part and slot 0's row */
    TICK0();
    int nblk = 0, bw = 1, Fc = (int)M[MF_F];
    if (Fc) {                                                /* pixels over the same thumbnail block share its five rows: sum them once */
        bw = C / Fc > 1 ? C / Fc : 1; nblk = bw * bw;
        for (int b = 0; b < nblk; b++) {
            int j = blkpix[b];                               /* first pixel of the block in this cell */
            const int16_t *tp = B16(M[MF_TP]) + (t * NS + j) * 5;
            int32_t *ts = tsum[b], *ys = ytsum[b];
            int row[5];
            for (int q = 0; q < 5; q++) row[q] = q * P1 + (tp[q] < 0 ? 256 : thumb[tp[q]]);
            const int8_t *ht = B8(M[MF_HTH]); const int32_t *hs = B32(M[MF_HTH_ST]); const int16_t *hy = B16(M[MF_HTH_Y]);
            zero32(ts, DH / 8); zero32(ys, HH / 8);          /* five rows in three passes each */
            axpy8_2(ts, ts, ht + (uint32_t)row[0] * DH, hs[row[0]], ht + (uint32_t)row[1] * DH, hs[row[1]], DH);
            axpy8_2(ts, ts, ht + (uint32_t)row[2] * DH, hs[row[2]], ht + (uint32_t)row[3] * DH, hs[row[3]], DH);
            addrow(ts, M[MF_HTH], M[MF_HTH_ST], row[4], DH);
            sum16_3(ys, ys, hy + (uint32_t)row[0] * HH, hy + (uint32_t)row[1] * HH, hy + (uint32_t)row[2] * HH, HH);
            add16(ys, hy + (uint32_t)row[3] * HH, HH); add16(ys, hy + (uint32_t)row[4] * HH, HH);
        }
    }
    TICK(PP_HTAB);
    const int16_t *iny = 0;
    for (int j = 0; j < NS; j++) {
        TICK0();
        if (pix_pcopy && Fc && NS > 1 && pcopy_ok(t, j)) {
            const int16_t *tp = B16(M[MF_TP]) + (t * NS + j) * 5;
            int px = thumb[tp[0]];
#ifndef __arm__
            if (pix_probe) { int32_t gg[16], all[256]; for (int i = 0; i < 16; i++) gg[i] = -0x7FFFFFFF; for (int i = 0; i < 256; i++) all[i] = 0; all[0] = px; all[1] = 2; pix_probe(t, j, gg, all); }
#ifndef __wasm__
            pix_npcopy++;
#endif
#endif
#if !defined(__arm__) || defined(PIX_FORCE)
            if (FORCED(perm[j])) px = pix_force[perm[j]];
#endif
            pix_img[perm[j]] = (uint8_t)px;
            if (iny) { add16(ycell, iny, HH); iny = 0; }      /* (the pixel before's row, which this pixel's sum would have taken in) */
            if (j + 1 < NS) {
                addrow(cellsum, M[MF_IN], M[MF_IN_ST], j * P1 + px, DH);
                iny = B16(M[MF_IN_Y]) + (uint32_t)(j * P1 + px) * HH;
            }
            TICK(PP_PICK);
            continue;
        }
#ifdef __wasm__
        if (FORCED(perm[j])) {                               /* (browser) a held pixel needs no scores: only its rows and the two draws */
            int px = pix_force[perm[j]];
            uint32_t r = *rngp;
            r ^= r << 13; r ^= r >> 17; r ^= r << 5; r ^= r << 13; r ^= r >> 17; r ^= r << 5; *rngp = r;
            pix_img[perm[j]] = (uint8_t)px;
            if (iny) { add16(ycell, iny, HH); iny = 0; }
            if (j + 1 < NS) {
                addrow(cellsum, M[MF_IN], M[MF_IN_ST], j * P1 + px, DH);
                iny = B16(M[MF_IN_Y]) + (uint32_t)(j * P1 + px) * HH;
            }
            continue;
        }
#endif
        int row[4];
        for (int q = 0; q < 4; q++) row[q] = q * P1 + (nb[j * 4 + q] < 0 ? 256 : pix_img[nb[j * 4 + q]]);
        const int8_t *nbt = B8(M[MF_NB]); const int32_t *nbs = B32(M[MF_NB_ST]), *ts = 0;
        axpy8_2(facc, cellsum, B8(M[MF_SLOT]) + (uint32_t)j * DH, B32(M[MF_SLOT_ST])[j], nbt + (uint32_t)row[0] * DH, nbs[row[0]], DH);
        axpy8_2(facc, facc, nbt + (uint32_t)row[1] * DH, nbs[row[1]], nbt + (uint32_t)row[2] * DH, nbs[row[2]], DH);
        addrow(facc, M[MF_NB], M[MF_NB_ST], row[3], DH);
        int b = Fc ? blkof[j] : 0;
        if (Fc) ts = tsum[b];                                /* head_u adds it */
        if (NS > 1) {
            const int16_t *ny = B16(M[MF_NB_Y]);
            /* the slot rows are folded into ycell (slot 0) and the IN_Y rows (export.py: the row of the pixel painted at
             * slot j carries slot j + 1's row minus slot j's), so a pixel adds one row fewer (exact) */
            if (iny) sum16_4u(yacc, ycell, ny + (uint32_t)row[0] * HH, ny + (uint32_t)row[1] * HH, ny + (uint32_t)row[2] * HH, iny, HH);     /* ycell takes in the pixel before */
            else sum16_3(yacc, ycell, ny + (uint32_t)row[0] * HH, ny + (uint32_t)row[1] * HH, ny + (uint32_t)row[2] * HH, HH);
            if (Fc) sum16_1w(yacc, ytsum[b], ny + (uint32_t)row[3] * HH, HH);
            else add16(yacc, ny + (uint32_t)row[3] * HH, HH);
        }
        TICK(PP_HTAB);
        int32_t g[16], g0[16], mm[16], m0[16];
        int gi, px;
        head(&st[0], g, ts);
#ifndef __arm__
        if (pix_probe) {
            int32_t all[256];
            for (int k = 0; k < 16; k++) members(k, all + k * 16);
            pix_probe(t, j, g, all);
        }
#endif
        if (twice) {
#ifndef __arm__
            int32_t all0[256], all1[256];                    /* tests: the guided scores of all 256 colours */
            if (pix_probe2) for (int k = 0; k < 16; k++) members(k, all0 + k * 16);
#endif
            copy32(zsave, z, DH);
            head(&st[1], g0, ts);
            for (int i = 0; i < 16; i++) g0[i] += (int32_t)(((int64_t)(g[i] - g0[i]) * guide) >> 8);
#ifndef __arm__
            if (pix_probe2) {
                for (int k = 0; k < 16; k++) members(k, all1 + k * 16);
                for (int i = 0; i < 256; i++) all1[i] += (int32_t)(((int64_t)(all0[i] - all1[i]) * guide) >> 8);
                pix_probe2(t, j, g0, all1);
            }
#endif
            gi = pick(g0);
            members(gi, m0);
            tern_build(tbl, zsave, DH / 4); hbase = 0;
            members(gi, mm);
            for (int i = 0; i < 16; i++) m0[i] += (int32_t)(((int64_t)(mm[i] - m0[i]) * guide) >> 8);
            px = gi * 16 + pick(m0);
        } else {
            gi = pick(g);
            members(gi, mm);
            px = gi * 16 + pick(mm);
        }
#if !defined(__arm__) || defined(PIX_FORCE)
        if (FORCED(perm[j])) px = pix_force[perm[j]];
#ifndef __arm__
        if (in_shadow && !FORCED(perm[j])) px = thumb[B16(M[MF_TP])[(t * NS + j) * 5]];
#endif
#endif
        pix_img[perm[j]] = (uint8_t)px;
        if (j + 1 < NS) {                                    /* the cell's last pixel has no reader */
            addrow(cellsum, M[MF_IN], M[MF_IN_ST], j * P1 + px, DH);
            iny = B16(M[MF_IN_Y]) + (uint32_t)(j * P1 + px) * HH;        /* added to ycell by the next pixel's sum16_4u */
        }
        TICK(PP_PICK);
    }
    cellno = t + 1;
    return NC - cellno;
}
