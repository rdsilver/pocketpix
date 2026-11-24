/* Picture model engine. One source for the GBA and for the native twin. Integers only. */
#ifndef PIX_H
#define PIX_H
#include <stdint.h>
#include "pix_layout.h"
#ifdef __arm__
extern const uint8_t pix_blob[];
#else
extern const uint8_t *pix_blob;
#endif
enum { PP_INPUT, PP_NORM, PP_MIXER, PP_ATT, PP_W1, PP_W2, PP_HIN, PP_HTAB, PP_H1, PP_H2, PP_OUT, PP_PICK, PP_N };
extern uint32_t pix_prof[PP_N];
extern void (*pix_yield)(void);              /* optional: called between layers (button polling, music) */
extern uint8_t pix_img[64 * 64];             /* the picture being painted: palette indices, S x S, reading order */
/* Start a picture with model m. words: up to 8 word ids (0 = empty slot), may be NULL. thumb: the finished smaller
 * picture for models that draw over one (it is copied). guide_q8: 256 = off, 768 = 3.0. inv_temp_q8: 256 / temperature. */
int pix_begin(int m, const uint16_t *words, const uint8_t *thumb, int guide_q8, int inv_temp_q8, uint32_t *rng);
int pix_cell(void);                          /* paint the next cell; returns the number of cells still to paint */
int pix_cells_done(void);
uint32_t pix_info(int m, int field);         /* a model's MF_ field */
const int16_t *pix_perm(void);               /* pixel index of slot j of cell t: perm[t * NS + j] */
#ifndef __arm__
extern const uint8_t *pix_force;             /* tests: write these pixels instead of sampling */
extern void (*pix_probe)(int t, int j, const int32_t *g, const int32_t *m_all);   /* tests: Q12 scores, 16 groups and 256 colours */
extern int64_t pix_maxsum;                   /* largest |row sum| seen (must stay below 2^31) */
#endif
#ifdef __wasm__
#define PIX_WASM_TBL 0x1F0000u                 /* the browser build's tables: two of 48 * 81 words (pix.c tbl, sparse_out) */
#endif
#endif
