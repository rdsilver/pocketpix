/* Native build of the cartridge's picture engine.
 *   pix_twin build/pix.bin force <model> <ref.bin>   teacher-forced: ref.bin = 8 words (u16), thumb flag (u8), pixels, [thumb]; writes Q12 scores (int32) to stdout
 *   pix_twin build/pix.bin paint <seed> <guide_q8> <inv_temp_q8 for the first pass> <n> [word ids...]   chain of all models; writes n pictures of 64x64 bytes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../pix/pix.h"
const uint8_t *pix_blob;
extern int pix_tau_h, pix_tau_b; extern long long pix_nact;
#ifdef PIX_W1PROBE
void (*pix_w1probe)(int, int, const int32_t *, int);
static FILE *w1f; static int w1m;
static void w1p(int l, int s, const int32_t *x, int D) { int32_t h[4] = { w1m, l, s, D }; fwrite(h, 4, 4, w1f); fwrite(x, 4, D, w1f); }
#endif
static void probe(int t, int j, const int32_t *g, const int32_t *m) { (void)t; (void)j; fwrite(g, 4, 16, stdout); fwrite(m, 4, 256, stdout); }
#include <math.h>
/* PPROB=1 (paint, with SHADOW=1): the model's probability of the colour flat / body-only cells copy (64 x 64 pass) */
static double pcs[8][6]; static int pcm, pcwant = 2;
static void pcprobe(int t, int j, const int32_t *g, const int32_t *m) {
    extern uint8_t pix_cellclass[], pix_copycol[];
    if (pcm != pcwant) return;
    int c = pix_copycol[t * (pcm == 2 ? 16 : 4) + j], k = pix_cellclass[t] & 3;
    for (int ti = 0; ti < 2; ti++) {
        double it = ti ? 366.0 / 256 : 1, lg[16], lm[16], mx = -1e30, z = 0, mz = 0, mxm = -1e30;
        for (int i = 0; i < 16; i++) { lg[i] = g[i] / 4096.0 * it; if (lg[i] > mx) mx = lg[i]; }
        for (int i = 0; i < 16; i++) z += exp(lg[i] - mx);
        for (int i = 0; i < 16; i++) { lm[i] = m[(c >> 4) * 16 + i] / 4096.0 * it; if (lm[i] > mxm) mxm = lm[i]; }
        for (int i = 0; i < 16; i++) mz += exp(lm[i] - mxm);
        double lp = lg[c >> 4] - mx - log(z) + lm[c & 15] - mxm - log(mz);
        pcs[k][ti * 2] += exp(lp); pcs[k][ti * 2 + 1] -= lp;
    }
    pcs[k][4] += 1;
}
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
    uint8_t *b = malloc(n); if (fread(b, 1, n, f) != (size_t)n) return 1; pix_blob = b;
    uint32_t rng = 12345;
    { extern int pix_flatskip; if (getenv("FLAT")) pix_flatskip = atoi(getenv("FLAT")); }
    { extern int pix_gcut16, pix_awin; if (getenv("GCUT")) pix_gcut16 = atoi(getenv("GCUT")); if (getenv("AWIN")) pix_awin = atoi(getenv("AWIN")); }
    { extern int pix_pcopy, pix_pcopynd; if (getenv("PCOPY")) pix_pcopy = atoi(getenv("PCOPY")); if (getenv("PCOPYND")) pix_pcopynd = atoi(getenv("PCOPYND")); }
    { extern int pix_flatrule, pix_flatnd; if (getenv("FLATRULE")) pix_flatrule = atoi(getenv("FLATRULE")); if (getenv("FLATND")) pix_flatnd = atoi(getenv("FLATND")); }
    { extern int pix_hwin[][PIX_MAX_NH]; extern unsigned pix_hdrop, pix_adrop; const char *e = getenv("HWIN");      /* HWIN=layer:head:window,... */
      if (getenv("HDROP")) pix_hdrop = (unsigned)strtoul(getenv("HDROP"), 0, 0);
      if (getenv("ADROP")) pix_adrop = (unsigned)strtoul(getenv("ADROP"), 0, 0);
      while (e && *e) { int a, h, w; if (sscanf(e, "%d:%d:%d", &a, &h, &w) == 3) pix_hwin[a][h] = w; e = strchr(e, ','); if (e) e++; } }
    { extern int pix_w1x, pix_w1x2; if (getenv("W1X")) pix_w1x = atoi(getenv("W1X")); if (getenv("W1X2")) pix_w1x2 = atoi(getenv("W1X2")); }     /* tests: early exit off (see pix.c) */
    { extern int pix_headskip, pix_hsnd, pix_shadow, pix_hsb; if (getenv("HSB")) pix_hsb = atoi(getenv("HSB")); { extern int pix_hsnd32; if (getenv("HSND32")) pix_hsnd32 = atoi(getenv("HSND32")); } if (getenv("HEADSKIP")) pix_headskip = atoi(getenv("HEADSKIP")); if (getenv("HSND")) pix_hsnd = atoi(getenv("HSND")); if (getenv("SHADOW")) pix_shadow = atoi(getenv("SHADOW")); }
    { extern int pix_w1xs0; if (getenv("W1XS0")) pix_w1xs0 = atoi(getenv("W1XS0")); }     /* W1XS0=0: no early exit with two streams */     /* tests: early exit off (see pix.c) */
    { extern int pix_g3; if (getenv("G3")) pix_g3 = atoi(getenv("G3")); }
    { extern int pix_g2b; if (getenv("G2B")) pix_g2b = atoi(getenv("G2B")); }     /* G2B=0: each guidance stream picks its own attention top 8 */
    if (getenv("TAU_B")) pix_tau_b = atoi(getenv("TAU_B"));
    if (getenv("TAUL")) { extern const int *pix_taul; static int tl[16]; const char *e = getenv("TAUL"); for (int i = 0; i < 16; i++) { tl[i] = *e ? atoi(e) : -1; while (*e && *e != ',') e++; if (*e) e++; } pix_taul = tl; }     /* TAUL=a,b,..: body thresholds per layer */
    if (getenv("TAU_H")) pix_tau_h = atoi(getenv("TAU_H"));
    if (!strcmp(argv[2], "force")) {
        int m = atoi(argv[3]);
        FILE *r = fopen(argv[4], "rb"); uint16_t words[8]; uint8_t img[4096], th[1024];
        int S = pix_info(m, MF_S), F = pix_info(m, MF_F);
        if (fread(words, 2, 8, r) != 8 || fread(img, 1, S * S, r) != (size_t)(S * S)) return 1;
        if (F && fread(th, 1, (S / F) * (S / F), r) != (size_t)((S / F) * (S / F))) return 1;
        pix_force = img; pix_probe = probe;
        { extern void (*pix_probe2)(int, int, const int32_t *, const int32_t *); if (getenv("FORCE_GUIDE")) { pix_probe = 0; pix_probe2 = probe; } }     /* guided scores, for the cells that run with guidance */
        if (pix_begin(m, words, th, getenv("FORCE_GUIDE") ? atoi(getenv("FORCE_GUIDE")) : 256, 256, &rng)) return 2;     /* FORCE_GUIDE: with guidance (tools/w1kl2.py) */
        while (pix_cell()) {}
        fprintf(stderr, "active %lld\n", pix_nact);
        if (getenv("CLASSOUT")) { extern uint8_t pix_cellclass[], pix_copycol[]; FILE *o = fopen(getenv("CLASSOUT"), "wb"); fwrite(pix_cellclass, 1, PIX_MAX_NC, o); fwrite(pix_copycol, 1, PIX_MAX_NC * PIX_MAX_NS, o); fclose(o); }
    } else {
        int base = !strcmp(argv[2], "paint3") ? 3 : 0;      /* paint3: the scenes base */
        rng = (uint32_t)strtoul(argv[3], 0, 0); int guide = atoi(argv[4]), invt = atoi(argv[5]), np = atoi(argv[6]);
        uint16_t words[8] = { 0 };
        for (int i = 7; i < argc && i < 15; i++) words[i - 7] = (uint16_t)atoi(argv[i]);
        static uint8_t th[1024];
        for (int p = 0; p < np; p++) {
            for (int m = 0; m < 3; m++) {
#ifdef PIX_W1PROBE
                if (getenv("W1DUMP")) { if (!w1f) w1f = fopen(getenv("W1DUMP"), "wb"); pix_w1probe = w1p; w1m = m ? m : base; }
#endif
                pcm = m; if (getenv("PPROB")) { pcwant = getenv("PCM") ? atoi(getenv("PCM")) : 2; pix_probe = m == pcwant ? pcprobe : 0; }
                if (pix_begin(m ? m : base, words, th, m ? 256 : guide, m ? 366 : invt, &rng)) return 2;
                while (pix_cell()) {}
                { extern int pix_nskip, pix_nhskip; extern long long pix_nact; static long long hsk; hsk += pix_nhskip; static long long sk[3], act[3], last; sk[m] += pix_nskip; act[m] += pix_nact - last; last = pix_nact; if (getenv("W1XSTAT") && p == np - 1 && m == 2) { extern long long pix_npcopy; fprintf(stderr, "cells painted from the thumbnail: %lld %lld (+%lld bodies only); pixels copied in running cells (PCOPY): %lld; active hidden units: %lld %lld %lld\n", sk[1], sk[2], hsk, pix_npcopy, act[0], act[1], act[2]); } }
                int S = pix_info(m, MF_S);
                if (m + 1 < 3) memcpy(th, pix_img, S * S); else fwrite(pix_img, 1, S * S, stdout);
                if (getenv("PIXDUMP")) { FILE *d = fopen(getenv("PIXDUMP"), m || p ? "ab" : "wb"); fwrite(pix_img, 1, S * S, d); fclose(d); }     /* all three passes' pictures */
            }
        }
        { extern int pix_lanemax; fprintf(stderr, "largest shared-table half-sum %d of 32767\n", pix_lanemax); }
        { extern long long pix_shared[2]; fprintf(stderr, "second MLP matrix: both streams in one pass %lld times, split %lld times\n", pix_shared[0], pix_shared[1]); }
    }
    if (getenv("PPROB")) for (int k = 0; k < 4; k++) fprintf(stderr, "class %d (%s): pixels %.0f  p(copy) model %.4f sampling %.4f   -ln p model %.4f sampling %.4f\n", k, k == 0 ? "model runs" : k == 1 ? "window rule only" : k == 2 ? "own rule only (body-only)" : "both rules", pcs[k][4], pcs[k][0] / pcs[k][4], pcs[k][2] / pcs[k][4], pcs[k][1] / pcs[k][4], pcs[k][3] / pcs[k][4]);
    { extern long long pix_w1x_look[4]; if (getenv("W1XSTAT")) for (int k = 0; k < 4; k += 2) fprintf(stderr, "first matrix, %s: %lld of %lld lookups (%.1f%% saved)\n", k ? "two streams" : "one stream", pix_w1x_look[k], pix_w1x_look[k + 1], 100.0 - 100.0 * pix_w1x_look[k] / (pix_w1x_look[k + 1] ? pix_w1x_look[k + 1] : 1)); }
    return 0;
}
