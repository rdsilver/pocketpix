/* Headless mGBA driver for the tests.
 *
 *   runner rom.gba STEP...
 *     f:N                 run N frames
 *     k:MASK              press keys (GBA key bits) for 3 frames, then release for 3
 *     hold:MASK           hold these keys down from now on (hold:0 lets go)
 *     until:ADDR:VAL:MAX  run until the 32-bit word at ADDR equals VAL (at most MAX frames)
 *     shot:FILE.ppm       write the screen
 *     dump:ADDR:LEN:FILE  write LEN bytes of memory
 *     frames              print how many frames have run
 *     loadsave:FILE       put FILE into the cartridge's save memory (use it as the first step; a missing file = no save yet)
 *     savetype:N          force the kind of save memory (first step): 0 none, 1 SRAM, 2 flash 64K, 3 flash 128K
 *     storesave:FILE      write the cartridge's save memory to FILE (nothing is written if the game never touched it)
 */
#include <mgba/core/core.h>
#include <mgba/core/log.h>
#include <mgba/internal/gba/gba.h>
#include <mgba-util/vfs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void quiet(struct mLogger *l, int c, enum mLogLevel lv, const char *f, va_list a) {
    (void)l; (void)c; (void)lv; (void)f; (void)a;
}
static struct mLogger logger = { .log = quiet };

static uint32_t rd32(struct mCore *core, uint32_t a) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) v |= (core->busRead8(core, a + i) & 0xFF) << (8 * i);
    return v;
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    mLogSetDefaultLogger(&logger);
    struct mCore *core = mCoreFind(argv[1]);
    if (!core) { fprintf(stderr, "not a ROM mGBA recognises\n"); return 1; }
    core->init(core);
    mCoreInitConfig(core, NULL);
    unsigned w, h;
    core->desiredVideoDimensions(core, &w, &h);
    color_t *buf = malloc(w * h * sizeof(color_t));
    core->setVideoBuffer(core, buf, w);
    if (!mCoreLoadFile(core, argv[1])) { fprintf(stderr, "load failed\n"); return 1; }
    core->reset(core);
    long frames = 0;
    for (int i = 2; i < argc; i++) {
        char *s = argv[i];
        if (!strncmp(s, "f:", 2)) {
            for (long n = atol(s + 2); n > 0; n--, frames++) core->runFrame(core);
        } else if (!strncmp(s, "k:", 2)) {
            core->setKeys(core, (uint32_t)strtoul(s + 2, 0, 0));
            for (int n = 0; n < 3; n++, frames++) core->runFrame(core);
            core->setKeys(core, 0);
            for (int n = 0; n < 3; n++, frames++) core->runFrame(core);
        } else if (!strncmp(s, "hold:", 5)) {
            core->setKeys(core, (uint32_t)strtoul(s + 5, 0, 0));
        } else if (!strncmp(s, "until:", 6)) {
            char *e;
            uint32_t addr = (uint32_t)strtoul(s + 6, &e, 0), val = (uint32_t)strtoul(e + 1, &e, 0);
            long max = atol(e + 1), n = 0;
            while (rd32(core, addr) != val && n < max) { core->runFrame(core); n++; frames++; }
            if (rd32(core, addr) != val) { fprintf(stderr, "until: timed out after %ld frames\n", n); return 2; }
        } else if (!strncmp(s, "shot:", 5)) {
            FILE *f = fopen(s + 5, "wb");
            fprintf(f, "P6\n%u %u\n255\n", w, h);
            for (unsigned p = 0; p < w * h; p++) {
                uint32_t c = buf[p];
                fputc(c & 0xFF, f); fputc((c >> 8) & 0xFF, f); fputc((c >> 16) & 0xFF, f);
            }
            fclose(f);
        } else if (!strncmp(s, "dump:", 5)) {
            char *e;
            uint32_t addr = (uint32_t)strtoul(s + 5, &e, 0), len = (uint32_t)strtoul(e + 1, &e, 0);
            FILE *f = fopen(e + 1, "wb");
            for (uint32_t a = 0; a < len; a++) fputc((int)(core->busRead8(core, addr + a) & 0xFF), f);
            fclose(f);
        } else if (!strncmp(s, "loadsave:", 9)) {
            FILE *f = fopen(s + 9, "rb");
            if (f) {
                static unsigned char sv[1 << 17];
                size_t n = fread(sv, 1, sizeof sv, f);
                fclose(f);
                if (n && !core->loadSave(core, VFileMemChunk(sv, n))) { fprintf(stderr, "loadsave failed\n"); return 1; }
            }
        } else if (!strncmp(s, "savetype:", 9)) {
            GBASavedataForceType(&((struct GBA *)core->board)->memory.savedata, atoi(s + 9));
        } else if (!strncmp(s, "storesave:", 10)) {
            struct GBASavedata *sd = &((struct GBA *)core->board)->memory.savedata;      /* GBA only; the core's savedataClone is unreliable here */
            size_t n = sd->data ? GBASavedataSize(sd) : 0;
            if (n) { FILE *f = fopen(s + 10, "wb"); fwrite(sd->data, 1, n, f); fclose(f); }
        } else if (!strcmp(s, "frames")) {
            printf("%ld\n", frames);
        } else { fprintf(stderr, "bad step %s\n", s); return 1; }
    }
    return 0;
}
