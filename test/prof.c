/* Per-instruction cycle profile of the test ROM in mGBA: prof rom.gba out.bin
 * out.bin: for IWRAM (0x03000000, 0x8000), VRAM (0x06010000, 0x8000), ROM (0x08000000, 0x10000): per word uint64 cycles, uint64 count */
#include <mgba/core/core.h>
#include <mgba/core/log.h>
#include <mgba/core/timing.h>
#include <mgba/internal/gba/gba.h>
#include <mgba/internal/arm/arm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void quiet(struct mLogger *l, int c, enum mLogLevel lv, const char *f, va_list a) { (void)l; (void)c; (void)lv; (void)f; (void)a; }
static struct mLogger logger = { .log = quiet };
#define NI 0x2000
#define NV 0x2000
#define NR 0x4000
static uint64_t h[(NI + NV + NR + 1) * 2];
int main(int argc, char **argv) {
    mLogSetDefaultLogger(&logger);
    struct mCore *core = mCoreFind(argv[1]);
    core->init(core); mCoreInitConfig(core, NULL);
    unsigned w, hh; core->desiredVideoDimensions(core, &w, &hh);
    color_t *buf = malloc(w * hh * sizeof(color_t)); core->setVideoBuffer(core, buf, w);
    if (!mCoreLoadFile(core, argv[1])) return 1;
    core->reset(core);
    struct GBA *gba = core->board; struct ARMCore *cpu = core->cpu;
    uint32_t *done = (uint32_t *)((uint8_t *)gba->memory.wram + 4);
    uint32_t minsp = 0x03007F00;
    uint64_t n = 0; int32_t last = mTimingCurrentTime(&gba->timing);
    while (*done != 1) {
        uint32_t pc = cpu->gprs[15] - (cpu->executionMode == MODE_THUMB ? 2 : 4);
        core->step(core);
        int32_t now = mTimingCurrentTime(&gba->timing); uint32_t d = (uint32_t)(now - last); last = now;
        uint32_t ix;
        if (pc >> 24 == 3 && (pc & 0xFFFFFF) < NI * 4) ix = (pc & 0xFFFFFF) >> 2;
        else if (pc >= 0x06010000 && pc < 0x06010000 + NV * 4) ix = NI + ((pc - 0x06010000) >> 2);
        else if (pc >= 0x08000000 && pc < 0x08000000 + NR * 4) ix = NI + NV + ((pc - 0x08000000) >> 2);
        else ix = NI + NV + NR;
        h[2 * ix] += d; h[2 * ix + 1]++;
        if ((uint32_t)cpu->gprs[13] < minsp && (uint32_t)cpu->gprs[13] > 0x03007000) minsp = cpu->gprs[13];
        n++;
    }
    fprintf(stderr, "%llu instructions; deepest stack %u bytes below 0x03007F00\n", (unsigned long long)n, 0x03007F00 - minsp);
    FILE *f = fopen(argv[2], "wb"); fwrite(h, 1, sizeof h, f); fclose(f);
    return 0;
}
