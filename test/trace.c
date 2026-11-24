/* trace rom.gba N: run N instructions, print the last 40 PCs with mode and registers */
#include <mgba/core/core.h>
#include <mgba/core/log.h>
#include <mgba/internal/gba/gba.h>
#include <mgba/internal/arm/arm.h>
#include <stdio.h>
#include <stdlib.h>
static void quiet(struct mLogger *l, int c, enum mLogLevel lv, const char *f, va_list a) { (void)l; (void)c; (void)lv; (void)f; (void)a; }
static struct mLogger logger = { .log = quiet };
int main(int argc, char **argv) {
    mLogSetDefaultLogger(&logger);
    struct mCore *core = mCoreFind(argv[1]);
    core->init(core); mCoreInitConfig(core, NULL);
    unsigned w, hh; core->desiredVideoDimensions(core, &w, &hh);
    color_t *buf = malloc(w * hh * sizeof(color_t)); core->setVideoBuffer(core, buf, w);
    if (!mCoreLoadFile(core, argv[1])) return 1;
    core->reset(core);
    struct ARMCore *cpu = core->cpu;
    long N = atol(argv[2]); uint32_t ring[64]; int md[64];
    for (long i = 0; i < N; i++) {
        uint32_t pc = cpu->gprs[15] - (cpu->executionMode == MODE_THUMB ? 2 : 4);
        ring[i & 63] = pc; md[i & 63] = cpu->executionMode == MODE_THUMB;
        core->step(core);
    }
    for (long i = N - 40; i < N; i++) printf("%08x %c\n", ring[i & 63], md[i & 63] ? 'T' : 'A');
    for (int r = 0; r < 16; r++) printf("r%d=%08x ", r, cpu->gprs[r]); printf("\n");
    return 0;
}
