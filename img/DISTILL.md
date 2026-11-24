# Teaching Pocket Pix from clean pictures (trial run)

Idea: the cartridge models learned from messy photos. FLUX.1-schnell (Apache 2.0) draws clean, centred pictures of
each word; the three cartridge models (b16C objects base, u32M, u64M) are fine-tuned on them, shrunk to 64x64 in the
cartridge palette. Same model sizes, so the same speed on the GBA and in the browser.

Trial: 98 words (img/distill.py TRIAL) x 64 pictures = 6,272 pictures, fine-tuned from the night models in three ways:
a) gently, all weights (lr 2e-4, ternary matrices x2), with 7,692 general photos (empty prompts) mixed in against forgetting;
b) matrices frozen, only word tables / norms / gains (lr 5e-4); c) as a, without the photos.
Only these words are taught, so `score` also measures general photos with an empty prompt, and the sheets include
untaught words (sky, beach, mountain, street) and the empty prompt: "better" means the taught words improve while
those stay about the same.

## Running it (experimental)
Needs a machine with several large GPUs (the trial was sized for four RTX 5090-class cards, CUDA 12.8 or newer, and
150 GB of disk or more). Run the steps below from the repository root on that machine; they start from the shipped models in img/remote/night/.

No Hugging Face login: the weights come from an ungated copy (unsloth/FLUX.1-schnell), and every file is checked
against the official repository's fingerprints (img/flux_official.json, read from the Hugging Face API).

| Step | What | Time on 4x RTX 5090 (estimate) |
|---|---|---|
| bash img/jobs/x_setup.sh | packages, checks, FLUX download (about 34 GB) | 5-15 min |
| bash img/jobs/x_probe.sh | speed at 512 and 384 on one GPU, 48 pictures each, contact sheets | 3-5 min |
| bash img/jobs/x_gen.sh | the 6,272 pictures, one FLUX process per GPU | 25-40 min |
| bash img/jobs/x_train.sh | training files, three rounds of three fine-tunes, scores, old/new sheets, bundles | 30-50 min |
| bash img/jobs/x_train2.sh, x_train3.sh | stronger fine-tune recipes (30 passes, lr 5e-4) | |

`python3 -m img.distill photoval` (run by x_setup.sh) needs a held-out set of general photos in
img/data/photos64_test, made with `python3 -m img.data` (see img/data.py); it is not included here.
The weights from the trial run are not published: they were an experiment and the cartridge does not use them.

Then, on any machine: `PIX_MODELS=<folder with the new b16C/u32M/u64M + night b16O> PIX_OUT=<dir>/pix.bin PIX_LAYOUT=<dir>/pix_layout.h
python3 -m pix.export b16C u32M u64M b16O`. The new blob has its own layout (export drops dead hidden units, so the
offsets move): build a second twin against it, `gcc -O2 -Wall -I<dir> -Igen -o <dir>/pix_twin host/pix_twin.c pix/pix.c -lm`
(the -I<dir> first, so its pix_layout.h wins), and draw "car red" etc. with both twins. The engine's early-exit and
threshold settings were tuned on the night weights: compare with W1X=0 as well, and recalibrate (tools/w1cal.py,
pix/tausearch.py) before shipping new weights.

Full run later: `WORDS=all` (4,094 words) and about 50 pictures per word in x_gen.sh, about 12-14 hours of drawing,
200 GB of disk or more. The word list itself lacks many everyday nouns (no pizza, rabbit, frog...): worth widening
for a full retrain from scratch.
