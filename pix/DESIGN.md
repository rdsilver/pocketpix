# Pocket Pix engine notes (22 Nov 2026)

(Development log, kept as written. Files under build/ other than pocketpix.gba, pix.bin and pix_test48.u8 are local
working files: reference twins, checksums and older blobs that are not part of this repository.)

Game: build/pocketpix.gba (pix/build_game.py). Models in the blob (pix/export.py b16C u32M u64M b16O): objects base,
32 enlarger, 64 enlarger, scenes base (SELECT in the game). Engine pix/pix.c + pix/kernels.s, native twin
build/pix_twin (host/pix_twin.c), emulator test pix/test_gba.py (picture must equal the twin's; prints cycles).
Quality check against PyTorch: python3 -m pix.check b16C u32M u64M build/pix_test48.u8 (bits/pixel, KL).

## Speed history (mGBA, "car red", guidance 3)
6:49 first working version -> 3:09 now (sunset 2:55, no prompt 3:05). Same pictures as the twin throughout.
What gave it, largest first:
- W2 and the head's second matrix run transposed through the 81-entry tables (tern_scatter + tern_unbuild): only
  active hidden units are walked, one table add per 4 output rows. Exact.
- Structurally dead hidden units removed at export (u32M: a third of them). Exact.
- Both guidance streams share one table build and one scan (two 16-bit sums per entry, inputs lose 5 bits), and
  guidance stops after 11/16 of the cells (pix_gcut16).
- Cells whose thumbnail window is flat within FLAT steps (u32M 1, u64M 2 of 32 per channel) are painted straight
  from the thumbnail; later cells' attention skips them. About 15% / 27% of cells on the test pictures.
- Hidden values up to TAU are dropped (base 16, u32M 64, u64M 32; head 32).
- Unrolled tern_scan, attention kept in fast RAM (noinline!), rotation tables once per cell, 32-bit divisions and an
  assembly square root in the norms, head's first matrix folded into its tables (C > 1), thumbnail rows per block.
Tried and dropped: one-pass "stronger prompt" instead of guidance (clearly worse pictures), early exit in W1 rows
(does not trigger), Thumb for ROM code (no gain).
Not done (each a few seconds, exact): ReLU fused into the W1 scan, tern_build with stmia, 8-bit key tables,
attention glue in assembly, caching the head's neighbour rows.
Measured quality cost of the approximations (48 pictures, teacher-forced): KL 0.0007 / 0.0020 / 0.0020 nats,
bits per pixel within 0.2% of the float model.

## Second round (22 Nov 2026, evening): 3:09 -> about 2:45 ("car red"); sunset 1:52, no prompt 2:42
All still byte-identical to the twin in mGBA (pix/test_gba.py, five prompt/base combinations).
- Kernels from the idea panel: tern_build81 with 9 values in registers and stmia, 16-entry build for sign-only keys
  (build16), unrolled scatter8 / scatter12, unrolled tern_unbuild. Exact.
- Slow RAM at one wait state (0x04000800 = 0x0E000020, with a pattern test at boot that undoes it on failure).
- pix.c's non-IWRAM code runs from spare video memory (pix/gba.ld region vram2 at 0x06012000, copied by pix/crt0.s);
  word buffers live there and in vram0 (0x06009600) / vram1 (0x06010000, shared with the top of the word arena, so
  only for passes with more than one pixel per cell). Byte data must stay out of video memory.
  A bug this caught: clearing ycell during the guided base pass overwrote stream 1's prompt sums.
- Enlargers' attention looks back 64 cells only (AWIN; bits per pixel +0.03% / +0.16%; needs the row above: 17 is
  still fine, 8 is not). The base keeps full attention.
- Shared-table sums use XSH = 6: largest half-sum seen 7,206 of 32,767 (was 16,253 at XSH = 5 over 114 pictures).
IWRAM is full (a few hundred bytes free); the profiling build needs mixv outside it.
Ideas checked and rejected by the panel or by measurement: Thumb anywhere, 8-bit key tables (build cost eats the
gain), lowering the guidance cut below 11/16 (KL doubles), punching holes in the guided stream.

## Third round (whole-pipeline audit): 2:47 -> 2:22 ("car red"), no prompt 2:42 -> 2:21. All exact
Twin output byte-identical to the round-two twin (paint, paint3, force); emulator pictures equal the twin's.
- Fast RAM was the blocker, so the head's buffers now reuse the body's (facc = x, u = xn, yacc = sc, z and
  sparse_out's result = proj, the query = the spare end of proj): 2.8 KB freed and spent on the kernels below.
  pix_begin and the general (non sign-only) key path run from ROM (.romcode); stream 1's x/xn sit in vram2 and its
  proj/mixv in vram0 instead of slow RAM. Fast RAM is full again (about 40 bytes in the profiling build).
- 32-bit division (libgcc __udivsi3) is linked into fast RAM (gba.ld); division by 3 in the norms is a multiply.
- Head rows per pixel: axpy8_2 / sum16_3 / sum16_2w / sum16_4u / head_u sum several ROM rows per pass; the cell's
  own yin is added once per cell (ycell starts from it); the thumbnail block rows are added by head_u; the last
  pixel of a cell no longer updates sums nobody reads.
- Attention glue: topsel (bias add and top 8 in one pass, walking j downwards so the bias row is read forwards;
  skipped cells are left out instead of entering with a huge negative score: their weight was always 0), mix8
  (values a word at a time), scale_rs, key_k (key gain + rotation + sign packing fused, signs only), key rows that
  are zero or have zero gain are not scanned (kseg_prep / krows), unrolled scans for 4- and 6-word rows.
- relu_sq decides with one compare per input (in >= in0, worked out from tau, the shift and the multiplier), two
  inputs per load; relu_sq_hi reads stream 0 straight out of the packed rows and leaves stream 1 in place (no
  unpack pass for W1). Two-stream rows are unpacked in place; the packed inputs are built in fast RAM (sc).
- Kernel loops: mixer_k (coefficients by ldmia, constants by shifts), norm_k / scale_add / sumsq_k / pack2 / unpack2
  two or four per iteration, the shorter operand as the multiplier, zero32 with 12-word stores, the first step of a
  scan row starts the sum.
Looked at and left: compacting skipped cells out of the key scan (bias gather costs what the shorter scan saves),
8-bit key tables again (no room for a second key format), a direct (table-free) head output for few active units
(65 of 256 are active on average: the tables win), skipping all-zero step groups (1-2% of W1's groups).

## Merge of the MLP auditors' patches (key "mlp"): 142.5 -> 139.1 s ("car red"), no prompt 140.6 -> 139.7 s, scenes base 143.6 -> 139.9 s. All exact
Twin output byte-identical (paint with and without prompt, paint3); emulator pictures equal the twin's (five runs).
Measured one at a time with full mGBA runs (car / no prompt):
- Second MLP matrix shared by both guidance streams (mlp_out2; -2.6 s / 0): the units active in either stream (about
  138 instead of 122 + 120) go through the tables once; each table entry holds stream 0's sum above bit k and the sum
  of (stream 1 - stream 0) below it, k chosen per call from the sums of the values (which bound every output), so the
  result is exact; scale_add2 takes the two apart. About one call in eight does not fit one word (187 of 1408 on the
  test picture) and runs the same units once per stream (packk with k = 0 makes stream 1's values). relu_sq_hi is gone.
- tern_unbuild multiplies by the matrix scale and adds into the stream itself (m = 0: plain sums, for mlp_out2): no
  acc buffer and no scale_add pass for W2 and the head's second matrix (-0.45 / -0.55 s).
- ReLU fused into the W1 row kernel (tern_scan_relu; -0.4 / -0.35 s): the 8- and 12-word loops of tern_scan end in
  "movcc pc, r3" (plain scans run with carry set and pay one cycle per row; the masks became immediates to free
  r12 for the loop address). The threshold costs no register: tern_build81b adds -in0 to every entry of the first
  table group (each row takes exactly one entry of each group), so "kept" is the sign of the row sum, or of each
  half with two streams (-in0 * 65537; the two-stream tail does relu_sq2's job: lists, sums). The per-layer
  constants (ScanR) are worked out once per picture in pix_begin, which also removed a division per MLP call.
  Assumptions, checked in pix_begin: in0 <= 2^20 for one stream (row sums must stay above -2^31 + in0) and
  in0 <= 4096 for two (half-sums minus in0 must fit 16 bits: the twin's "largest shared-table half-sum" plus in0,
  here 4970 + 77, must stay below 32768). Other row lengths or thresholds take the unfused path (scan + relu_sq, or
  the C relu_sq2 in ROM), as does the twin.
Memory: stream 0's yin moved to the video memory that passes with more than one pixel per cell leave free (vram1 is
now full), mixv to vram2 (no measurable cost), rows / mlp_out2 / routed experts / the generic scan loop out of fast
RAM (rows runs from video memory, the others from ROM). Free now: fast RAM 56 bytes in the test build (240 in the
game), vram2 296 bytes (test build). Stack use unchanged (pix_cell's frame is 16 bytes smaller).
Tried and left: sparse_out and rows in ROM (+0.17 s without a prompt: keep them in RAM), the sum of squares for
the head's norm inside the fold-back (about 0.08 s, no room), a separate unpacking tail for attention rows (costs
what unpack2 costs).

## Merge of the attention auditors' patches (key "attention"): 139.08 -> 136.71 s ("car red"), no prompt 139.74 -> 137.67 s, scenes base 139.93 -> 137.56 s. All exact
Twin output byte-identical (paint with and without prompt, paint3); emulator pictures equal the twin's (six runs).
Measured one at a time with full mGBA runs (car / no prompt):
- mixk (-0.64 / -0.56 s): one head's value mix in one pass, the 8-bit values a word at a time and four sums in
  registers; values stored with 7 or more bits dropped have the shift folded into the weight (list a), the others
  are rounded one by one (list b). No accumulator buffer, no zeroing and no final rounding pass (mix8 is gone); the
  two lists live in sc.
- keytop (-0.55 / -0.44 s, on top of the pair insertion below): key scan, distance bias and top 8 in one pass for
  keys of 2 or 3 words; the scores are never stored. Biases are loaded two rows at a time (one ROM access per two
  rows: the row that gets its bias first starts its sum with it, the other adds it at the end), skipped cells are
  looked up only for rows that would be inserted. Other key lengths: keysel (scan + the C topsel) from ROM. The
  unrolled 2- and 3-word loops of tern_scan and the topsel kernel are gone (that is what paid for keytop's space).
- Top 8 as (score, row) pairs with a stop pair (-0.21 / -0.18 s): the insertion walk is ldmdb / stmia per pair
  with no bounds test. A score of 2^31 - 1 (equal to the stop) takes a counted walk, so the result is the C
  function's for every input.
- att_val_k (-0.39 / -0.35 s): the largest magnitude from the largest and smallest input (the scaling is
  monotonic), then scale, shift, clamp and store in one pass.
- qbuild16 (-0.35 / -0.31 s): the query is scaled in registers on its way into the 16-entry tables (build16 and the
  scale_rs pass are gone from this path; the general table goes through qbuild in ROM).
- rope_k (-0.23 / -0.21 s): the rotation in assembly, adding up each head's sum of squares for the query norm while
  the values are in registers (alone it gave -0.14 s and freed 112 bytes of fast RAM).
Measured and reverted: the donors' own 32-bit division that skips known-zero quotient bits (+0.07 s: libgcc's in
fast RAM is faster), stream 1's projections and mixed values in xn / proj instead of vram0 (-0.06 s), pack2 four at
a time (-0.06 s; only the base runs two streams, the enlargers never do).
Not ported: skipping empty Q rows (0 to 6 rows of 128 / 192 per layer: about 0.05 s), the shared 12/8/4 scan chain
(saves space but costs 2 cycles a row), unpack2 four at a time (about 0.04 s).
Free now: fast RAM 96 bytes in the test build (256 in the game). Stack: att_mix's frame is 56 bytes larger (220).
Profile note: "qnorm" also holds the top 8 and mix of the three heads before it; "keyscan" now includes the top 8.

## Merge of the glue / memory-map / head auditors' patches (key "glue"): 136.71 -> 133.74 s ("car red"), no prompt 137.67 -> 134.71 s, scenes base 137.56 -> 134.57 s. All exact
Twin output byte-identical (paint with and without prompt, paint3); emulator pictures equal the twin's (six runs,
among them other prompts, guidance 2 and the scenes base without a prompt). Measured one at a time with full mGBA
runs (car / no prompt); found with the per-instruction cycle profile (a PC histogram from mGBA stepped one
instruction at a time). What mGBA charges, measured there: a single 32-bit read from ROM 8 cycles, a block read 4
per word (pairs are as good as fours); 32-bit reads and writes in video memory one cycle more per word; code in video
memory one cycle more per instruction.
- norm_k with the high-word rounding (-0.98 / -0.91 s): rs(in * inv, 28) is the high word of in * (inv << 4) + 2^31
  and the gain product fits a plain 32-bit mul, four values per round. Needs r above 2^13 (inv below 2^27) and gains
  of at most GMAX = 37,000 (checked per model in pix_begin: the largest here is 21,741; |t| stays below 56,800, so
  t * g + 2^11 fits 32 bits). All but 12 of 78,061 norms on the test pictures qualify; the rest take the C loop.
- Glue loops that ran from video memory are kernels in fast RAM (-0.66 / -0.61 s): copy32 (prompt sums into x, yin
  into ycell, zsave), rs4_k (the input's shift by 4), mulrs_k (gain or multiplier rows: z = hin * gn, the 16 scores
  times their multipliers, also for members), zero32 for cellsum; with one pixel per cell head_u gets cellsum (all
  zeros there) as its block rows.
- 8-bit ROM rows two words per load (-0.49 / -0.54 s): axpy8_2 (two values of 8 per turn; the end of the input is on
  the stack to free a register) and axpy8. Both now want n to be a multiple of 8 (pix_begin refuses a model whose D
  or DH is not a multiple of 16; the table kernels never worked for such a model anyway).
- The head's per-cell sums in the idle tail of tbl (-0.38 / -0.42 s): the head builds tables for DH inputs only, so
  words 2592.. of tbl are free fast RAM while it runs: cellsum, ycell, tsum and stream 0's hin moved there from video
  memory (hin became a pointer in Stream; stream 1's is in vram0: vram1 is the arena's top for the base models, which
  is what a first attempt overwrote). ytsum (1024 words) does not fit and stays in vram1.
- div_small shifts once for the power of two instead of looping over a 64-bit shift, with the count found in at
  most three steps (-0.21 / -0.20 s together with a tighter loop in isqrt64's 32-bit case).
- zero32 clears 96 words per turn with the count test at the bottom (-0.15 / -0.16 s).
- The pixel's thumbnail block and the block's first pixel come from tables made in pix_begin (-0.11 / -0.12 s):
  no signed divisions (libgcc, in ROM) per pixel.
Fast RAM: the region is 0x7900 long now (gba.ld; 1.5 KB of stack reserve instead of 1.75; the deepest use measured
in the test ROM is 796 bytes, unchanged). Free: 16 bytes in the test build, 176 in the game.
Measured and left out: the donors' mixer_k (coefficients three at a time, history stored early: the same cycle
count as the current one), the head's rows five at a time (fsum_k / ysum_k: about 0.2 s for 220 bytes of fast RAM;
the passes are bound by registers, and int16 rows are already block reads), the block's five int16 rows in two
passes (-0.06 s), stream 1's buffers in the tail of tbl for the base model (about 0.15 s with a prompt only, needs
the buffers placed at run time; the attention merge measured the same for proj1 / mixv1).
What is left in this area: about 1 s of C glue spread thinly over pix_cell / body / head / norm in video memory (no
loop above 0.05 s), the mixer state in video memory (0.4 s; 12 to 18 KB, no faster place), sumsq_k (1.6 s, bound by
the multiplier), zero32 (2.1 s, one cycle per word).

## Gallery / save memory (23 Nov 2026)
Up to 15 finished pictures are kept, with their prompts, in the cartridge's 64 KB flash save memory (0x0E000000,
byte access only; all the code is in pix/game.c and runs from ROM, its data is 968 bytes of slow RAM). The ROM
carries the marker "FLASH512_V131" so emulators and flash carts choose that save type.
Layout: 16 sectors of 4 KB. Sectors 0..14 hold one picture each (4096 palette indices). Sector 15 is a diary: an
8-byte heading "PPIXGAL\1" in the first 32 bytes, then up to 127 notes of 32 bytes:
  0 0xA5 | 1 slot | 2 used (1 kept, 0 deleted) | 3 model (0 objects, 1 scenes) | 4 number of words | 5..7 zero |
  8..11 sequence number (larger = kept later) | 12..27 eight word ids | 28..29 check of the picture | 30..31 check
  of bytes 0..29 (both checks: 32-bit FNV-1a folded to 16 bits).
The last good note about a slot wins; a note whose check, fields or picture check fail is ignored, so blank,
zeroed or random save memory reads as an empty gallery. Keeping a picture erases and writes its sector, verifies
it, and only then adds a note; deleting adds a note. The diary sector itself is erased only when all 127 notes
are used (about once per 112 keeps or deletes): it is then rewritten with one note per kept picture, the heading
last. A note that does not read back is tried once in the next place and then reported as "save failed": a bad
write never leads to an erase.
Power cuts: during a keep or a delete the other pictures and notes are not touched (a half-replaced slot fails
its picture check, a half-written note its own). During the diary rewrite the listing can be lost: the pictures
are still in their sectors but nothing points to them (there is no spare sector in 15 + 1).
Chips: the identity is read at boot. SST (0xD4BF), Macronix (0x1CC2) and Panasonic (0x1B32) 64K chips are used
(sector erase 0x80 + 0x30, byte program 0xA0, completion by read-back with a time limit counted in screen lines:
4.7 ms a byte, 3 s an erase; nothing can hang). Atmel (0x3D1F: 128-byte page writes, no sector erase), 128K
chips, plain SRAM and "no answer" turn saving off: the gallery says "Can't save on this cartridge", the done
screen "can't save", and the rest of the game works. Flash carts that fake a save by patching Nintendo's library
will also land there.
Tested in mGBA only (build/runner with loadsave: / storesave: / savetype: / hold:), never on real hardware:
a painted picture kept, the emulator restarted, and the viewer's pixels and flash sector equal to the engine's
picture byte for byte; delete and restart; 15 kept and the oldest replaced (made-up pictures from
`python3 pix/build_game.py test`, and real paints on the shipped ROM); no save file, blank, zeros, random,
random with a good heading, a short file; save type forced to none / SRAM / 128K; a save snapshot after every
frame of a diary rewrite booted again. Not testable there: real erase and program times, the stale-RAM case of
the chip probe after a warm restart (mGBA clears RAM), weak cells, the Atmel refusal. Also note that crt0.s
leaves the Nintendo logo in the header blank, so the ROM as built starts in emulators and through flash-cart
loaders but not from a real console's own boot check.

### Gallery: now 7 slots in 32 KB (battery RAM first, flash as a fallback)
The save marker is `SRAM_V113` and the layout is 7 picture sectors + the diary in the first 32 KB. At boot the game
first checks whether a plain byte write sticks (battery RAM: emulators' default, flash carts); only if not does it
send flash commands (64K chips as before; two 128K chip IDs are accepted and only their first half used, untested
on real chips). Checked in mGBA with the save type left automatic and forced to SRAM, flash 64K and flash 128K:
keep, restart, view, fill, replace oldest, delete; and starting from all-zero, all-0xFF, random and old 64 KB save
files. `pix/test_gallery.py` still describes the 15-slot flash layout and needs updating before it is used again.

## Night round 1 (23 Nov 2026, key "r1"): 133.7 -> 123.8 s ("car red"), no prompt 134.7 -> 123.3 s, scenes base 134.6 -> 124.5 s
Six explorers (first matrix, attention, 64x64 head, machine level, quality-neutral, wildcard), merged one change at a
time with full mGBA runs. Two kinds of change, kept apart:
- EXACT (133.7 -> 130.6 / 134.7 -> 131.6 / 134.6 -> 131.4 s): twin byte-identical to build/pix_twin_ref.
- APPROXIMATIONS, two, each behind one constant (-> 123.8 / 123.3 / 124.5 s on the same pictures): W1X and PIX_TAUL in
  pix.c. The default twin's pictures therefore differ from pix_twin_ref's; build/pix_twin_ref_r1 is the new twin, and
  `W1X=0 TAU_B=-2 build/pix_twin ...` is still byte-identical to pix_twin_ref (paint with and without a prompt,
  paint3), as is a ROM built with `python3 pix/test_gba.py w1x=0 cflags=-DPIX_TAUL=0` under `TAU_B=-2` (130.8 s).
(The blob's layout changed: two layer fields more, LF_W1X / LF_W1X2 / LF_TAU. build/pix_twin_ref was compiled for the
old layout and must be given the old blob, build/pix.bin; rebuild the twin after every export.)
An approximation changes which picture gets sampled, and the picture alone moves the time by several seconds (flat
cells). So like-for-like figures come from `python3 pix/test_gba.py force=1`: every pass paints the exact engine's
picture (the twin with W1X=0 TAU_B=-2 FLATRULE=0 writes it; "painted the given picture: True"). Free-running, the
three test prompts take 122.9 / 122.7 / 126.3 s now; emulator picture equal to the twin's in seven runs (the three,
words=958, words=935,2 guide_q8=512, scenes base without a prompt, and all switches off).

### Exact changes, in the order merged (car / no prompt where measured)
- Tables are not cleared before the transposed second matrix (PIX_KEEPTBL; -1.45 s): folding back a table that
  tern_build made from x gives exactly 54 * x, and a constant added to a whole group folds to 0. When the scatter
  starts, the tables still hold the build of a vector that is still in memory (the body: xn, or the packed pair in
  sc; the head: z of the pixel before, hbase), so the scatter goes on top and tern_unbuild (fifth argument base)
  takes 54 * base off: 25 cycles per group instead of 81. All sums are modulo 2^32, so the bits are the same.
  INVARIANT: between the first matrix's table build and the layer's sparse_out nothing may write the tables or the
  base vector; in the head, tbl[0 .. DH/4*81) and z stay untouched from tern_build(tbl, z) of one pixel to the
  sparse_out of the next. Code that rebuilds tbl in between must pass base 0 / set hbase = 0.
- Machine level (-1.3 s together): sc (1 KB of words) lives in sprite memory (OAM, 0x07000000), which mGBA charges
  like fast RAM (words only: byte writes are ignored there; the game never turns sprites on, a game that does must
  clear OAM after painting; PIX_SC_OAM 0 puts sc in video memory, +0.43 s). The freed fast RAM went to the fused
  first-matrix kernels having their own row loops (the plain 8/12-word loops lost their "movcc pc"), whose shift and
  rounding term pix_begin writes into nine instructions (relu_sites; self-modifying code in fast RAM). isqrt64's
  32-bit case is the unrolled three-instructions-per-bit routine (-0.17 s). tern_unbuild loads its three stack words
  with one ldm (-0.09 s). Stream 1's buffers run in the idle tail of tbl for the first-pass models (PIX_S1_FAST;
  XS / XNS / PJ / MV are set by pix_begin; -0.19 s with a prompt).
- The head's second matrix (8 KB) and its 16 group rows are copied into the word arena's unused video memory when a
  pass leaves room (64x64 pass: both; -0.2 s). vram1 moved to the arena's top 5 KB (gba.ld 0x06010C00, VRAM1_WORDS).
  -DPIX_NO_H2COPY turns it off.
- keytop's early row exit for two-word sign-only keys (the first-pass models; -0.16 s): a row whose first word +
  bias is below (8th best - the largest the second word can add, read off the query table) skips its second word
  (47% of rows). keytop takes its constants from the block pix_ktp and is called through kscan (keysel4 in ROM for
  other key shapes or gains of 2^20 and more).

### Approximation 1: early exit in the first MLP matrix, one stream (W1X in pix.c; 0 = off). -5.1 / -7.0 / -5.2 s
A row whose partial sum is below a calibrated limit at one of seven check points (28, 24, ... 4 groups of four
inputs to go) is dropped: its hidden unit counts as inactive. Limits: seven int32 per layer in the blob (LF_W1X, from
pix/w1exit.json, made by tools/w1cal.py from twin dumps: the most lookups saved for a set share of lost hidden
energy, b16C 0.3%, b16O 0.2%, u32M 0.5%, u64M 0.6%). About a fifth of the one-stream lookups go. On the GBA the
limits are table biases (pix_begin: w1xb; tern_build81x builds the tables in one call), so a check is "adds, bmi":
kernel tern_scan_x (12-word rows; 8-word rows enter at its ninth word). Two streams (guided cells) are not touched:
tern_scan_relu is now the two-stream kernel only, with its own 8-word loop (other lengths: the unfused path).
With W1X 0 the same kernel runs with limits that never trigger (and `test_gba.py w1x=0` assembles it without the
checks): that build is 0.2 to 0.5 s slower than the exact tree before this change (130.8 against 130.6 s).
The two-stream exit the explorer also built (W1X2) saved 0.6 s and its quality could not be certified: the twin still
has it for study (env W1X2=1), the GBA kernel was not merged (#error).
To recalibrate: gcc -O2 -Igen -DPIX_W1PROBE -o build/pix_twin_probe host/pix_twin.c pix/pix.c; run it with W1X=0
W1DUMP=build/d1.bin ... on a handful of prompts (paint and paint3); python3 tools/w1cal.py
b16C=0.003,b16C2=0.0015,u32M=0.005,u64M=0.006,b16O=0.002,b16O2=0.0015 build/d*.bin; export; rebuild the twin.

### Approximation 2: per-layer body thresholds (PIX_TAUL in pix.c; 0 = the model's plain threshold). -1.9 / -1.8 / -1.9 s
Later layers stand much larger thresholds than early ones (found by pix/tausearch.py, one layer at a time):
export.py TAUL -> layer field LF_TAU; pix_begin works out p0 and in0 per layer. Lists: b16C 16,32,16,16,16,32,32,64;
u32M 64,64,64,128,256,256,256,256; u64M 32,64,128,288,192,128; b16O 16,32,16,16,16,32,32,64. The explorer's lists
ended in 384,384 (u32M) and 96 (b16O); with both approximations together those went over the KL allowance (u32M
+0.00031), so they were taken back.

### Quality of both together (python3 -m pix.check b16C u32M u64M b16O build/pix_test48.u8; allowed +0.0015 / +0.0003)
| model | bits/pixel before -> after | KL before -> after | W1X alone |
|---|---|---|---|
| b16C | 2.4993 -> 2.5002 (+0.0009) | 0.00074 -> 0.00100 (+0.00026) | 2.4988 / 0.00088 |
| u32M | 1.2332 -> 1.2336 (+0.0004) | 0.00241 -> 0.00268 (+0.00027) | 1.2334 / 0.00253 |
| u64M | 0.9174 -> 0.9181 (+0.0007) | 0.00262 -> 0.00285 (+0.00023) | 0.9181 / 0.00270 |
| b16O | 2.5277 -> 2.5280 (+0.0003) | 0.00058 -> 0.00083 (+0.00025) | 2.5281 / 0.00072 |
The KL allowance is nearly used up: no further approximation of this kind fits without taking one of these back.
Free sampling, 288 pictures per setting (six prompt / base combinations, build/drift.sh): cells painted flat in the
32 / 64 pass: exact 8,234 / 15,948; W1X 7,315 / 15,436; thresholds 7,349 / 15,516; both 8,027 / 16,297 (batches of
48 differ by 20% either way): no drift towards flatter pictures. Contact sheet tools/sheet.py (exact above, both
below, same seeds): different samples, the same kind of picture.

### Measured and NOT switched on
- New flat-cell rule for the 64x64 pass (PIX_FLAT_OWN in pix.c, default 0; twin: FLATRULE=1): copy a cell when its own
  four thumbnail blocks are one colour and at most 5 of the eight blocks touching it differ. -3.4 / -2.7 / -7.0 s in
  the explorer's tree, and by its own measures better than the old rule (model's probability of the copied colour
  0.949 -> 0.978; whole-pass KL to the PyTorch model 0.0221 -> 0.0132 nats/pixel), but pix.check's u64M line goes
  0.9174 / 0.00262 -> 1.0302 / 0.00457 and on the pixels both rules sample KL rises 0.00231 -> 0.00358, because
  skipped cells drop out of later cells' attention and mixer history. That fails the rule (+0.0003), so it is off.
  It is the largest thing left on the table, if the owner accepts a different yardstick for flat cells.
- "Flat spots" (single pixels or whole cells copied when the painted neighbours agree; 1.0 to 1.2 s; +0.00055 KL on
  the other pixels): not ported.
Side finding from two explorers: more than half of the enlargers' measured KL comes from skipped flat cells missing
from later cells' attention (the existing rule costs about +0.0013 KL on the other pixels), not from arithmetic.

### Memory
Fast RAM free: 116 bytes in the test build, 208 in the game (python3 pix/build_game.py); stack reserve 1.5 KB,
deepest use in the test ROM 800 bytes (test/prof.py). vram2 has about 2 KB free; OAM is taken by sc. New section
.vram.text (gba.ld) for assembly that may run from video memory (rs4_k). sparse_out runs from video memory.
Moved out of fast RAM and back while merging (all exact): the model's scalars, rcs and members() are in fast RAM
again (there was room after the kernels were merged; -0.07 s).

### Tools added
pix/test_gba.py: force=1, w1x=0, cflags=...; test/prof.c + test/prof.py + test/profdis.py (per-instruction cycle
profile and deepest stack from mGBA stepped one instruction at a time; build/prof), bench/ (what mGBA charges per
access type and region), tools/w1cal.py w1eval.py w1flat.py w1kl2.py w1lib.py sheet.py, pix/qc.py (pix.check's
numbers in 9 s after building a cache), pix/tausearch.py, pix/copystat.py, pix/sweep.py (teacher-forced sweeps with
the twin hooks HWIN / HDROP / ADROP; it has a scratch path hard-coded), pix/check.py ONLY=name. Twin environment:
W1X, W1X2, TAU_B=-2, TAUL=a,b,.., FLATRULE, FLATND, W1XSTAT, PIXDUMP, FORCE_GUIDE, HWIN, HDROP, ADROP.

### mGBA against hardware
mGBA 0.10 charges a ROM block read 4n cycles; by the documented bus timing the console pays 4n + 4. With about 21
million such reads per picture, real hardware should be about 5 s slower than every figure here. "Pairs are as good
as fours" is an emulator artefact: on hardware fewer, longer ROM bursts are better. Nothing here exploits the gap.

### Tried without gain in this round (measured or counted; do not repeat without a new angle)
First matrix: an exact early exit with true bounds and inputs sorted at export (6% of lookups, 0.6 s, made
redundant by the calibrated limits); sorting inputs for the approximate exit (the per-call gather eats it); limits
that follow each call's inputs; 9 or 10 check points; shared sub-rows / prefix trees (matrices 64-71% dense, random);
243-entry tables (38 KB against 15.5 KB of fast RAM; 8 cycles a lookup in slow RAM); "napping" units (row sums are
not predictable from cell to cell: KL +0.007 and up); pruning weak or rare units (every unit fires in more than 1%
of calls); one loop for plain and exit scans (a cycle per check for every attention row).
Attention: values on demand (94-98% are used later); dropping a layer (KL +0.003..0.10) or a head (+0.0002..0.02);
windows for the first pass (AWIN 128: KL +0.0006) and smaller ones for the enlargers (48: +0.0003..0.0005); per-head
windows (0.7 + 0.5 s for the whole allowance); the early row exit for three-word keys; 256-entry key tables; Q/K/V
sharing a scan; qbuild16 in 3 instructions (net zero).
Head / 64x64: no exact shortcut for near-certain pixels (1.0 bit of entropy per sampled pixel, top colour above
0.99 in 14%); loosening the old flat tolerance (copied colour's probability 0.76-0.86); head threshold 64 (0.4 s
for KL +0.0001..0.0002); borrowing keys/values for skipped cells (worse); neighbour-row caches (0.3-0.4 s for MBs of
ROM); slot rows folded into input rows (0.27 s, exact, needs a blob change: the range check is done, 17,513 and
19,828 fit 16 bits: still open).
Machine: slow RAM without wait (0.5 s slower in mGBA); cartridge first-access wait 2 (slower in mGBA, unsafe on
cartridges); zero32 from ROM (+0.07 s, frees 132 bytes); the fast-RAM mirror to save the "and" per step (break-even);
6-bit key groups (0.3 s, no room); Thumb glue; DMA / BIOS helpers; tern_build81x without a call per block of limits
(0.06 s).
Model level: dropping an MLP layer (KL +0.007..0.25); late layers at 16x the threshold (+0.0006..0.0023); guidance
streams sharing the head's first matrix (0.6 s, not exact, not measurable with pix.check). Not tried: the
empty-prompt stream reusing the prompt stream's top-8 attention picks (about 1.5 s; needs a guided teacher-forced
measure).

## Night round 2 (23 Nov 2026, key "r2"): 123.8 -> 115.2 s ("car red", like-for-like), no prompt 123.3 -> 117.5 s, scenes base 124.5 -> 115.5 s
Six explorers again (first matrix, attention, 64x64 head, machine level, quality-neutral, wild card), merged into
merged_r2 one change at a time, biggest first, with full mGBA runs after each. The exact changes went in first, the
approximations last. Each approximation has its own constant.
Free-running, the three test prompts now take 113.9 / 116.4 / 83.7 s. Free-running times move by seconds with the
sampled picture, because of flat cells: the scenes figure is low because its new picture has many flat cells. The
emulator picture equals the twin's in every run: the three test prompts, words=958, words=935,2 guide_q8=512, the
scenes base without a prompt, and a ROM with every approximation off (126.1 s; it was 130.8 after round 1).

References and checks (./m.sh runs all three; the blob changed, so rebuild with `python3 -m pix.export b16C u32M u64M
b16O` and then the twin, which now needs -lm):
- `env G2B=0 W1XS0=0 PCOPY=0 build/pix_twin ...` is byte-identical to build/pix_twin_ref_r1 (build/ref_r1.md5: five
  paint / paint3 commands, guidance 2 and 3).
- With W1X=0 TAU_B=-2 added it is byte-identical to the ORIGINAL exact reference build/pix_twin_ref. Give that twin
  the pre-round-1 blob, build/pre_night/pix.bin: build/pix.bin is now the round-1
  blob, and pix_twin_ref writes nothing with it.
- The default twin (all approximations on) is build/pix_twin_ref_r2; its checksums are in build/ref_r2.md5.

### EXACT changes, in the order merged (car red, free-running = forced for exact changes; measured in the merged tree)
| change (explorer) | car red | note |
|---|---|---|
| start (merged_r1) | 122.9 s | forced 123.8 / 123.3 / 124.5 |
| T3 step words for every ternary matrix + PIX_V1SQUEEZE (r2_w1) | 119.4 s (-3.5) | blob change |
| Thumb outside fast RAM, hcopy, Thumb rs4_k, udivr reciprocal division (r2_machine) | 118.9 s (-0.5) | all four together |
| slot rows folded into IN_Y (r2_head64) | 118.7 s (-0.2) | blob change (MF_YQ bit 8) |
| zero Q rows not scanned (r2_attention) | 118.8 s (+0.1: +0.4M cycles) | NOT kept |
Exact total: free 118.7 / 118.3 / 121.9 s; forced 119.5 / 118.8 / 120.3 s.
Merging notes: the machine explorer's `.type x, %function` rule was applied to the new T3 kernels as well (scatter32,
scatter48, tern_scan8, sum16_1w). The machine explorer's hcopy pointers were combined with the first-matrix
explorer's v1buf / ytsum pointers in pix_begin. The slot fold's export was rewritten to emit T3 words for H2.

### APPROXIMATIONS of this round (each switchable; added to round 1's W1X and PIX_TAUL)
| switch (pix.c; twin env to turn it off) | car / no prompt / scenes, forced | quality |
|---|---|---|
| PIX_G2B (G2B=0) + PIX_W1XS0 (W1XS0=0), guided cells only | 119.5 -> 116.8 / 118.8 -> 118.8 / 120.3 -> 117.6 | see below |
| PIX_PCOPY (PCOPY=0, PCOPYND=n), 32/64 passes | 116.8 -> 115.2 / 118.8 -> 117.5 / 117.6 -> 115.5 | see below |
| PIX_HEADSKIP (HEADSKIP=1 to turn it on): NOT switched on | 115.2 -> 114.8 / 117.5 -> 117.0 / 115.5 -> 114.2 | |
- G2B and W1XS0 count as one package: W1XS0 is only valid on top of G2B. Measured apart in the merged tree (car,
  forced): G2B -1.3 s, W1XS0 -1.4 s. pix.check does not reach guided cells, so they were measured again in the merged
  tree with tools/gkl.py: guided, teacher-forced, 48 pictures x 4 prompts, against PyTorch guided the same way.
  - b16C, old -> new: bits 2.7734 -> 2.7328, KL 0.03299 -> 0.02452 (-0.0085 +- 0.0004). G2B alone: 0.02409.
  - b16O: bits 2.7464 -> 2.6963, KL 0.02437 -> 0.02102 (-0.0034 +- 0.0004).
  - These are the explorer's figures exactly: with the same switches, the merged twin's pictures are byte-identical
    to the explorer's twin.
  In the T3 tree, the W1XS0 checks sit in tern_scan_relu's 32-group loop (.Lrelu11, check points after groups
  4, 8, ... 28 as for one stream, exit stub .Lya). The C copy is scan_exit3, using t3pos.
- PCOPY: pix.check, all approximations on, gives b16C 2.5002 / 0.00100 and b16O 2.5280 / 0.00083, unchanged from
  round 1. u32M reads 1.2617 / 0.00273 and u64M 1.1367 / 0.00347. With PCOPY=0 the u32M / u64M lines are round 1's
  exactly (1.2336 / 0.00268, 0.9181 / 0.00285).
  - Why the raw lines move: copied pixels leave the average, the same convention that already excludes flat cells.
    Every pixel that still runs the head keeps byte-identical scores; the explorer checked this, and the merged
    twin equals the explorer's twin.
  - So PCOPY is judged, like the flat rule, by the model's probability of the copied colour: u64M 0.954 against
    0.928 for the accepted flat rule, u32M 0.962 against 0.932. Free sampling showed no drift towards flatter
    pictures. The wild-card notes below have the tables.
  - The no-prompt saving (1.3 s) is under the 1.5 s bar; the three-prompt average is 1.7 s.
- HEADSKIP is still in the tree but switched off (PIX_HEADSKIP 0). On top of PCOPY it gives only 0.4 / 0.5 / 1.3 s
  (forced), because PCOPY already takes the good pixels of those cells. That is under the bar, and its extra pixels
  are the riskier ones (drawn neighbours of another colour).
- Contact sheet build/sheet_r2.png: top half round 1's behaviour (G2B=0 W1XS0=0 PCOPY=0), bottom half the new default,
  same seeds. Different samples, the same kind of pictures.
- Dropped while merging: the wild card's PIX_HSKIP (whole cells with body but no head: HEADSKIP does the same with
  less work, and both are off) and the quality explorer's PIX_FLATPROT context cells (off, and they conflicted with
  the own_flat refactor). Their measurements are kept in the notes below. The head64 explorer's probe was renamed
  from env PCOPY to PPROB in the twin, because PCOPY is now the switch.

### Memory (merged tree)
- Fast RAM free: 480 bytes in the test build, 648 in the game (python3 pix/build_game.py builds).
- vram2 free: about 0.8 KB in the test build (with the profiling hooks), 1.8 KB in the game.
- Deepest stack in the test ROM: 832 bytes, which leaves 704 of the 1.5 KB reserve (test/prof.py).
- ROM: 25.2 MiB test ROM. The reciprocal table adds 128 KB, and the T3 blob is 0.5 MB larger.

### Profile now (car red, 114.0 s; test/prof.py)
| kernel | time | share |
|---|---|---|
| tern_scan | 55.2 s | 48% |
| scatter32 + scatter48 | 15.6 s | |
| tern_unbuild | 6.8 s | |
| tern_build81 | 6.2 s | |
| keytop | 4.5 s | |
| mixer_k | 3.6 s | |
| norm_k | 3.2 s | |
Fast-RAM kernels take 111.4 of the 114 s; video memory 2.2 s; ROM 0.4 s.

### Attention explorer (not merged: its one exact change measured +0.4M cycles in the merged tree)
The change was to skip all-zero query rows (kseg[1]). It saved 0.04 s in the round-1 tree, but in the T3 tree it adds
work. Its findings, measured in the round-1 tree (do not repeat):
- PIX_AWIN as a quality trade. The 64-cell window is a large part of the enlargers' KL.
  - u32M / u64M KL against the original baseline: window 64 +0.00027 / +0.00023; 96 -0.00004 / -0.00010; 128
    -0.00015 / -0.00021; all cells -0.00020 / -0.00038.
  - Cost: window 96 +0.8 s, 128 +1.4 s, all cells +2.5 s.
  - Spending the freed KL on looser W1X limits or higher enlarger thresholds returned no more than the window
    cost, and no combination reached 1.5 s net.
- Skipping the last cell's K/V is impossible (a cell's query scores its own key).
- Shorter keys for heads with dead pairs: about 0.05 s.
- Key loads from slow RAM cost 0.77 s, and no faster memory is free.
- Batching two cells like the two guidance streams is not feasible (every layer is chained to the previous cell).
- K/Q/V/O projections run at the row kernel's floor.

### Explorer notes (details, written in each explorer's own tree; figures there are against the round-1 tree)
#### Explorer: first matrix (key "r2_w1"): 122.9 -> 119.4 s ("car red"), no prompt 122.7 -> 119.0 s, scenes base 126.3 -> 122.6 s. All exact
Like-for-like (test_gba.py force=1): 123.8 / 123.3 / 124.5 -> 120.2 / 119.5 / 120.9 s. With both approximations off
(w1x=0 cflags=-DPIX_TAUL=0, TAU_B=-2): 130.8 -> 126.7 s. The twin's pictures are byte-identical to build/pix_twin_ref_r1
(five paint commands, build/ref_r1.md5; ./m.sh checks them) and, with W1X=0 TAU_B=-2, to build/pix_twin_ref; pix.check
prints the round-1 numbers unchanged; emulator pictures equal the twin's (seven runs).

##### T3: three groups per word of steps, two of them read without masking (-3.3 s)
The row loop was at "5 cycles a lookup + 1 for the ROM byte": and, ldr with write-back, add. The `and` is gone for
two lookups in three. Every ternary matrix in the blob (Q, K, V, O, W1, HIN, H1, OUTG, OUTM, and the transposed W2
and H2) is now stored as 32-bit words `A << 24 | B << 11 | C` (export.py step_words; rows of G groups take
(G + 2) / 3 words, a short FIRST word when G is not a multiple of 3). With p at the entry of the group before:
- `ldr [p, w, lsr #9]`: the first group's entry, B words on. The shift leaves A's bits in address bits 15 to 22, and
  fast RAM repeats every 32 KB all the way to 0x03FFFFFF, so the read lands on the same entry: no mask, no write-back.
  (Bits 9, 10 and 19 to 23 of a word are zero; A < 256 keeps the address under 0x03800000.)
- `ldr [p, w, lsr #22]!`: the second group's entry, A words on from the same p (A is a two-table step, 82..242).
- `and` + `ldr [p, C, lsl #2]!`: the third group, as before.
13 cycles per word instead of 15 for the same three lookups, 4 instead of 3 for the ROM: 17 against 18, on every
row scan (48% of the run) and, with read / add / write, on the scatter of the transposed matrices (14%).
Kernels: tern_scan (unrolled for 48, 32, 24 and 16 groups; others: scan_c in ROM), tern_scan_x (48 and 32 groups, the
last ten words are one macro; same check points: after groups 20, 24, ... 44 of 48, which no longer sit on word
borders and need not), tern_scan_relu (32 groups), scatter32 / scatter48. Step words are loaded 6 + 5 + 5 per row
(r3 and r12 became step registers: in0 is on the stack, the kept-row code is a subroutine): three ROM bursts per
48-group row as before, so the gain should hold on hardware (4n + 4 per burst) as it does in mGBA (4n).
The keys made at run time are still step bytes (keytop; tern_scan8 in ROM for other key shapes).
REQUIREMENT: the tables these kernels read must be in fast RAM (0x03000000 region: tbl is), nowhere else.
Measured steps: row scans alone 122.9 -> 120.1 s; + scatter 119.6 s; + the copy below 119.4 s.
(scatter48 loads four words at a time, four bursts a column where the byte format took three: on hardware its gain
is 12 cycles a column instead of 16.)
Fast RAM: the new loops are shorter. Free now: 472 bytes in the test build, 600 in the game (was 116 / 208);
deepest stack 808 bytes (was 800). Room for what earlier rounds left out for lack of it (the head's rows five at a
time, the sum of squares inside the fold-back, ...).

##### The head's second matrix still fits in video memory for the 64 x 64 pass (PIX_V1SQUEEZE; -0.23 s)
As T3 the matrix is 11 KB instead of 8 and no longer fitted under vram1. vram1 is now one array of five rows
(v1buf: four thumbnail-block sums and stream 0's yin); a pass with several pixels per cell and one stream moves
three of them into stream 1's idle buffers (yin1, proj1, x1 + xn1) and keeps only the top 2 KB of the arena for
vram1, so the copy (and OUTG's) fits again. -DPIX_V1SQUEEZE=0 turns it off.

##### Looked at for the first matrix and not done (computed from dumps of the real inputs, study/*.py)
- Mean-centred early exit (tables built from x - mean, each row's sum started at its mean: the partial sum then is
  the expected final sum): the inputs' means are only 0.2-0.3 of their spread (row means 0.2-0.45 of the row sums'
  spread); 1 to 1.5 points more lookups saved at equal lost energy, about the 5 cycles a row the start value costs,
  and it generalised worse to other prompts (lost energy 0.3 -> 0.6%, b16O 1.8%). Dropped.
- A low-rank predictor of the row sums (drop rows from a sketch of the input): the inputs are not low-rank; the top 8
  directions hold 30-45% of the second moment in the enlargers (16: 50-60%). Dropped.
- Recalibrating the exit limits against the per-layer thresholds (the limits in the blob were fitted with the plain
  threshold): 1 to 1.7 points more lookups (0.2-0.3 s), not worth a new quality round.
- 5-trit tables once more with the new address forms (a 13-bit B field reaches any table): 8 tables of 243 + 22 of
  81 fit the 15.5 KB for D = 128 only: 30 lookups instead of 32, minus the larger build, minus the kept-table trick
  for the second matrix (its tables would no longer coincide): under 0.3 s. 16-bit entries: the build (strh) costs
  what the lookups save. Chunked 5-trit passes for D = 192 (three table loads per call): about -8% of the first
  matrix before the exit bookkeeping and the lost kept-table trick; not built.
- More lookups without a mask: a word has room for one top field (A), one field read through the 32 KB repeat (B,
  which needs the 5 bits above it clear) and one masked byte; lsl / ror forms put A into the address's top byte.
  Two unmasked lookups per word and nothing else is 12 cycles for two groups: no better than before.
- Loading the exit kernel's words in smaller blocks (rows that leave early have loaded up to four words they never
  use: 0.56 s in mGBA): {6,7} + {8,9,10} instead of five would give 0.26 s in mGBA and lose about 0.07 s on
  hardware (one more burst for the rows that stay). Left: it would exploit the emulator's burst timing.
- The same address forms for the keys (keytop): with compact 16-entry query tables a key word could hold four groups
  (one top field, one through the repeat, two masked 5-bit fields): 22 cycles per four groups instead of 24, about
  0.3 s, no extra key memory. T3 keys as they are would need a third more key memory (the guided first pass does not
  have it). Not built: the attention explorer's ground.
- Kept rows (the squared ReLU, 0.75 s one stream + 0.8 s two): a 32-bit multiply when the values allow saves 3
  cycles of 36 after its range test: 0.07 s. Not done.
#### Explorer: machine level (key "r2_machine"): 122.9 -> 122.4 s ("car red"), no prompt 122.7 -> 122.2 s, scenes 126.3 -> 125.7 s. All EXACT
Cycle totals 2,062.5M -> 2,053.1M / 2,059.4M -> 2,049.9M / 2,118.4M -> 2,108.7M (-0.56 / -0.57 / -0.58 s). Twin
byte-identical to build/pix_twin_ref_r1 (nine paint / paint3 commands, guidance 1, 2 and 3; with W1X=0 TAU_B=-2 equal
to the merged tree's twin); emulator picture equal to the twin's in six runs (the three test prompts, words=958,
words=935,2 guide_q8=512, scenes base without a prompt). Fast RAM free: 104 bytes in the test build (was 16), 272 in
the game; deepest stack 832 bytes (was 800; 1.5 KB reserve).
What mGBA charges for code (bench/code.py, cycles per loop turn: 8 ALU ops / 4 fast-RAM loads / 2 muls):
  IWRAM ARM 12.3 / 16.3 / 10.3 | VRAM ARM 24.3 / 24.3 / 16.3 | VRAM Thumb 12.3 / 16.3 / 8.4 | ROM ARM 50 / 34 / 26 |
  ROM Thumb 26 / 18 / 14. So Thumb code in video memory runs at fast-RAM speed per instruction (16-bit bus, one
  fetch per instruction), and ROM Thumb that mostly loads from fast RAM is nearly as fast (the prefetcher hides it).
Kept, in the order measured (car red, cycles):
- Thumb for everything outside fast RAM (pix.c: `#pragma GCC target("thumb")` on the GBA; IWRAM_CODE stays ARM;
  ARM_GLUE = ARM in video memory for div_small, whose 64-bit shifts are slow in Thumb; -DPIX_ARM_GLUE turns it all
  back to ARM): -3.6M (-0.22 s). The C glue in video memory went 2.16 -> 1.98 s, and video memory code is 2 KB
  smaller. kernels.s now marks every global with `.type x, %function` (needed for interworking: without it a Thumb
  caller jumps into ARM code with bl and crashes).
- The head's per-pixel ROM words (final norm gains MF_HF, group and member multipliers MF_OUTG_M / MF_OUTM_M, 1.6 KB)
  copied to video memory by pix_begin (hcopy; -DPIX_NO_HCOPY turns it off): -1.4M (-0.08 s).
- rs4_k as Thumb in video memory, four values per turn: -0.9M (-0.05 s).
- Division-free 40-by-24-bit quotients (udivr in kernels.s, C copy in pix.c for the twin): div_shl (every norm's
  2^40 / r, every query norm's (c << 20) / r), the attention weights' 2^31 / sum and relu_in0's division per head
  call. r is normalised to [2^23, 2^24), the ROM table pix_rcp (32768 words, generated by .rept in kernels.s) gives
  2^55 / rn to 2^-16, the quotient comes from (A >> 9) * y, then one correction from the remainder and a final +-1.
  Domain: A <= 2^40, r in [64, 2^24), quotient below 2^31; other cases keep the division. About 75 cycles against two
  libgcc divisions of about 110 each. Tested against the division for every r in [64, 2^24) with four A each (2^40
  and the largest A with a quotient below 2^31 among them) and 10^8 random (A, r): never more than one +-1 step.
  libgcc's __udivsi3 left fast RAM (gba.ld; 244 bytes; it now runs about 70 times a picture). -DPIX_DIVSHL_OLD goes
  back to the divisions (then slower than before, since the division is in ROM now). -3.5M (-0.21 s) in all.
Measured and left out:
- att_mix as Thumb in video memory: +3.4M (+0.2 s) for 1.3 KB of fast RAM; nothing found that would earn that back.
- A C version of the reciprocal (gcc's 64-bit code, ARM in video memory): 208 cycles a call, no faster than the two
  divisions. Only the hand-written fast-RAM routine pays.
- Reciprocal from a 256-entry table plus one Newton step: 93 cycles (the 32768-word table saves the Newton step:
  -1.2M cycles more).
Looked at and not done (numbers from the profile / mGBA's source, src/gba/memory.c):
- The scan's step bytes: ROM ldm costs 4 cycles a word in mGBA (16 per 16 steps of 6 cycles = 17% of tern_scan, 10 s
  in all). Video memory would be 10 per 4 words, fast RAM 6, but the weights are read once per cell (no reuse to pay
  for a copy) and DMA from ROM costs more per word than ldm. Denser step coding needs more than one instruction to
  extract (steps take 161 values; fields that need no `and` -- the top byte, or a field above two zero bits used with
  lsr -- give at most two steps per word: break-even). WAITCNT is already at the fastest legal ROM timing (3/1).
- Weights as code (immediate offsets in ldr): 8 ROM bytes per step instead of 1, and ROM ARM costs 4-5 cycles an
  instruction: far slower, and would not fit in 32 MB.
- Keys (EWRAM, ldmdb 10 / 14 cycles for 2 / 3 words, about 0.75 s in keytop) would save about 0.3 s in video memory
  and 0.45 s in fast RAM, but need 32-64 KB (base) or, with a doubled ring buffer for the enlargers' window of 64
  cells, 18-24 KB: no room in either. The mixer state (12-18 KB in video memory, about 0.37 s over fast RAM) the same.
- mixer_k operand order (coefficients reach 15 bits, states 17: both early-terminate at m = 2; swapping gains about
  0.05 cycle a multiply), 16-bit mixer coefficients or norm gains (the unpacking costs what the ROM read saves),
  sumsq_k with 32-bit mla (needs |x| below 46,341, not guaranteed), unbuild (its ALU count, 152 per group, is at the
  158-operation minimum of the 81 -> 27 -> 9 -> 3 -> 1 fold, and the nine ldm of 9 words are the floor for 81 words).
- Sprite palette RAM (512 bytes, unused by the game) and OAM: OAM already holds sc; palette RAM has video memory's
  timing, so it adds nothing over the remaining 1.2 KB of vram2.
- The 0.21 s spent in Thumb->ARM veneers (about 340K calls a picture): a hand-made long call costs the same.
mGBA vs hardware: on the console, video memory also stalls while the screen is drawn (mGBA 0.10 does not model it);
Thumb halves the glue's fetches from video memory, so it should help at least as much there. Tools: bench/code.py
(the table above), test/trace.c (build/trace rom N: last PCs and registers after N instructions, found the
interworking crash), pix/test_gba.py pcflags=... (flags for pix.c only).
#### Explorer: 64x64 pass / head (key "r2_head64"): 122.9 -> 120.9 s ("car red"), no prompt 122.7 -> 121.0 s, scenes 126.3 -> 124.9 s
Free-running figures, emulator picture equal to the twin's in every run. Two changes, kept apart:

##### EXACT: slot rows folded into the IN_Y rows (export.py + pix.c; -0.16 / -0.18 / -0.17 s)
The head's per-pixel Y sum added SLOT_Y[j] for every pixel. export.py now adds SLOT_Y[j + 1] - SLOT_Y[j] (as int16,
after rounding, so the sums are the same to the bit; largest entry 19,828) to every IN_Y row of slot j, and pix_cell
starts ycell from yin + SLOT_Y[0]: a pixel adds one int16 row of HH fewer. sum16_2w is replaced by sum16_1w (acc +=
w + a), which is smaller (fast RAM: 104 bytes free in the test build, 232 in the game). The blob says so in bit 8 of
MF_YQ; pix_begin refuses an enlarger whose blob lacks it (an old blob gives no picture instead of a wrong one), so
build/pix.bin must be re-exported with this export.py (python3 -m pix.export b16C u32M u64M b16O). Measured: u64M
-2.7 to -3.1 M cycles, u32M -0.4 M. Twin with HEADSKIP=0 (and new blob) byte-identical to build/pix_twin_ref_r1 (with
the round-1 blob) for paint / paint without prompt / paint3; W1X=0 TAU_B=-2 HEADSKIP=0 equals pix_twin_ref_r1 under
W1X=0 TAU_B=-2; ROM with all switches off (w1x=0 cflags=-DPIX_TAUL=0 hs=0, TAU_B=-2) 130.7 s, picture identical.
(build/pix_twin_ref does not run on build/pix.bin any more: that blob is the round-1 layout.)

##### APPROXIMATION (free sampling only): body-only cells in the 64x64 pass (PIX_HEADSKIP in pix.c; 0 = off; twin HEADSKIP=0; test_gba.py hs=0)
A cell that the window rule does not paint flat but whose own four thumbnail blocks are one colour and at most
PIX_HS_ND = 3 of the eight blocks touching them differ (the "own rule" of PIX_FLAT_OWN, stricter) runs only what later
cells read: layers 0..L-2 in full and the last layer's keys and values (att_kv; a last mixer layer would run its
mixer), then takes its pixels from the thumbnail like a flat cell. The last layer's query / scan / mix / output, its
MLP, the final norm, the head's input rows and first matrix and all 16 head runs are left out: 0.094 s per such cell
(1.58 M cycles, measured: 20 cells on "car red" = 31.6 M cycles). Unlike a skipped cell it stays in later cells'
attention (skipped[] is not set), so every other pixel is computed exactly as before given the same picture.
How often: 20 / 16 / 12 cells on the three test prompts; 23 per picture on average over 240 free-sampled pictures
(five prompt / base settings, 48 each: 23.6, 19.4, 25.2, 26.0, 21.4), i.e. about 2.2 s per picture on average.
Like-for-like (force=1, the exact engine's picture): car 123.6 -> 122.9, no prompt 123.1 -> 121.9, scenes 124.3 ->
121.9 s (the forced pictures have 8 / 13 / 25 such cells).
Quality, three measures:
- pix.check: rows of scores the twin computes with and without it are identical to the bit on every pixel both
  evaluate (tools/hscheck.py: 123,664 of 196,608 pixels; same bits/pixel 1.0357 and KL 0.00317 on that set). The
  plain pix.check line changes only because the copied pixels drop out of the mean (u64M "skipped 27.0% -> 37.1%",
  bits/pixel float 0.9165 -> 1.0344 integer 0.9181 -> 1.0357, KL 0.00285 -> 0.00317: the float model's own figure moves
  just as much, so it is the choice of pixels, not the engine). b16C / u32M / b16O lines unchanged.
- The yardstick earlier rounds used for flat cells, the model's probability of the copied colour. Teacher-forced on
  the 48 test pictures (tools/hsstat.py): window-rule cells (copied today) mean p 0.9275, -ln p 0.119; body-only cells
  p 0.940, -ln p 0.098, 10.1% of pixels. Free sampling, 72 pictures in three batches (twin SHADOW=1 PCOPY=1,
  tools/pcsum.py; p of the model, sampling temperature 1): window-rule cells p 0.939 / 0.925 / 0.887, body-only cells
  0.941 / 0.948 / 0.929 (-ln p 0.080 / 0.100 / 0.152 against 0.074 / 0.065 / 0.091). So these cells are copied at
  least as faithfully as the ones the existing rule copies, in every batch. PIX_HS_ND = 3 is the largest that holds
  this in all four comparisons (4 fails one batch, 5 fails the teacher-forced one: p 0.909). Side finding: the
  window rule's cells that the own rule would NOT accept are the weak ones (p 0.79-0.83, against 0.96-0.98 for
  cells both rules accept).
- Contact sheet build/sheet_hs.png (tools/sheet_hs.py; top HEADSKIP=0, bottom default, same seeds): the same kind
  of pictures. The decision uses the thumbnail only (not painted pixels) and this is the last pass, so nothing can
  drift towards flatter pictures.
Not certified by pix.check's KL (it cannot see copied pixels at all); it is a flat-cell rule like the existing one and
should be judged as one. Stack: deepest use 848 bytes (was 800; body is always_inline now, used by pix_cell and by
cell_kv in ROM).

##### Measured in this round and left out
- The same tier for the 32x32 pass (one block per cell; twin HSND32=n: at most n window blocks of another colour):
  0-3 cells per picture at n = 1, about 12 at n = 2, and the copied colour's p only 0.86 (worse than the window rule).
- Requiring the painted pixels bordering the cell to have the cell's colour (twin HSB=1): p 0.975-0.98 teacher-forced
  but only 3 cells on the test prompts (0.3 s).
- PIX_HS_ND 2 (more conservative): p 0.954 teacher-forced, but 11 / 7 / 6 cells on the test prompts (about 1 s).
- Head internals, counted with the per-instruction profile: 66 of 256 head units are active per pixel; the scatter
  is 7 cycles per (unit, group) (and, ldr, add, str), the fold-back about 300 cycles per group, both at their floor
  for this table scheme. Restricting the second matrix to the chosen group's rows is impossible (the norm needs all
  128 outputs before any score); computing the scores as (G x W2) a needs the norm of h = h0 + W2 a, whose a'Qa term
  costs more than the scatter (and is not exact). Packing two pixels into the 16-bit halves of one scatter (as the
  two guidance streams do) does not work: pixel j + 1's units depend on pixel j's sampled colour.
- Folding the slot rows into the 8-bit IN rows as well (facc): not exact (int8 rows with per-row scales).
Twin hooks added: HEADSKIP, HSND, HSB, HSND32, SHADOW (run the model on painted cells, changing nothing), CLASSOUT
(force: per-cell class and copied colours), PCOPY=1 [PCM=model] (free sampling: model's p of the copied colour).
#### Explorer: quality-neutral explorer (key "r2_quality-neutral"): guided cells. 123.8 -> 121.0 s ("car red", like-for-like)
Two (B) approximations, both only in the guided cells of the first pass (two streams), each behind one constant.
pix.check cannot see them (it runs one stream), so they were measured with tools/gkl.py: teacher-forced with guidance
on 48 pictures x 4 prompts, the twin's guided scores against the PyTorch model guided the same way (KL in nats,
+- standard error of the paired difference). Like-for-like times: `test_gba.py force=1` (EXACT now also sets G2B=0).
| setting | b16C guide 3: bits / KL | b16O guide 3 | b16C guide 2 | car / scenes (forced) |
|---|---|---|---|---|
| before (G2B=0 W1XS0=0) | 2.7734 / 0.03299 | 2.7464 / 0.02437 | 2.6501 / 0.01379 | 123.8 / 124.5 s |
| PIX_G2B | 2.7369 / 0.02409 (-0.0089 +- 0.0004) | 2.7092 / 0.01909 (-0.0053) | 2.6380 / 0.01044 (-0.0034) | 122.1 / 122.9 s |
| + PIX_W1XS0 | 2.7328 / 0.02452 (+0.0004 on G2B) | 2.6963 / 0.02102 (+0.0019 on G2B) | 2.6366 / 0.01061 | 121.0 / 121.8 s |
| PIX_W1XS0 alone | 2.7678 / 0.03329 (+0.0003 +- 0.00015) | 2.7306 / 0.02509 (+0.0007) | 2.6479 / 0.01390 | |
(PyTorch's own guided bits/pixel on these cells: b16C 2.7626, b16O 2.7400.) Without a prompt nothing changes (122.8 s).
- PIX_G2B (pix.c; twin G2B=0 turns it off): in a guided cell the empty-prompt stream uses the prompt stream's top-8
  attention picks AND weights, with its own values (att_mix2, in vram2). Stream 1 needs no keys (no key_k, no key
  storage), no query table, no key scan. -1.7 s (base pass 38.2 -> 36.6 s). The engine's guided KL to the model drops by
  a quarter: separate picks in the two streams differ on near-ties of the sign-key scores, and guidance multiplies
  every such difference by 3; with shared picks it cancels. (Tried: shared picks with stream 1's own scores: KL +0.097,
  much worse; stream 1 keeping only the hidden units stream 0 keeps (twin G3=1): +0.0013 on top of G2B, dropped.)
- PIX_W1XS0 (twin W1XS0=0): guided cells take the first matrix's early exit too, decided by stream 0's half of the
  packed row sum against the one-stream limits scaled by 2^-6 (tables made by tern_build81x from the packed inputs,
  w1xb2 in pix_begin; tern_scan_relu's 8-word loop got the seven "adds, bmi" checks: +36 bytes of fast RAM). 19% of the
  two-stream lookups go. -1.1 s (base 36.6 -> 35.4). On its own it is at the KL allowance for b16C and over it for b16O
  (+0.0007); together with G2B both models stay far below the old engine. Drop it if each change must pass alone.
- Free sampling, first pass (tools/freeflat.py, 96 pictures "car red", PyTorch sampling guided the engine's way):
  equal neighbours PyTorch 0.265, old 0.272, G2B 0.281, G2B+W1XS0 0.264 (+-0.006); flat cells the 32 pass would skip
  13.1 / 15.8 / 16.6 / 13.8 (+-1.3). Contact sheet build/sheet_r2.png (old above, new below): the same kind of pictures.
- Memory: w1xb, w1xb2 and scanr moved from vram2 to slow RAM (no measurable cost) to make room for att_mix2.
  Free: fast RAM 136 bytes (test build), 272 (game); vram2 about 0.7 KB (test build).
Free-running now: car 120.6 s, no prompt 122.8 s, scenes 89.8 s (pictures differ from before, so these move by
seconds with the flat cells; emulator pictures equal the twin's in all three and with w1x=0).

##### Measured, built, switched OFF: context cells with the own-blocks flat rule (PIX_FLAT_OWN=1 + PIX_FLATPROT=1)
Most of the enlargers' KL is the flat rule's (u32M 0.0016 of 0.0027; AWIN 0.0005), and it comes from the cells right
before the next computed one: a flat cell whose next cell is not flat runs as a "context cell" (body up to its last
stored keys / values or mixer state, no head, pixels from the thumbnail; skipped[] stays 0). Measured on the pixels
both settings compute (tools/qc2.py; pix.check's averages move with the set of skipped pixels): u64M old rule ->
own rule + context cells: KL 0.00337 -> 0.00310 (+-0.0003), bits +0.0004; copied pixels 27.0% -> 35.7%, the model's
probability of the copied colour 0.928 -> 0.947 (tools/pcopy.py). Times (forced): car -0.8 s, no prompt -0.9 s, scenes
-4.3 s. Off because car is under the 1.5 s bar and pix.check's raw u64M line changes (1.0313 bits: other pixels).
Same idea for u32M: no gain (tolerance 2 + protection copies colours the model gives only 0.87).
Other numbers (common pixels, u32M / u64M KL change): protecting the next cell with the old rule -0.0008 / -0.0007
(costs 2.8% / 5% of cells), also the cell below -0.0012 / -0.0013; AWIN 128: -0.00015 / -0.00021, AWIN off
-0.0002 / -0.0004 (would cost about 1.6 s of key scans).
#### Explorer: wild card (key "r2_wildcard")
One change kept, an approximation behind one constant: PIX_PCOPY in pix.c (twin: PCOPY=0 turns it off, PCOPYND=n).
`PCOPY=0 build/pix_twin ...` is byte-identical to build/pix_twin_ref_r1 (paint with and without a prompt, paint3;
build/cmp.sh runs the three), and with W1X=0 TAU_B=-2 as well to the merged round-1 tree's exact mode. The default
twin is build/pix_twin_ref_r2. Tools: twin PCOPY / PCOPYND / HSKIP / HSKIPND, W1XSTAT also prints copied pixels;
pix/copystat.py kind 3; tools/pcopy_drift.py; tools/sheet.py out.png A=env B=env; test_gba.py force=1 paints a picture
made with PCOPY=0 HSKIP=0 as well.

##### Approximation 3: copied pixels in cells that run (PIX_PCOPY; 0 = off). -1.7 / -1.4 / -2.2 s like-for-like
Observation behind it: the head is a pure output stage. Nothing but the pixel's own colour depends on it: later cells
see a cell only through its keys, values and mixer state (made by the body) and through its painted pixels. Every
earlier copying rule skipped whole cells, body included, and that is what made them fail the yardstick (skipped
cells drop out of later cells' attention: the own-block rule cost +0.0011 KL on the other pixels). Skipping only the
head for a pixel costs nothing anywhere else.
Rule (pcopy_ok): in a cell that runs, pixel j is painted with its own thumbnail block's colour c without running the
head (no row sums, no group / member scores, no sampling) when its own block and the (up to four) blocks right /
below / left / above it have colour c, in the 64x64 pass at most one of those four may differ (PIX_PCOPY_ND = 1; the
32x32 pass allows none), and every drawn neighbour pixel (left, up, up-left, up-right) has colour c too. The pixel's
input row still goes into cellsum / ycell for the cell's later pixels (the pending row of the pixel before is added
with add16). The check comes before the pixel's own row sums, so a copied pixel costs about 200 cycles instead of
about 85,000 (64x64 pass).
Measured (mGBA, force=1: both builds paint the exact engine's picture; force now also sets PCOPY=0 HSKIP=0 for it):
car red 123.8 -> 122.1 s, no prompt 123.3 -> 121.9 s, scenes 124.5 -> 122.3 s (u64M -28.6M / -22.1M / -37.6M
cycles; u32M within +-0.3M: few pixels qualify there and every pixel pays the check). Free running: 122.9 -> 120.6,
122.7 -> 120.7, 126.3 -> 125.1 s; emulator pictures equal the twin's in all three.
Quality (all teacher-forced on build/pix_test48.u8 unless said otherwise):
- Every pixel that still runs the head gets byte-identical scores with and without PCOPY (u32M 40,746 of 40,746,
  u64M 111,636 of 111,636 modelled pixels compared; the body is untouched, the head's state per cell is rebuilt from
  the same drawn pixels). So pix.check's bits and KL on the modelled pixels are unchanged. pix.check's printed
  averages do move (u32M 1.2617 / 0.00273, u64M 1.1367 / 0.00347) only because the copied pixels, which are the
  easy ones, leave the average: the same convention that already excludes flat cells. Compare like with like.
- The copies themselves against the old flat rule's copies (pix/copystat.py, which now also lists kind 3 = HSKIP):
  | | share of pixels | model's p(copied colour) | p > 0.9 | mean -log p | off by > 16/255 |
  |---|---|---|---|---|---|
  | u64M old flat rule | 26.98% | 0.9275 | 0.8436 | 0.1188 | 1.22% |
  | u64M PCOPY | 16.24% | 0.9538 | 0.8937 | 0.0489 | 0.98% |
  | u32M old flat rule | 14.83% | 0.9322 | 0.8642 | 0.1145 | 0.95% |
  | u32M PCOPY | 2.27% | 0.9620 | 0.9123 | 0.0401 | 0.98% |
  Better than the accepted rule on every column but u32M's last (0.98 vs 0.95%).
- Free sampling, 288 pictures each, same seeds (tools/pcopy_drift.py): final pixels unlike their own thumbnail block
  30.95% +- 0.67 (off) vs 30.35% +- 0.65 (on); colours per picture 86.5 vs 85.8; flat cells (old rule) 8,027 / 16,297
  vs 8,072 / 16,424 in the 32 / 64 pass; about 650 copied pixels per picture. No drift towards flatter pictures.
- Contact sheet build/sheet_pcopy.png (tools/sheet.py out A=PCOPY=0 B=PCOPY=1): the first picture of each pair is
  the same picture with small differences; the second differs because skipped picks shift the random numbers.
Variants measured (u64M copies; on the three test prompts, copied pixels per picture chain):
- no block may differ (strict everywhere): 198 / 190 / 143 pixels, p 0.970, -log p 0.031, > 16: 0.50% (-0.8 s);
- one block may differ (kept): 607 / 473 / 477, p 0.954, 0.049, 0.98%;
- two may differ: 855 / 747 / 752, p 0.937, 0.069, 1.68% (worse than the old rule on the last column);
- one may differ and blocks within 1 step per channel count as equal: 708 / 495 / 541, p 0.948, 0.056, 0.97% (about
  +0.2 s for more code on the GBA; not taken);
- the two blocks next to the pixel's corner must match, the far ones may differ: no better than the kept rule;
- strict blocks, no condition on the drawn neighbours: 236 / 208 / 164, p 0.954, -log p 0.065, > 16: 1.43%; strict
  blocks, only left and up drawn neighbours: 203 / 191 / 161, like strict.
Whole cells (PIX_HSKIP in pix.c, default 0; twin HSKIP=1 HSKIPND=n): cells the own-block rule finds flat but the old
rule does not run the body and copy all their pixels. Exact history like PCOPY, but the copies are worse: at ND 5
(the explorer's rule) -2.2 / -1.4 / -3.0 s like-for-like with p 0.909, -log p 0.174, > 16: 4.64% (worse than the old
rule); at ND 3 p 0.940 / 0.098 / 2.85% for about 1.1 s; at ND 2 0.954 / 0.074 / 2.18% for about 0.6 s. PCOPY takes
the good pixels of those cells anyway. Left in the tree, off.

##### Looked at with fresh eyes and dropped (counts from the twin and the per-instruction profile; do not repeat)
- Where the time is (profile, car red, 123 s): table scans 59 s, scatter 17 s, fold-back 7 s, table builds 6.5 s,
  keytop 5.5 s, mixer 3.6 s, norms 5 s, everything else under 3 s each. A scan step costs and + ldr + add + a quarter
  of a 4-cycle ROM word = 6 cycles per 4 inputs; a scatter step 8. Both are at the floor of this CPU.
- The two-lane trick for one stream (two matrices, or two halves of a row, in the halves of one table entry): it
  needs the SAME table index in both halves, i.e. the same matrix row applied to two vectors. Q / K / V / O / W1 /
  W2 / HIN are each applied to one vector per cell, and cells (and pixels within a cell) are strictly sequential.
  Packing two inputs of one row, or two rows, would need 81 x 81-entry tables. Not possible.
- Exact two streams with 64-bit table entries (ldm of a pair: 9 cycles per step for both streams instead of 12 for
  two single scans): only the guided head's first matrix runs per stream (0.26 s), and its tables would need 20 KB.
- The exact k-split of mlp_out2 for the head's first matrix (stream 0 high, difference low): sum |z0| needs 19 bits,
  so the difference would have to stay below 2^11 summed over 128 inputs. Never fits.
- mlp_out2's non-fitting calls (175 of 1408 per picture) split by units instead of by streams (only the units whose
  differences do not fit take a second pass, average 31 of 152; exact, twin identical): -0.06 s, because the
  partition and the recombination are C outside fast RAM and the fold-back is a fixed cost. Removed.
- Mixer in 32-bit arithmetic: the worst-case bound of sum |c| * max |xn| per dimension is 2^31..2^33.6, so it is not
  exact without a per-call check. int16 coefficients (all fit) save one ROM word in two but cost three unpacking
  instructions per word: about 3 cycles per dimension. Swapping the multiplier operands: states already have
  16-bit magnitudes (smlal at 5 cycles).
- Whole-scan termination in keytop by the distance bias (rows sorted by bias, stop when the 8th best beats bias +
  sum |q|): the biases are learned and not monotonic, and sum |q| (about 15,000) dwarfs their spread far back.
- Bias correction for dropped hidden values (thresholds drop only positive values): the mean part is about a tenth
  of the error energy (p ~ 0.1 of being dropped), so it would recover very little KL.
- KL headroom from the head threshold: TAU_H 0 instead of 32 gives back only 0.00001-0.00005 KL per model.
- Caches across cells (input rows, thumbnail rows, neighbour rows, rope): each row is position- and colour-specific
  (prev / up / thumbnail rows depend on the slot and the window position), nothing repeats between cells.
