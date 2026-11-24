"""Gallery tests in headless mGBA. Screenshots go to the directory given as the first argument.
    python3 pix/test_gallery.py OUTDIR [real]     ('real' also paints one true picture with the shipped ROM: +40 s)
Needs build/pocketpix.gba (python3 pix/build_game.py) and build/pocketpix_galtest.gba (python3 pix/build_game.py test;
run the plain build again afterwards, the two share build/pix_engine.o)."""
import subprocess, sys, os, re, random
from PIL import Image
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # the repository root
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
ROM, TROM = 'build/pocketpix.gba', 'build/pocketpix_galtest.gba'
A, B, SELECT, START, RIGHT, LEFT, UP, DOWN, R, L = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
def sym(mapfile, name): return int(re.search(r'(0x[0-9a-f]+)\s+' + name + r'\n', open(mapfile).read()).group(1), 16)
def k(*keys): return [f'k:{x}' for x in keys]
def run(rom, steps, save=None, load=True, extra=()):
    """steps: runner steps; ('shot', name) and ('dump', addr, len) are filled in; returns the dumps in order"""
    args, dumps, shots = list(extra) + ([f'loadsave:{save}'] if save and load else []), [], []
    for s in steps:
        if isinstance(s, tuple) and s[0] == 'shot': args.append(f'shot:{OUT}/{s[1]}.ppm'); shots.append(s[1])
        elif isinstance(s, tuple): f = f'{OUT}/_d{len(dumps)}.bin'; args.append(f'dump:{s[1]}:{s[2]}:{f}'); dumps.append(f)
        else: args.append(s)
    if save: args.append(f'storesave:{save}')
    subprocess.run(['build/runner', rom, *args], check=True)
    for n in shots: Image.open(f'{OUT}/{n}.ppm').resize((480, 320), 0).save(f'{OUT}/{n}.png'); os.remove(f'{OUT}/{n}.ppm')
    r = [open(f, 'rb').read() for f in dumps]
    for f in dumps: os.remove(f)
    return r
FLASH = ('dump', 0x0E000000, 65536)
VIEW = ('dump', 0x06000000 + 4 * 240, 128 * 240)              # the viewer's picture rows
def viewed(v):
    """the 64x64 picture the viewer shows, checked to be drawn at exactly 2x"""
    rows = [v[y * 240 + 8:y * 240 + 136] for y in range(128)]
    for y in range(64): assert rows[2 * y] == rows[2 * y + 1] and rows[2 * y][0::2] == rows[2 * y][1::2]
    return b''.join(rows[2 * y][0::2] for y in range(64))
def check16(b):
    h = 2166136261
    for x in b: h = ((h ^ x) * 16777619) & 0xFFFFFFFF
    return (h ^ h >> 16) & 0xFFFF
def note(slot, used, seq, words=(), base=0, pic=b''):
    w = list(words) + [0] * (8 - len(words))
    b = bytes([0xA5, slot, used, base, len(words), 0, 0, 0]) + seq.to_bytes(4, 'little') + b''.join(x.to_bytes(2, 'little') for x in w) + check16(pic).to_bytes(2, 'little')
    return b + check16(b).to_bytes(2, 'little')
def notes(fl):
    """parse the diary the way the game does: {slot: (seq, nwords, words, base)} and how many notes are used"""
    d, used = {}, 0
    assert fl[0xF000:0xF008] == b'PPIXGAL\1'
    for r in range(127):
        b = fl[0xF020 + r * 32:0xF040 + r * 32]
        if b == b'\xff' * 32: continue
        used = r + 1
        if b[0] != 0xA5 or check16(b[:30]) != int.from_bytes(b[30:], 'little'): continue
        if b[2] and check16(fl[b[1] * 4096:b[1] * 4096 + 4096]) == int.from_bytes(b[28:30], 'little'):
            d[b[1]] = (int.from_bytes(b[8:12], 'little'), b[4], [int.from_bytes(b[12 + 2 * i:14 + 2 * i], 'little') for i in range(b[4])], b[3])
        else: d.pop(b[1], None)
    return d, used
def fab(n):                                                   # the test ROM's made-up picture number n
    p = bytearray((i * 3 + (i >> 6) * n + n * 16) & 255 for i in range(4096)); p[0] = n; return bytes(p)
def newest_first(fl): d, _ = notes(fl); return [fl[s * 4096:s * 4096 + 4096] for s in sorted(d, key=lambda s: -d[s][0])]
GAL = k(DOWN, DOWN, RIGHT, RIGHT, RIGHT, A) + ['f:4']         # from a fresh entry screen: open the gallery
KEEP1 = k(START) + ['f:6'] + k(START) + ['f:12']              # test ROM, from the entry screen: paint and keep
def sv(name):
    p = f'{OUT}/{name}.sav'
    if os.path.exists(p): os.remove(p)
    return p
def write(p, data): open(p, 'wb').write(data)
ok = lambda name: print('ok  ', name)

# ---- 1. empty states: no save file, blank flash, garbage, short file; nothing crashes, the gallery is empty
random.seed(7)
magic_garbage = bytearray(random.randbytes(65536)); magic_garbage[0xF000:0xF008] = b'PPIXGAL\1'
almost = bytearray(b'\xff' * 65536); almost[0xF000:0xF008] = b'PPIXGAL\1'
for r in range(127): almost[0xF020 + r * 32:0xF040 + r * 32] = note(r % 15, 1, r + 1, [5, 9], 0, b'not the picture')
for name, data in [('nofile', None), ('blank', b'\xff' * 65536), ('zeros', bytes(65536)), ('garbage', random.randbytes(65536)), ('magic_garbage', bytes(magic_garbage)),
                   ('notes_without_pictures', bytes(almost)), ('short', random.randbytes(1000)), ('half', random.randbytes(32768))]:
    s = sv('e_' + name)
    if data is not None: write(s, data)
    fl0, v = run(TROM, ['f:30', FLASH] + GAL + [('shot', 'empty_' + name), VIEW], s)
    blank = run(TROM, ['f:30'] + GAL + [VIEW], sv('e_ref'))[0]
    assert v == blank, name                                   # exactly the "No pictures kept yet" screen
    if data is not None and len(data) == 65536: assert fl0 == data, name          # booting and looking changes nothing
    # and a first keep works from that state, and is there after a restart
    fl, = run(TROM, ['f:30'] + KEEP1 + [FLASH], s)
    fl2, v = run(TROM, ['f:30', FLASH] + GAL + [VIEW], s)
    assert fl2 == fl and newest_first(fl) == [fab(0)] and viewed(v) == fab(0), name
    assert notes(fl)[1] == 1, name
ok('no save / blank / garbage / short save files: empty gallery, first keep works')

# ---- 2. no usable save chip: clear message, the game still paints, START does nothing
for t, name in [(0, 'none'), (1, 'sram'), (3, 'flash128k')]:
    run(TROM, ['f:30'] + GAL + [('shot', 'nochip_gallery_' + name)] + k(B) + ['f:4'] + k(START) + ['f:6', ('shot', 'nochip_done_' + name)] + k(START) + ['f:12', ('shot', 'nochip_done_start_' + name)]
        + k(A) + ['f:6'] + k(B) + ['f:6', ('shot', 'nochip_entry_' + name)], extra=[f'savetype:{t}'])
ok('forced no save / SRAM / 128K flash: ran (see nochip_*.png)')

# ---- 3. keep, no double keep, "again" then keep, restart, flip, delete with confirm, restart
s = sv('t3')
fl_a, fl_b, fl_c = run(TROM, ['f:30'] + KEEP1 + [('shot', 't_kept1'), FLASH] + k(START) + ['f:12', FLASH, 'f:200', ('shot', 't_kept1_later')] + k(A) + ['f:6'] + k(START) + ['f:12', ('shot', 't_kept2'), FLASH], s)
assert fl_a == fl_b and newest_first(fl_a) == [fab(0)] and newest_first(fl_c) == [fab(1), fab(0)]
v1, v2, v3, v4, v5, v6, fl_d = run(TROM, ['f:30'] + GAL + [VIEW, ('shot', 't_view_1of2')] + k(RIGHT) + ['f:4', VIEW, ('shot', 't_view_2of2')] + k(RIGHT) + ['f:4', VIEW] + k(L) + ['f:4', VIEW]
    + k(SELECT) + ['f:4', ('shot', 't_delete_ask')] + k(B) + ['f:4', VIEW, ('shot', 't_delete_no')] + k(SELECT) + ['f:4'] + k(A) + ['f:8', VIEW, ('shot', 't_deleted'), FLASH], s)
assert [viewed(v) for v in (v1, v2, v3, v4, v5, v6)] == [fab(1), fab(0), fab(1), fab(0), fab(0), fab(1)]
assert newest_first(fl_d) == [fab(1)]
v1, fl_e = run(TROM, ['f:30'] + GAL + [VIEW] + k(SELECT) + ['f:4'] + k(A) + ['f:8', ('shot', 't_deleted_last'), FLASH], s)
assert viewed(v1) == fab(1) and newest_first(fl_e) == []
fl_f, v = run(TROM, ['f:30', FLASH] + GAL + [VIEW], s)
assert fl_f == fl_e and v == blank
ok('keep once only, again+keep, restart, flip with LEFT/RIGHT and L/R, delete asks first, deletes survive a restart')

# ---- 4. fill 15, replace-oldest question (no, then yes), order newest first, restart
s = sv('t4')
one = k(START) + ['f:6'] + k(START) + ['f:12'] + k(B) + ['f:6']
fl15, fl_no, fl16 = run(TROM, ['f:30'] + one * 14 + KEEP1 + [('shot', 'f_kept15'), FLASH] + k(B) + ['f:6'] + k(START) + ['f:6'] + k(START) + ['f:6', ('shot', 'f_replace_ask')]
    + k(B) + ['f:12', ('shot', 'f_replace_no'), FLASH] + k(START) + ['f:6'] + k(A) + ['f:12', ('shot', 'f_replace_yes'), FLASH], s)
assert newest_first(fl15) == [fab(n) for n in range(14, -1, -1)] and fl_no == fl15
assert newest_first(fl16) == [fab(n) for n in range(15, 0, -1)] and fl16[:4096] == fab(15)       # the oldest sat in slot 0
steps = ['f:30'] + GAL + [VIEW, ('shot', 'f_view_1of15')]
for i in range(15): steps += k(RIGHT if i % 2 else R) + ['f:4', VIEW]
steps += k(LEFT) + ['f:4', VIEW, ('shot', 'f_view_15of15')]
vs = run(TROM, steps, s)
assert [viewed(v) for v in vs] == [fab(n) for n in range(15, 0, -1)] + [fab(15), fab(1)]
ok('15 kept, "Replace oldest?" no/yes, newest first after a restart, flipping wraps round')

# ---- 5. the diary fills up and is rewritten; nothing is lost. Then deletes in the middle and refills.
rep = k(START) + ['f:6'] + k(START) + ['f:6'] + k(A) + ['f:12'] + k(B) + ['f:6']
fl, = run(TROM, ['f:30'] + rep * 120 + [FLASH], s)             # made-up pictures 0..119 in this run, on top of 1..15
d, used = notes(fl)
assert newest_first(fl) == [fab(n) for n in range(119, 104, -1)], 'after 120 replacements'
assert len(d) == 15 and used < 127, used
print('     notes in the diary after 16 + 120 keeps:', used, '(127 fit; it was rewritten once on the way)')
steps = ['f:30'] + GAL + k(RIGHT, RIGHT, SELECT) + ['f:4'] + k(A) + ['f:8', VIEW] + k(LEFT, LEFT, SELECT) + ['f:4'] + k(A) + ['f:8', VIEW, FLASH]
v1, v2, fl = run(TROM, steps, s)                               # delete the 3rd newest (117), then step back to the newest (119) and delete it
assert viewed(v1) == fab(116) and viewed(v2) == fab(118)
assert newest_first(fl) == [fab(n) for n in (118, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105)]
fl, fl2 = run(TROM, ['f:30'] + one * 2 + [FLASH] + k(START) + ['f:6'] + k(START) + ['f:6', ('shot', 'f_full_again')] + k(A) + ['f:12', FLASH], s)
assert newest_first(fl)[:3] == [fab(1), fab(0), fab(118)] and len(newest_first(fl)) == 15
assert newest_first(fl2)[:2] == [fab(2), fab(1)] and newest_first(fl2)[-1] == fab(106) and len(newest_first(fl2)) == 15
ok('diary rewrite, deletes in the middle, freed slots are reused, replace picks the oldest')

# ---- 6. power cuts, made by hand from a good 3-picture save
s = sv('t6')
good, = run(TROM, ['f:30'] + one * 3 + [FLASH], s)
def boot_view(data, n, name):
    write(s, bytes(data)); steps = ['f:30'] + GAL + [VIEW]
    for i in range(n): steps += k(RIGHT) + ['f:4', VIEW]
    r = run(TROM, steps + [('shot', name)], s)
    return r[n] == blank if n == 0 else [viewed(v) for v in r[:n]] + [viewed(r[n])]
c = bytearray(good); c[3 * 4096:4 * 4096] = fab(77)           # a 4th picture was written, its note was not
assert boot_view(c, 3, 'cut_no_note') == [fab(2), fab(1), fab(0), fab(2)]
c = bytearray(good); c[1 * 4096:2 * 4096] = b'\xff' * 2048 + fab(77)[2048:]      # slot 1 was being replaced
assert boot_view(c, 2, 'cut_replacing') == [fab(2), fab(0), fab(2)]
c = bytearray(good); c[0xF020 + 3 * 32:0xF020 + 3 * 32 + 20] = note(3, 1, 4, [], 0, fab(77))[:20]; c[3 * 4096:4 * 4096] = fab(77)   # a note half written
assert boot_view(c, 3, 'cut_half_note') == [fab(2), fab(1), fab(0), fab(2)]
fl, = run(TROM, ['f:30'] + KEEP1 + [FLASH], s)                 # the next keep goes after the spoiled note and works
assert newest_first(fl) == [fab(0), fab(2), fab(1), fab(0)] and notes(fl)[1] == 5
c = bytearray(good); c[0xF000:] = b'\xff' * 4096               # the worst case: cut while the full diary was being rewritten
assert boot_view(c, 0, 'cut_diary_erased') is True
ok('power-cut states: other pictures stay, a spoiled note is skipped; an erased diary reads as empty')

# ---- 7. the shipped ROM: type a prompt, paint for real, keep, restart, compare byte for byte
if 'real' in sys.argv:
    s = sv('real'); img_at = sym('build/pocketpix.map', 'pix_img')
    typed = k(RIGHT, RIGHT, A, LEFT, LEFT, A, DOWN, DOWN, A) + k(RIGHT, RIGHT, RIGHT, A, DOWN, LEFT, LEFT, A, R, DOWN, A, SELECT)   # "ca" ADD, "do" next match ADD, scenes
    img, fl, fl2 = run(ROM, ['f:30', ('shot', 'real_entry')] + typed + [('shot', 'real_prompt')] + k(START) + ['f:600', ('shot', 'real_painting'), 'f:9400', ('shot', 'real_done'), ('dump', img_at, 4096)]
        + k(START) + ['f:20', ('shot', 'real_kept'), FLASH] + k(START) + ['f:20', FLASH, 'f:200', ('shot', 'real_kept_later')], s)
    d, used = notes(fl)
    assert fl == fl2 and used == 1 and fl[:4096] == img and len(set(img)) > 20, (used, len(set(img)))
    print('     kept prompt: words', d[0][2], 'scenes' if d[0][3] else 'objects')
    fl3, v, v2 = run(ROM, ['f:30', FLASH] + GAL + [VIEW, ('shot', 'real_view')] + k(A) + ['f:4', ('shot', 'real_use_prompt')] + k(A) + ['f:4', VIEW], s)     # the cursor is still on GALLERY
    assert fl3 == fl and viewed(v) == img and viewed(v2) == img
    ok('shipped ROM: real picture kept, byte-identical in flash and in the viewer after a restart, prompt restored')
print('ALL GALLERY TESTS PASSED')
