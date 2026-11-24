"""Build the test ROM, run it in headless mGBA, require the same picture as the native build, report cycles.
    python3 pix/test_gba.py [guide_q8=768] [words=288,10] [cells=N] [base=3] [cflags=...] [w1x=0: without the early exit of the first MLP matrix] [hs=0: without PIX_HEADSKIP] [force=1: paint the exact engine's picture, for like-for-like timing]"""
import subprocess, struct, sys, os, numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
o = dict(a.split('=', 1) for a in sys.argv[1:])
guide, words, cells, base = int(o.get('guide_q8', 768)), o.get('words', '288,10'), int(o.get('cells', 100000)), int(o.get('base', 0))
w8 = (words.split(',') + ['0'] * 8)[:8] if words else ['0'] * 8
CF = '-mcpu=arm7tdmi -mtune=arm7tdmi -marm -O2 -ffreestanding -fno-builtin -Wall -Igen -Ipix -DPIX_PROF -DPIX_PROFA'.split() + o.get('cflags', '').split() + [f'-DTEST_GUIDE={guide}', f'-DTEST_WORDS={",".join(w8)}', f'-DTEST_CELLS={cells}', f'-DTEST_BASE={base}'] + ([f"-DW1X={o['w1x']}"] + (['-Wa,--defsym,NOW1X=1'] if o['w1x'] == '0' else []) if 'w1x' in o else [])
force = int(o.get('force', 0))      # force=1: every pass paints the picture the twin makes WITHOUT the approximations (EXACT below): the same work for any setting
EXACT = dict(W1X='0', TAU_B='-2', FLATRULE='0', HEADSKIP='0', G2B='0', PCOPY='0')
if 'hs' in o: CF += [f"-DPIX_HEADSKIP={o['hs']}"]       # hs=0: without the body-only cells of the 64x64 pass (PIX_HEADSKIP)
if force:
    subprocess.run([os.environ.get('PIX_TWIN', 'build/pix_twin'), 'build/pix.bin', 'paint' if not base else 'paint3', '12345', str(guide), '284', '1', *[x for x in w8 if x != '0']], capture_output=True, env=dict(os.environ, PIXDUMP='build/force.bin', **EXACT))
    open('build/force.c', 'w').write('const unsigned char force_pix[] = {' + ','.join(str(b) for b in open('build/force.bin', 'rb').read()) + '};\n')
    CF += ['-DPIX_FORCE']
def sh(*a): subprocess.run(a, check=True)
sh('arm-none-eabi-gcc', *CF, *o.get('pcflags', '').split(), '-c', 'pix/pix.c', '-o', 'build/pix_engine.o')
sh('arm-none-eabi-gcc', *CF, '-nostdlib', '-Wl,-T,pix/gba.ld', '-Wl,-Map,build/pix_test.map', '-o', 'build/pix_test.elf',
   'pix/crt0.s', 'pix/gba_test.c', 'build/pix_engine.o', 'pix/kernels.s', 'pix/blob.s', *(['build/force.c'] if force else []), '-lgcc')
sh('arm-none-eabi-objcopy', '-O', 'binary', 'build/pix_test.elf', 'build/pix_test.gba')
sh('python3', 'tools/fixrom.py', 'build/pix_test.gba')
sh('arm-none-eabi-size', 'build/pix_test.elf')
lay = open('gen/pix_layout.h').read(); NM = 3; names = lay.split('/* models: ')[1].split(' */')[0].split()
PP = ['input rows', 'norm', 'mixer', 'attention', 'mlp in', 'mlp out', 'to head', 'head rows', 'head in', 'head out', 'scores', 'pick']
n = 4 + NM + NM * len(PP)
sh('build/runner', 'build/pix_test.gba', 'until:0x02000004:1:200000', f'dump:0x02000000:{4 * n + 4096 + 132}:build/pix_dbg.bin')
raw = open('build/pix_dbg.bin', 'rb').read(); v = struct.unpack(f'{n}I', raw[:4 * n])
assert v[0] == 0x50495831, hex(v[0])
img = np.frombuffer(raw[4 * n:4 * n + 4096], np.uint8); pa = np.frombuffer(raw[4 * n + 4096:], np.uint32).reshape(3, 11)
if force: print('painted the given picture:', bytes(img) == open('build/force.bin', 'rb').read()[1280:])
elif cells >= 100000:
    ref = subprocess.run([os.environ.get('PIX_TWIN', 'build/pix_twin'), 'build/pix.bin', 'paint' if not base else 'paint3', '12345', str(guide), '284', '1', *[x for x in w8 if x != '0']], capture_output=True, env=dict(os.environ, **({'W1X': o['w1x']} if 'w1x' in o else {}), **({'W1X2': o['w1x2']} if 'w1x2' in o else {}), **({'HEADSKIP': o['hs']} if 'hs' in o else {}))).stdout
    print('picture identical to the native build:', bytes(img) == ref)
tot = 0
for m in range(NM):
    cyc = v[4 + m] * 256; tot += cyc; prof = v[4 + NM + m * len(PP):4 + NM + (m + 1) * len(PP)]; ps = sum(prof) * 256
    print(f"{names[m]}: {cyc / 16777216:7.1f} s  ({cyc:,} cycles)")
    print('   ' + '  '.join(f"{nm} {100 * p * 256 / ps:.1f}%" for nm, p in zip(PP, prof)))
    print('   attention parts %: ' + ' '.join(f"{nm} {100 * a / max(1, pa[m].sum()):.0f}" for nm, a in zip('build Krows key Vrows vquant Qrows Qrope qnorm keyscan top8+mix out'.split(), pa[m])))
print(f"TOTAL {tot / 16777216:.1f} s = {int(tot / 16777216 // 60)}:{int(tot / 16777216 % 60):02d}")
