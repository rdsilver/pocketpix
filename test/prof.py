"""python3 test/prof.py [top]: per-function cycles of build/pix_test.gba (build it first with pix/test_gba.py)"""
import subprocess, numpy as np, sys
subprocess.run(['build/prof', 'build/pix_test.gba', 'build/prof.bin'], check=True)
h = np.fromfile('build/prof.bin', np.uint64).reshape(-1, 2)
NI, NV, NR = 0x2000, 0x2000, 0x4000
syms = []
for l in subprocess.run(['arm-none-eabi-nm', '-n', 'build/pix_test.elf'], capture_output=True, text=True).stdout.splitlines():
    a, t, nm = l.split()
    if t in 'tT' and not nm.startswith('$') and not nm.startswith('.L'): syms.append((int(a, 16), nm))
def addr(ix): return 0x03000000 + ix * 4 if ix < NI else 0x06010000 + (ix - NI) * 4 if ix < NI + NV else 0x08000000 + (ix - NI - NV) * 4
import bisect
keys = [s[0] for s in syms]; tot = {}; cnt = {}
for ix in np.nonzero(h[:-1, 0])[0]:
    a = addr(ix); k = bisect.bisect_right(keys, a) - 1; nm = syms[k][1] if k >= 0 else '?'
    tot[nm] = tot.get(nm, 0) + int(h[ix, 0]); cnt[nm] = cnt.get(nm, 0) + int(h[ix, 1])
tot['(elsewhere)'] = int(h[-1, 0]); cnt['(elsewhere)'] = int(h[-1, 1])
T = sum(tot.values())
print(f"total {T / 16777216:.2f} s; iwram {h[:NI, 0].sum() / 16777216:.2f}  vram {h[NI:NI + NV, 0].sum() / 16777216:.2f}  rom {h[NI + NV:-1, 0].sum() / 16777216:.2f} s")
for nm, c in sorted(tot.items(), key=lambda x: -x[1])[:int(sys.argv[1]) if len(sys.argv) > 1 else 60]:
    print(f"{nm:24s} {c / 16777216:8.2f} s {100 * c / T:5.1f}%  {cnt[nm]:12d} instr  {c / max(1, cnt[nm]):.2f} cyc/instr")
