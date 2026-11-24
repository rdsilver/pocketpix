"""python3 test/profdis.py func...: per-instruction cycles (from build/prof.bin) next to the disassembly"""
import subprocess, numpy as np, sys
h = np.fromfile('build/prof.bin', np.uint64).reshape(-1, 2)
NI, NV, NR = 0x2000, 0x2000, 0x4000
def ix(a): return (a - 0x03000000) // 4 if a >> 24 == 3 else NI + (a - 0x06010000) // 4 if a >> 24 == 6 else NI + NV + (a - 0x08000000) // 4
for fn in sys.argv[1:]:
    d = subprocess.run(['arm-none-eabi-objdump', '-d', f'--disassemble={fn}', 'build/pix_test.elf'], capture_output=True, text=True).stdout
    for l in d.splitlines():
        p = l.split('\t')
        if len(p) >= 3 and p[0].strip().endswith(':'):
            a = int(p[0].strip()[:-1], 16); c, n = h[ix(a)]
            print(f"{c / 16777216:7.3f}s {n:10d} {c / max(1, n):5.2f}  {p[0].strip()} {' '.join(p[2:])}")
        elif l.endswith('>:'): print(l)
