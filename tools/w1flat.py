"""Free sampling with and without the early exit (twin, same seeds): how many cells the enlargers paint straight from
the thumbnail (flat windows) and how many hidden units are active: a picture model that drifted towards flatter or
emptier pictures would show here.   python3 tools/w1flat.py [pictures per prompt = 24]"""
import os, re, subprocess, sys, numpy as np
n = sys.argv[1] if len(sys.argv) > 1 else '24'
jobs = [('paint', 1000, '288 10'), ('paint', 2000, '14 5 300'), ('paint', 3000, ''), ('paint3', 4000, '500 77'), ('paint3', 5000, '')]
tot = {'0': np.zeros(5), '1': np.zeros(5)}
for mode, seed, words in jobs:
    for x in '01':
        e = subprocess.run(['build/pix_twin', 'build/pix.bin', mode, str(seed), '768', '284', n, *words.split()], capture_output=True, env=dict(os.environ, W1X=x, W1XSTAT='1')).stderr.decode()
        v = np.array([int(a) for a in re.search(r'thumbnail: (\d+) (\d+); active hidden units: (\d+) (\d+) (\d+)', e).groups()], float); tot[x] += v
        print(mode, words, 'W1X=' + x, v.astype(int), flush=True)
print('all: without', tot['0'].astype(int), 'with', tot['1'].astype(int), 'ratio', np.round(tot['1'] / tot['0'], 3))
