"""Free sampling with and without PIX_PCOPY (288 pictures each, same seeds and prompts): how many pixels of the final
picture differ from their own thumbnail block, how many distinct colours a picture has, flat cells, copied pixels.
    python3 tools/pcopy_drift.py"""
import os, subprocess, numpy as np
from concurrent.futures import ThreadPoolExecutor
JOBS = ["paint 1000 768 284 48 288 10", "paint 2000 768 284 48", "paint3 3000 768 284 48 500 77", "paint 4000 768 284 48 14 5 300", "paint3 5000 768 284 48", "paint 6000 768 284 48 901 33 7"]
def run(a):
    job, env = a
    fn = f"build/drift_{abs(hash((job, tuple(env.items()))))}.bin"
    r = subprocess.run(["build/pix_twin", "build/pix.bin", *job.split()], capture_output=True, env=dict(os.environ, W1XSTAT="1", PIXDUMP=fn, **env))
    d = np.fromfile(fn, np.uint8); os.remove(fn)
    line = [l for l in r.stderr.decode().splitlines() if "cells painted" in l][0]
    return d, line
for lab, env in (("PCOPY off", {"PCOPY": "0"}), ("PCOPY on", {"PCOPY": "1"})):
    with ThreadPoolExecutor(6) as ex: res = list(ex.map(run, [(j, env) for j in JOBS]))
    off = []; ncol = []; lines = []
    for d, line in res:
        per = 256 + 1024 + 4096; d = d.reshape(-1, per)
        th = d[:, 256:1280].reshape(-1, 32, 32); fin = d[:, 1280:].reshape(-1, 64, 64)
        up = th.repeat(2, 1).repeat(2, 2)
        off.append((fin != up).mean(axis=(1, 2))); ncol += [len(np.unique(p)) for p in fin]; lines.append(line)
    off = np.concatenate(off)
    print(f"{lab}: final pixels unlike their thumbnail block {100 * off.mean():.2f}% (+- {100 * off.std() / np.sqrt(len(off)):.2f})  colours per picture {np.mean(ncol):.1f}")
    for l in lines: print("   ", l.split(": ", 1)[1][:130])
