"""Pixels the engine paints without the model (teacher-forced on the 48 test pictures): how likely the float model
finds the copied colour. kind 0: flat cells (old rule), 1: cells made of flat spots, 2: flat spots in running cells."""
import subprocess, sys, os, numpy as np
from concurrent.futures import ThreadPoolExecutor
ALL = ["b16C", "u32M", "u64M", "b16O"]; twin = "build/pix_twin"
palb = np.load("img/remote/pal/p16o.npy").astype(np.int64)
def one(args):
    mi, k, ref = args
    fn = f"build/qc_refc_{os.getpid()}_{mi}_{k}.bin"; open(fn, "wb").write(ref.tobytes())
    r = subprocess.run([twin, "build/pix.bin", "force", str(mi), fn], capture_output=True, check=True); os.remove(fn)
    return np.frombuffer(r.stdout, np.int32).reshape(-1, 272)
for name in sys.argv[1:] or ["u32M", "u64M"]:
    mi = ALL.index(name); c = np.load(f"build/qc_cache_{name}.npz"); LP, REF, PX = c["lp"], c["ref"], c["px"]
    with ThreadPoolExecutor(4) as ex: outs = list(ex.map(one, [(mi, k, REF[k]) for k in range(len(LP))]))
    raw = np.concatenate(outs); lp = LP.reshape(-1, 256).astype(np.float64); px = PX.reshape(-1)
    cop = raw[:, 0] == -0x7FFFFFFF; col = raw[:, 16]; kind = raw[:, 17]
    pmax = np.exp(lp.max(1))
    print(f"{name}: {len(px)} pixels, copied {100 * cop.mean():.1f}%")
    for kd, lab in ((0, "flat cells (old rule)"), (1, "cells of flat spots"), (2, "flat spots in running cells"), (3, "head skipped (HSKIP)")):
        f = cop & (kind == kd)
        if not f.any(): print(f"   {lab:28s}: none"); continue
        p = np.exp(lp[f, col[f]]); err = np.abs(palb[px[f]] - palb[col[f]]).max(1)
        print(f"   {lab:28s}: {100 * f.mean():5.2f}% of pixels ({f.sum() / len(outs):6.1f} per picture)  model's prob of the copied colour: mean {p.mean():.4f}  above 0.5: {(p > .5).mean():.4f}  above 0.9: {(p > .9).mean():.4f} | picture has that colour: {(px[f] == col[f]).mean():.4f}  off by more than 16/255: {100 * (err > 16).mean():.2f}%  mean -log p {(-lp[f, col[f]]).mean():.4f}")
