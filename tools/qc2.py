"""pix/qc.py for several settings at once, with the averages also taken over the SAME pixels: those every setting
computes (a setting that paints more cells from the thumbnail measures fewer, easier-or-harder pixels, which moves
pix.check's averages by itself). Uses pix/qc.py's caches (run pix/qc.py once first).
    python3 tools/qc2.py [models=u32M,u64M] "label ENV=v ..." "label ENV=v ..." ...   (first = baseline)"""
import sys, os, subprocess, numpy as np
from concurrent.futures import ThreadPoolExecutor
o = dict(a.split('=', 1) for a in sys.argv[1:] if ' ' not in a and '=' in a)
sets = [a for a in sys.argv[1:] if ' ' in a or '=' not in a]
ALL = ["b16C", "u32M", "u64M", "b16O"]
def one(args):
    mi, k, ref, env = args
    fn = f"build/qc2_{os.getpid()}_{mi}_{k}_{abs(hash(env)) % 99999}.bin"; open(fn, "wb").write(ref.tobytes())
    e = dict(os.environ); e.update(dict(x.split('=', 1) for x in env.split()))
    r = subprocess.run([o.get('twin', 'build/pix_twin'), 'build/pix.bin', "force", str(mi), fn], capture_output=True, check=True, env=e); os.remove(fn)
    return np.frombuffer(r.stdout, np.int32).reshape(-1, 272)
for name in o.get('models', 'u32M,u64M').split(','):
    mi = ALL.index(name); c = np.load(f"build/qc_cache_{name}.npz"); LP, REF, PX = c["lp"], c["ref"], c["px"]
    per = []
    for st in sets:
        lab, *env = st.split(); env = ' '.join(env)
        with ThreadPoolExecutor(4) as ex: outs = list(ex.map(one, [(mi, k, REF[k], env) for k in range(len(LP))]))
        rows = []
        for k, raw in enumerate(outs):
            keep = raw[:, 0] != -0x7FFFFFFF
            s = raw.astype(np.float64) / 4096; s[~keep] = 0
            lg = s[:, :16] - np.log(np.exp(s[:, :16]).sum(1, keepdims=True)); mm = s[:, 16:].reshape(-1, 16, 16)
            lm = mm - np.log(np.exp(mm).sum(2, keepdims=True)); li = (lg[:, :, None] + lm).reshape(-1, 256); lp = LP[k].astype(np.float64)
            ar = np.arange(len(PX[k]))
            rows.append((keep, -li[ar, PX[k]] / np.log(2), (np.exp(lp) * (lp - li)).sum(1)))
        per.append((lab, rows))
    common = [np.logical_and.reduce([p[1][k][0] for p in per]) for k in range(len(LP))]
    base = None
    for lab, rows in per:
        own = np.mean([[r[1][r[0]].mean(), r[2][r[0]].mean(), (~r[0]).mean()] for r in rows], 0)
        com = np.mean([[r[1][cm].mean(), r[2][cm].mean()] for r, cm in zip(rows, common)], 0)
        d = np.array([[r[1][cm].mean(), r[2][cm].mean()] for r, cm in zip(rows, common)])
        if base is None: base = d
        dd = d - base; se = dd.std(0, ddof=1) / np.sqrt(len(dd))
        print(f"{name} {lab:14s} own: bits {own[0]:.4f} KL {own[1]:.5f} skipped {100*own[2]:.1f}% | common pixels: bits {com[0]:.4f} ({dd[:,0].mean():+.4f}+-{se[0]:.4f}) KL {com[1]:.5f} ({dd[:,1].mean():+.5f}+-{se[1]:.5f})", flush=True)
