"""Pixels painted from the thumbnail (teacher-forced on the 48 test pictures): how likely the float model finds the
copied colour, per setting. Kinds: 0 skipped flat cells, 3 context cells (PIX_FLATPROT). Uses pix/qc.py's caches.
    python3 tools/pcopy.py [models=u32M,u64M] "label ENV=v ..." ..."""
import sys, os, subprocess, numpy as np
from concurrent.futures import ThreadPoolExecutor
o = dict(a.split('=', 1) for a in sys.argv[1:] if ' ' not in a and '=' in a)
sets = [a for a in sys.argv[1:] if ' ' in a or '=' not in a]
ALL = ["b16C", "u32M", "u64M", "b16O"]
def one(args):
    mi, k, ref, env = args
    fn = f"build/pc_{os.getpid()}_{mi}_{k}_{abs(hash(env)) % 99999}.bin"; open(fn, "wb").write(ref.tobytes())
    e = dict(os.environ); e.update(dict(x.split('=', 1) for x in env.split()))
    r = subprocess.run(['build/pix_twin', 'build/pix.bin', "force", str(mi), fn], capture_output=True, check=True, env=e); os.remove(fn)
    return np.frombuffer(r.stdout, np.int32).reshape(-1, 272)
for name in o.get('models', 'u32M,u64M').split(','):
    mi = ALL.index(name); c = np.load(f"build/qc_cache_{name}.npz"); LP, REF, PX = c["lp"], c["ref"], c["px"]
    for st in sets:
        lab, *env = st.split(); env = ' '.join(env)
        with ThreadPoolExecutor(4) as ex: outs = list(ex.map(one, [(mi, k, REF[k], env) for k in range(len(LP))]))
        raw = np.concatenate(outs); lp = LP.reshape(-1, 256).astype(np.float64); px = PX.reshape(-1)
        cop = raw[:, 0] == -0x7FFFFFFF; col = raw[:, 16]; kind = raw[:, 17]; n = len(outs)
        p = np.exp(lp[cop, col[cop]]); nl = -lp[cop, col[cop]]
        H = -(np.exp(lp[cop]) * lp[cop]).sum(1); ex = (nl - H).sum() / n         # -log p(copy) - entropy: the excess over sampling
        msg = f"{name} {lab:12s} copied {100 * cop.mean():5.1f}% ({cop.sum() / n:6.1f} px/picture) p(copy) {p.mean():.4f}  >0.9 {(p > .9).mean():.3f}  -log p {nl.mean():.4f}  per picture {nl.sum() / n:7.2f} nats (minus entropy {ex:6.2f})"
        for kd in (0, 3):
            f = cop & (kind == kd)
            if f.any() and (kind[cop] == 3).any(): msg += f" | kind {kd}: {f.sum() / n:.1f} px p {np.exp(lp[f, col[f]]).mean():.4f}"
        print(msg, flush=True)
