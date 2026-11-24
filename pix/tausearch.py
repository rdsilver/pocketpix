"""Greedy search for per-layer body thresholds (export.py TAUL): raise one layer at a time, taking the step with the
most hidden units dropped per KL, while the model stays within the KL and bits limits against its plain threshold.
    PYTHONPATH=.:<repo> python3 pix/tausearch.py <name> <its number in the blob> <layers> <plain tau> <KL limit, e.g. 0.00018>
(counts include the head's units, which the body thresholds do not change)"""
import subprocess, os, sys, re
name, mi, L, t0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
KLMAX, BITMAX = float(sys.argv[5]), 0.0010
def ev(taus):
    env = dict(os.environ, TAUL=','.join(map(str, taus)))
    o = subprocess.run(["python3", "pix/qc.py", f"models={name}"], env=env, capture_output=True, text=True).stdout
    m = re.search(r"bits [\d.]+ \(([-+][\d.]+)\)\s+KL [\d.]+ \(([-+][\d.]+)\).*active (\d+)", o)
    return float(m.group(1)), float(m.group(2)), int(m.group(3)) // 1000
cur = [t0] * L; b0, k0, a0 = ev(cur); print("start", cur, b0, k0, a0, flush=True)
kl, act = k0, a0; dead = set()
while True:
    best = None
    for l in range(L):
        if l in dead: continue
        t = cur[:]; t[l] = t[l] * 3 // 2 if t[l] >= 4 * t0 else t[l] * 2
        b, k, a = ev(t)
        if k > KLMAX or b > BITMAX: dead.add(l); continue
        gain = (act - a) / max(k - kl, 2e-6)
        if best is None or gain > best[0]: best = (gain, t, b, k, a)
    if best is None: break
    _, cur, b, kl, act = best
    print("step", cur, f"bits {b:+.4f} KL {kl:+.5f} body {act}k ({100 * (act - a0) / a0:+.1f}%)", flush=True)
print("FINAL", name, cur)
