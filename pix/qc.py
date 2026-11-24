"""Fast version of pix/check.py (same numbers; the differences in brackets are against the engine before the per-layer
thresholds, BASE below): float log-probs cached in build/qc_cache_<name>.npz; the twin runs in parallel.
   python3 pix/qc.py [models=b16C,u32M,u64M] [twin=build/pix_twin] [blob=build/pix.bin] ; env passes through to the twin"""
import json, subprocess, sys, os, numpy as np
from concurrent.futures import ThreadPoolExecutor
o = dict(a.split('=') for a in sys.argv[1:])
ALL = ["b16C", "u32M", "u64M", "b16O"]
names = o.get('models', 'b16C,u32M,u64M').split(','); twin = o.get('twin', 'build/pix_twin'); blob = o.get('blob', 'build/pix.bin')
BASE = {"b16C": (2.4993, 0.00074), "u32M": (1.2332, 0.00241), "u64M": (0.9174, 0.00262), "b16O": (2.52771, 0.000582)}
def cache(name):
    fn = f"build/qc_cache_{name}.npz"
    if os.path.exists(fn): return np.load(fn)
    import torch
    from img.model import ImgConfig, ImgModel
    pal = torch.from_numpy(np.load("img/remote/pal/p16o.npy")).float() / 255
    full = torch.from_numpy(np.fromfile("build/pix_test48.u8", np.uint8).reshape(-1, 4096).astype(np.int64))
    def shrink(img, S, f):
        T = S // f; rgb = pal[img].view(-1, T, f, T, f, 3).mean((2, 4)).view(-1, T * T, 3)
        return torch.cdist(rgb, pal[None].expand(len(rgb), -1, -1)).argmin(-1)
    cfg = ImgConfig(**json.load(open(f"img/remote/night/{name}.json"))["config"])
    m = ImgModel(cfg, pal=pal); m.load_state_dict(torch.load(f"img/remote/night/{name}.pt", map_location="cpu")); m.eval()
    lps, refs, pxs = [], [], []
    for k in range(len(full)):
        img = full[k:k + 1]
        if cfg.S < 64: img = shrink(img, 64, 64 // cfg.S)
        cls = torch.tensor([[14, 5, 300, 0, 0, 0, 0, 0]]) if k % 2 == 0 else torch.zeros(1, 8, dtype=torch.long)
        th = m.thumb_of(img) if cfg.F else None
        with torch.no_grad(): lp = m(img, cls, th)[0][0].numpy().astype(np.float32)
        ref = cls[0].numpy().astype("<u2").tobytes() + img[0].numpy().astype(np.uint8).tobytes() + (th[0].numpy().astype(np.uint8).tobytes() if th is not None else b"")
        lps.append(lp.reshape(-1, 256)); refs.append(np.frombuffer(ref, np.uint8)); pxs.append(m.cells(img)[0].reshape(-1).numpy())
    np.savez(fn, lp=np.array(lps), ref=np.array(refs), px=np.array(pxs)); return np.load(fn)
def one(args):
    mi, k, ref = args
    fn = f"build/qc_ref_{os.getpid()}_{mi}_{k}.bin"; open(fn, "wb").write(ref.tobytes())
    r = subprocess.run([twin, blob, "force", str(mi), fn], capture_output=True, check=True); os.remove(fn)
    return np.frombuffer(r.stdout, np.int32).reshape(-1, 272), r.stderr.decode().strip()
res = {}
for name in names:
    mi = ALL.index(name); c = cache(name); LP, REF, PX = c["lp"], c["ref"], c["px"]
    with ThreadPoolExecutor(4) as ex: outs = list(ex.map(one, [(mi, k, REF[k]) for k in range(len(LP))]))
    tot = np.zeros(4); act = 0
    for k, (raw, note) in enumerate(outs):
        keep = raw[:, 0] != -0x7FFFFFFF
        s = raw.astype(np.float64) / 4096; s[~keep] = 0
        lg = s[:, :16] - np.log(np.exp(s[:, :16]).sum(1, keepdims=True)); mm = s[:, 16:].reshape(-1, 16, 16)
        lm = mm - np.log(np.exp(mm).sum(2, keepdims=True))
        li = (lg[:, :, None] + lm).reshape(-1, 256); lp = LP[k].astype(np.float64)
        px = PX[k]; ar = np.arange(len(px))
        tot += [-lp[ar, px][keep].mean() / np.log(2), -li[ar, px][keep].mean() / np.log(2), (np.exp(lp) * (lp - li)).sum(1)[keep].mean(), (~keep).mean()]
        act += int(note.split()[1])
    t = tot / len(outs); b = BASE.get(name, (t[1], t[2]))
    print(f"{name}: bits {t[1]:.4f} ({t[1]-b[0]:+.4f})  KL {t[2]:.5f} ({t[2]-b[1]:+.5f})  skipped {100*t[3]:.1f}%  active {act}", flush=True)
