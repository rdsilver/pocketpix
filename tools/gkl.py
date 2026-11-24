"""Guided teacher-forced quality (what pix/check.py does not reach), several engine settings against one baseline.
Like tools/w1kl2.py: the engine's guided scores (twin, FORCE_GUIDE) against the PyTorch model guided the same way,
for the cells that run with guidance. The PyTorch side is cached in build/gkl_<name>_<n>_<prompts>_<guide>.npz.
    PYTHONPATH=. python3 tools/gkl.py [name=b16C] [n=48] [prompts=4] [guide=768] "label ENV=v ENV=v" ...
First setting is the baseline; each other line prints its rise in bits/pixel and KL with standard errors (paired)."""
import os, json, subprocess, sys, numpy as np
from concurrent.futures import ThreadPoolExecutor
o = dict(a.split('=', 1) for a in sys.argv[1:] if ' ' not in a and '=' in a)
sets = [a for a in sys.argv[1:] if ' ' in a or '=' not in a]
name = o.get('name', 'b16C'); N = int(o.get('n', 48)); R = int(o.get('prompts', 4)); guide = int(o.get('guide', 768))
ALL = ["b16C", "u32M", "u64M", "b16O"]; model = ALL.index(name)
def lsm(a, ax): mx = a.max(ax, keepdims=True); return a - np.log(np.exp(a - mx).sum(ax, keepdims=True)) - mx
def split(lp):
    j = lp.reshape(-1, 16, 16); mx = j.max(2, keepdims=True); lg = (np.log(np.exp(j - mx).sum(2, keepdims=True)) + mx)
    return lg[:, :, 0], j - lg
def joint(g, mm): return (lsm(g, 1)[:, :, None] + lsm(mm, 2)).reshape(-1, 256)
words = [[14, 5, 300], [288, 10], [500, 77], [901, 33, 7]]
fn = f'build/gkl_{name}_{N}_{R}_{guide}.npz'
if not os.path.exists(fn):
    import torch
    from img.model import ImgConfig, ImgModel
    pal = torch.from_numpy(np.load('img/remote/pal/p16o.npy')).float() / 255
    full = torch.from_numpy(np.fromfile('build/pix_test48.u8', np.uint8).reshape(-1, 4096).astype(np.int64))
    cfg = ImgConfig(**json.load(open(f'img/remote/night/{name}.json'))['config'])
    m = ImgModel(cfg, pal=pal); m.load_state_dict(torch.load(f'img/remote/night/{name}.pt', map_location='cpu')); m.eval()
    def shrink(img, S, f):
        T = S // f; rgb = pal[img].view(-1, T, f, T, f, 3).mean((2, 4)).view(-1, T * T, 3)
        return torch.cdist(rgb, pal[None].expand(len(rgb), -1, -1)).argmin(-1)
    gq = guide / 256; refs, rbs, pxs = [], [], []
    for k in range(min(N, len(full)) * R):
        img = shrink(full[k // R:k // R + 1], 64, 64 // cfg.S); ww = words[(k // R + k % R) % len(words)]
        cls = torch.zeros(1, 8, dtype=torch.long); cls[0, :len(ww)] = torch.tensor(ww)
        with torch.no_grad():
            lc = m(img, cls, None)[0][0].numpy().astype(np.float64).reshape(-1, 256); lu = m(img, torch.zeros(1, 8, dtype=torch.long), None)[0][0].numpy().astype(np.float64).reshape(-1, 256)
        gc, mc = split(lc); gu, mu = split(lu); refs.append(joint(gu + gq * (gc - gu), mu + gq * (mc - mu)).astype(np.float32))
        rbs.append(np.frombuffer(cls[0].numpy().astype('<u2').tobytes() + img[0].numpy().astype(np.uint8).tobytes(), np.uint8)); pxs.append(m.cells(img)[0].reshape(-1).numpy())
    np.savez(fn, ref=np.array(refs), rb=np.array(rbs), px=np.array(pxs))
c = np.load(fn); REF, RB, PX = c['ref'], c['rb'], c['px']
def engine(args):
    k, env = args
    f = f'build/gkl_{os.getpid()}_{k}_{abs(hash(env)) % 100000}.bin'; open(f, 'wb').write(RB[k].tobytes())
    e = dict(os.environ, FORCE_GUIDE=str(guide)); e.update(dict(x.split('=', 1) for x in env.split()))
    r = subprocess.run([o.get('twin', 'build/pix_twin'), 'build/pix.bin', 'force', str(model), f], capture_output=True, check=True, env=e); os.remove(f)
    s = np.frombuffer(r.stdout, np.int32).reshape(-1, 272).astype(np.float64) / 4096
    return joint(s[:, :16], s[:, 16:].reshape(-1, 16, 16))
res = []
for st in sets:
    lab, *env = st.split(); env = ' '.join(env)
    with ThreadPoolExecutor(4) as ex: outs = list(ex.map(engine, [(k, env) for k in range(len(REF))]))
    v = []
    for k, a in enumerate(outs):
        nc = len(a); ref = REF[k][:nc].astype(np.float64); px = PX[k][:nc]; ar = np.arange(nc)
        v.append([-a[ar, px].mean() / np.log(2), (np.exp(ref) * (ref - a)).sum(1).mean(), -(np.exp(a) * a).sum(1).mean(), -(np.exp(ref) * ref).sum(1).mean()])
    res.append((lab, np.array(v), outs))
se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
b = res[0][1]
for lab, v, outs in res:
    kl0 = np.mean([(np.exp(a0) * (a0 - a)).sum(1).mean() for a0, a in zip(res[0][2], outs)])
    print(f'{name} {lab:12s} bits {v[:, 0].mean():.4f} ({(v[:, 0] - b[:, 0]).mean():+.4f} +- {se(v[:, 0] - b[:, 0]) if lab != res[0][0] else 0:.4f})  KL {v[:, 1].mean():.5f} ({(v[:, 1] - b[:, 1]).mean():+.5f} +- {se(v[:, 1] - b[:, 1]) if lab != res[0][0] else 0:.5f})  KL to baseline engine {kl0:.5f}  entropy {v[:, 2].mean():.4f} (model {v[:, 3].mean():.4f})', flush=True)
