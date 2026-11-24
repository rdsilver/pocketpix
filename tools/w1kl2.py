"""The first-pass model with guidance (two streams), which pix/check.py does not reach: teacher-forced on the test
pictures with a prompt, the engine's guided scores (twin, FORCE_GUIDE) against the PyTorch model's, guided the same
way (group scores and member scores each: empty prompt + guide * (prompt - empty prompt)); for the cells that run
with guidance. Prints bits per pixel and KL (nats) with and without the early exit (W1X), and the KL between the two.
    PYTHONPATH=. python3 tools/w1kl2.py [model=0] [n=48] [prompts=4] [guide=768] [name=b16C]   (+- : standard error over the runs)"""
import os, json, subprocess, sys, numpy as np, torch
from img.model import ImgConfig, ImgModel
o = dict(a.split('=') for a in sys.argv[1:]); model = int(o.get('model', 0)); N = int(o.get('n', 48)); guide = int(o.get('guide', 768))
name = o.get('name', open('gen/pix_layout.h').read().split('/* models: ')[1].split(' */')[0].split()[model])
pal = torch.from_numpy(np.load('img/remote/pal/p16o.npy')).float() / 255
full = torch.from_numpy(np.fromfile('build/pix_test48.u8', np.uint8).reshape(-1, 4096).astype(np.int64))
cfg = ImgConfig(**json.load(open(f'img/remote/night/{name}.json'))['config'])
m = ImgModel(cfg, pal=pal); m.load_state_dict(torch.load(f'img/remote/night/{name}.pt', map_location='cpu')); m.eval()
def shrink(img, S, f):
    T = S // f; rgb = pal[img].view(-1, T, f, T, f, 3).mean((2, 4)).view(-1, T * T, 3)
    return torch.cdist(rgb, pal[None].expand(len(rgb), -1, -1)).argmin(-1)
def lsm(a, ax): return a - np.log(np.exp(a - a.max(ax, keepdims=True)).sum(ax, keepdims=True)) - a.max(ax, keepdims=True)
def split(lp):                                               # joint log probabilities (n, 256) -> groups (n, 16), members (n, 16, 16)
    j = lp.reshape(-1, 16, 16); mx = j.max(2, keepdims=True); lg = (np.log(np.exp(j - mx).sum(2, keepdims=True)) + mx)
    return lg[:, :, 0], j - lg
def joint(g, mm): return (lsm(g, 1)[:, :, None] + lsm(mm, 2)).reshape(-1, 256)
def engine(ref, w1x):
    open('build/pix_ref2.bin', 'wb').write(ref)
    r = subprocess.run(['build/pix_twin', 'build/pix.bin', 'force', str(model), 'build/pix_ref2.bin'], capture_output=True, check=True, env=dict(os.environ, FORCE_GUIDE=str(guide), W1X=str(w1x)))
    s = np.frombuffer(r.stdout, np.int32).reshape(-1, 272).astype(np.float64) / 4096
    return joint(s[:, :16], s[:, 16:].reshape(-1, 16, 16))
words = [[14, 5, 300], [288, 10], [500, 77], [901, 33, 7]]
rowsv = []; gq = guide / 256; R = int(o.get('prompts', 1))          # prompts=4: every picture with each of the four prompts
for k in range(min(N, len(full)) * R):
    img = shrink(full[k // R:k // R + 1], 64, 64 // cfg.S); ww = words[(k // R + k % R) % len(words)]
    cls = torch.zeros(1, 8, dtype=torch.long); cls[0, :len(ww)] = torch.tensor(ww)
    with torch.no_grad():
        lc = m(img, cls, None)[0][0].numpy().astype(np.float64).reshape(-1, 256); lu = m(img, torch.zeros(1, 8, dtype=torch.long), None)[0][0].numpy().astype(np.float64).reshape(-1, 256)
    gc, mc = split(lc); gu, mu = split(lu); ref = joint(gu + gq * (gc - gu), mu + gq * (mc - mu))
    rb = cls[0].numpy().astype('<u2').tobytes() + img[0].numpy().astype(np.uint8).tobytes()
    a = engine(rb, 0); b = engine(rb, 1); nc = len(a); ref = ref[:nc]; px = m.cells(img)[0].reshape(-1).numpy()[:nc]; ar = np.arange(nc)
    kl = lambda p, q: (np.exp(p) * (p - q)).sum(1).mean()
    rowsv.append([-a[ar, px].mean() / np.log(2), -b[ar, px].mean() / np.log(2), kl(ref, a), kl(ref, b), kl(a, b)])
v = np.array(rowsv); t = v.mean(0); n = len(v); se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
print(f'{name} guided ({gq:g}), {n} runs, {nc} cells each: bits/pixel without {t[0]:.4f} with {t[1]:.4f} (rise {t[1] - t[0]:+.4f} +- {se(v[:, 1] - v[:, 0]):.4f}) | '
      f'KL to the model without {t[2]:.5f} with {t[3]:.5f} (rise {t[3] - t[2]:+.5f} +- {se(v[:, 3] - v[:, 2]):.5f}) | KL between them {t[4]:.5f}')
