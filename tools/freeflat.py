"""Free sampling of the first pass: the PyTorch model (guided the engine's way: guidance for the first 11/16 of the
cells, the engine's temperature) against the twin under several settings. Statistics of the 16x16 pictures: share of
equal neighbours, distinct colours, and the cells the 32x32 pass would paint flat from them (twin, force mode).
    PYTHONPATH=. python3 tools/freeflat.py [name=b16C] [n=48] [words=288,10] "label ENV=v ..." ...
PyTorch pictures are cached in build/ff_<name>_<words>_<n>.npy."""
import os, sys, json, subprocess, numpy as np
o = dict(a.split('=', 1) for a in sys.argv[1:] if ' ' not in a and '=' in a)
sets = [a for a in sys.argv[1:] if ' ' in a or '=' not in a]
name = o.get('name', 'b16C'); N = int(o.get('n', 48)); words = [int(w) for w in o.get('words', '288,10').split(',') if w]
guide, invt = 768, 284
fn = f"build/ff_{name}_{'_'.join(map(str, words))}_{N}.npy"
if not os.path.exists(fn):
    import torch
    from img.model import ImgConfig, ImgModel
    torch.manual_seed(int(o.get('seed', 1)))
    pal = torch.from_numpy(np.load('img/remote/pal/p16o.npy')).float() / 255
    cfg = ImgConfig(**json.load(open(f'img/remote/night/{name}.json'))['config'])
    m = ImgModel(cfg, pal=pal); m.load_state_dict(torch.load(f'img/remote/night/{name}.pt', map_location='cpu')); m.eval()
    cls = torch.zeros(N, 8, dtype=torch.long); cls[:, :len(words)] = torch.tensor(words)
    img = torch.zeros(N, 256, dtype=torch.long); gq = guide / 256; temp = 256 / invt; cut = 256 * 11 // 16
    with torch.no_grad():
        for t in range(256):
            h = m.h_in(m.body(m.cells(img), cls, None)[0][:, t])
            f = m.head_feats(img, m.cells(img), t, None)[:, 0]
            g, mm = m.head_scores(h + f)
            if t < cut:
                h0 = m.h_in(m.body(m.cells(img), torch.zeros_like(cls), None)[0][:, t])
                g0, m0 = m.head_scores(h0 + f)
                g = g0.log_softmax(-1) + gq * (g.log_softmax(-1) - g0.log_softmax(-1))
                mm = m0.log_softmax(-1) + gq * (mm.log_softmax(-1) - m0.log_softmax(-1))
            gi = torch.multinomial((g / temp).softmax(-1), 1)[:, 0]
            mi = torch.multinomial((mm[torch.arange(N), gi] / temp).softmax(-1), 1)[:, 0]
            img[:, m.perm[t, 0]] = gi * 16 + mi
    np.save(fn, img.numpy().astype(np.uint8))
def stats(P):
    P = P.reshape(-1, 16, 16).astype(np.int64)
    eq = ((P[:, 1:, :] == P[:, :-1, :]).mean((1, 2)) + (P[:, :, 1:] == P[:, :, :-1]).mean((1, 2))) / 2
    nd = np.array([len(np.unique(p)) for p in P])
    fl = []
    for p in P:          # cells of the 32x32 pass painted flat from this thumbnail
        ref = np.zeros(8, '<u2').tobytes() + bytes(1024) + p.astype(np.uint8).tobytes()
        f = f'build/ff_{os.getpid()}.bin'; open(f, 'wb').write(ref)
        r = subprocess.run(['build/pix_twin', 'build/pix.bin', 'force', '1', f], capture_output=True, env=dict(os.environ, W1X='0'))
        raw = np.frombuffer(r.stdout, np.int32).reshape(-1, 272); fl.append((raw[:, 0] == -0x7FFFFFFF).sum() / 4)
    fl = np.array(fl); se = lambda a: a.std(ddof=1) / np.sqrt(len(a))
    return f"equal neighbours {eq.mean():.4f}+-{se(eq):.4f}  colours {nd.mean():.1f}+-{se(nd):.1f}  flat cells (32 pass) {fl.mean():.1f}+-{se(fl):.1f}"
print(f"{name} {words} PyTorch     {stats(np.load(fn))}", flush=True)
for st in sets:
    lab, *env = st.split(); e = dict(os.environ); e.update(dict(x.split('=', 1) for x in env))
    mode = 'paint3' if name == 'b16O' else 'paint'
    pics = []
    for s0 in range(4):       # 4 runs of N/4 in parallel
        pass
    import concurrent.futures as cf
    def run(k):
        f = f'build/ffd_{os.getpid()}_{k}.bin'; e2 = dict(e, PIXDUMP=f)
        subprocess.run(['build/pix_twin', 'build/pix.bin', mode, str(1000 + 77 * k), str(guide), str(invt), str(N // 4), *map(str, words)], capture_output=True, env=e2)
        d = np.fromfile(f, np.uint8).reshape(-1, 256 + 1024 + 4096); os.remove(f); return d[:, :256]
    with cf.ThreadPoolExecutor(4) as ex: pics = np.concatenate(list(ex.map(run, range(4))))
    print(f"{name} {words} {lab:11s} {stats(pics)}", flush=True)
