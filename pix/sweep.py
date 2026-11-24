"""Quality sweeps: like pix.check, but the float model's answers are worked out once and the twin is run for many settings.
    python3 -m pix.sweep <model name> <pictures.u8> <file with one setting per line: label ENV=val ENV=val ...>"""
import json, subprocess, sys, os, numpy as np, torch, concurrent.futures as cf
from img.model import ImgConfig, ImgModel
name, pics, cfgf = sys.argv[1:4]
lay = open('gen/pix_layout.h').read(); names = lay.split('/* models: ')[1].split(' */')[0].split(); mi = names.index(name)
pal = torch.from_numpy(np.load("img/remote/pal/p16o.npy")).float() / 255
full = torch.from_numpy(np.fromfile(pics, np.uint8).reshape(-1, 4096).astype(np.int64))
NP = int(os.environ.get("NPIC", len(full)))
def shrink(img, S, f):
    T = S // f; rgb = pal[img].view(-1, T, f, T, f, 3).mean((2, 4)).view(-1, T * T, 3)
    return torch.cdist(rgb, pal[None].expand(len(rgb), -1, -1)).argmin(-1)
cfg = ImgConfig(**json.load(open(f"img/remote/night/{name}.json"))["config"])
m = ImgModel(cfg, pal=pal); m.load_state_dict(torch.load(f"img/remote/night/{name}.pt", map_location="cpu")); m.eval()
import tempfile; tag = os.path.join(tempfile.gettempdir(), f"pix_sweep_{name}_{os.getpid()}")
data = []
for k in range(NP):
    img = full[k:k + 1]
    if cfg.S < 64: img = shrink(img, 64, 64 // cfg.S)
    cls = torch.tensor([[14, 5, 300, 0, 0, 0, 0, 0]]) if k % 2 == 0 else torch.zeros(1, 8, dtype=torch.long)
    th = m.thumb_of(img) if cfg.F else None
    with torch.no_grad(): lp = m(img, cls, th)[0][0].numpy().astype(np.float64).reshape(-1, 256)
    ref = cls[0].numpy().astype("<u2").tobytes() + img[0].numpy().astype(np.uint8).tobytes() + (th[0].numpy().astype(np.uint8).tobytes() if th is not None else b"")
    open(f"{tag}_{k}.bin", "wb").write(ref)
    data.append((lp, m.cells(img)[0].reshape(-1).numpy()))
def one(k, env):
    r = subprocess.run([os.environ.get("TWIN", "build/pix_twin"), "build/pix.bin", "force", str(mi), f"{tag}_{k}.bin"], capture_output=True, check=True, env={**os.environ, **env})
    raw = np.frombuffer(r.stdout, np.int32).reshape(-1, 272); keep = raw[:, 0] != -0x7FFFFFFF
    s = raw.astype(np.float64) / 4096; s[~keep] = 0
    lg = s[:, :16] - np.log(np.exp(s[:, :16]).sum(1, keepdims=True)); mm = s[:, 16:].reshape(-1, 16, 16)
    lm = mm - np.log(np.exp(mm).sum(2, keepdims=True)); li = (lg[:, :, None] + lm).reshape(-1, 256)
    lp, px = data[k]; ar = np.arange(len(px))
    return np.array([-li[ar, px][keep].mean() / np.log(2), (np.exp(lp) * (lp - li)).sum(1)[keep].mean()])
base = None
with cf.ThreadPoolExecutor(int(os.environ.get("JOBS", 3))) as ex:
    for line in open(cfgf):
        w = line.split()
        if not w: continue
        env = dict(a.split('=', 1) for a in w[1:])
        t = np.mean(list(ex.map(lambda k: one(k, env), range(NP))), 0)
        if base is None: base = t
        print(f"{name} {w[0]:24s} bits {t[0]:.4f} ({t[0] - base[0]:+.4f})  KL {t[1]:.5f} ({t[1] - base[1]:+.5f})", flush=True)
for k in range(NP): os.remove(f"{tag}_{k}.bin")
