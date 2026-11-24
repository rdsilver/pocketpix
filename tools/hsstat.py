"""Teacher-forced: the model's probability of the colour a flat or body-only (PIX_HEADSKIP) cell copies from the thumbnail.
    PYTHONPATH=. python3 tools/hsstat.py [u64M] [HSND=5 in env]   (48 test pictures, build/pix_test48.u8)"""
import os, json, subprocess, sys, numpy as np, torch
from img.model import ImgConfig, ImgModel
name = sys.argv[1] if len(sys.argv) > 1 else "u64M"; mi = {"b16C": 0, "u32M": 1, "u64M": 2, "b16O": 3}[name]
pal = torch.from_numpy(np.load("img/remote/pal/p16o.npy")).float() / 255
full = torch.from_numpy(np.fromfile("build/pix_test48.u8", np.uint8).reshape(-1, 4096).astype(np.int64))
def shrink(img, S, f):
    T = S // f; rgb = pal[img].view(-1, T, f, T, f, 3).mean((2, 4)).view(-1, T * T, 3)
    return torch.cdist(rgb, pal[None].expand(len(rgb), -1, -1)).argmin(-1)
cfg = ImgConfig(**json.load(open(f"img/remote/night/{name}.json"))["config"])
m = ImgModel(cfg, pal=pal); m.load_state_dict(torch.load(f"img/remote/night/{name}.pt", map_location="cpu")); m.eval()
acc = {}
for k in range(len(full)):
    img = full[k:k + 1]
    if cfg.S < 64: img = shrink(img, 64, 64 // cfg.S)
    cls = torch.tensor([[14, 5, 300, 0, 0, 0, 0, 0]]) if k % 2 == 0 else torch.zeros(1, 8, dtype=torch.long)
    th = m.thumb_of(img)
    with torch.no_grad(): lp = m(img, cls, th)[0][0].numpy().astype(np.float64).reshape(-1, 256)
    ref = cls[0].numpy().astype("<u2").tobytes() + img[0].numpy().astype(np.uint8).tobytes() + th[0].numpy().astype(np.uint8).tobytes()
    open("build/pix_ref.bin", "wb").write(ref)
    r = subprocess.run(["build/pix_twin", "build/pix.bin", "force", str(mi), "build/pix_ref.bin"], capture_output=True, check=True,
                       env=dict(os.environ, SHADOW="1", CLASSOUT="build/hsclass.bin"))
    raw = np.frombuffer(r.stdout, np.int32).reshape(-1, 272).astype(np.float64) / 4096
    lg = raw[:, :16] - np.log(np.exp(raw[:, :16]).sum(1, keepdims=True)); mm = raw[:, 16:].reshape(-1, 16, 16)
    li = (lg[:, :, None] + mm - np.log(np.exp(mm).sum(2, keepdims=True))).reshape(-1, 256)
    cb = np.fromfile("build/hsclass.bin", np.uint8); NC = 256; NS = cfg.n_slots
    ccls = cb[:NC]; cc = cb[NC:NC + NC * NS].reshape(NC, NS)[:NC].reshape(-1)
    pcl = np.repeat(ccls, NS)
    px = m.cells(img)[0].reshape(-1).numpy(); ar = np.arange(len(px))
    groups = {"window rule (skipped now)": (pcl & 1) == 1, "own rule only (body-only cells)": ((pcl & 3) == 2), "own rule, all": (pcl & 2) == 2, "model runs (after)": (pcl & 3) == 0, "all": pcl >= 0}
    for g, sel in groups.items():
        a = acc.setdefault(g, np.zeros(8))
        pf = np.exp(lp[ar, cc]); pi = np.exp(li[ar, cc])
        ent = -(np.exp(lp) * lp).sum(1)
        a += [sel.sum(), pf[sel].sum(), pi[sel].sum(), -lp[ar, cc][sel].sum(), (pf[sel] > .99).sum(), ent[sel].sum(), (px == cc)[sel].sum(), (pf[sel] < .5).sum()]
print(f"{name}, 48 test pictures, teacher-forced; p = the float model's probability of the copied colour")
for g, a in acc.items():
    n = max(a[0], 1)
    print(f"{g:34s} pixels {int(a[0]):7d} ({100 * a[0] / acc['all'][0]:4.1f}%)  mean p {a[1] / n:.4f} (integer {a[2] / n:.4f})  -ln p {a[3] / n:.4f}  p>.99 {a[4] / n:.3f}  p<.5 {a[7] / n:.3f}  entropy {a[5] / n:.4f}  true pixel = copy {a[6] / n:.3f}")
