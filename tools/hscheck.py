"""pix.check's numbers for u64M with and without PIX_HEADSKIP on the SAME pixels (those both versions evaluate), and a
check that every kept row of scores is the same to the bit.   PYTHONPATH=. python3 tools/hscheck.py"""
import os, json, subprocess, numpy as np, torch
from img.model import ImgConfig, ImgModel
pal = torch.from_numpy(np.load("img/remote/pal/p16o.npy")).float() / 255
full = torch.from_numpy(np.fromfile("build/pix_test48.u8", np.uint8).reshape(-1, 4096).astype(np.int64))
name, mi = "u64M", 2
cfg = ImgConfig(**json.load(open(f"img/remote/night/{name}.json"))["config"])
m = ImgModel(cfg, pal=pal); m.load_state_dict(torch.load(f"img/remote/night/{name}.pt", map_location="cpu")); m.eval()
tot = {0: np.zeros(4), 1: np.zeros(4)}; same = True; nk = [0, 0, 0]
for k in range(len(full)):
    img = full[k:k + 1]
    cls = torch.tensor([[14, 5, 300, 0, 0, 0, 0, 0]]) if k % 2 == 0 else torch.zeros(1, 8, dtype=torch.long)
    th = m.thumb_of(img)
    with torch.no_grad(): lp = m(img, cls, th)[0][0].numpy().astype(np.float64).reshape(-1, 256)
    open("build/pix_ref.bin", "wb").write(cls[0].numpy().astype("<u2").tobytes() + img[0].numpy().astype(np.uint8).tobytes() + th[0].numpy().astype(np.uint8).tobytes())
    raws = {hs: np.frombuffer(subprocess.run(["build/pix_twin", "build/pix.bin", "force", str(mi), "build/pix_ref.bin"], capture_output=True, check=True,
                                             env=dict(os.environ, HEADSKIP=str(hs))).stdout, np.int32).reshape(-1, 272) for hs in (0, 1)}
    keep = (raws[0][:, 0] != -0x7FFFFFFF) & (raws[1][:, 0] != -0x7FFFFFFF)
    same &= bool((raws[0][keep] == raws[1][keep]).all())
    nk[0] += keep.sum(); nk[1] += (raws[0][:, 0] != -0x7FFFFFFF).sum(); nk[2] += len(keep)
    px = m.cells(img)[0].reshape(-1).numpy(); ar = np.arange(len(px))
    for hs in (0, 1):
        s = raws[hs].astype(np.float64) / 4096; s[~keep] = 0
        lg = s[:, :16] - np.log(np.exp(s[:, :16]).sum(1, keepdims=True)); mm = s[:, 16:].reshape(-1, 16, 16)
        li = (lg[:, :, None] + mm - np.log(np.exp(mm).sum(2, keepdims=True))).reshape(-1, 256)
        tot[hs] += [-lp[ar, px][keep].mean() / np.log(2), -li[ar, px][keep].mean() / np.log(2), (np.exp(lp) * (lp - li)).sum(1)[keep].mean(), 1]
print(f"pixels evaluated by both: {nk[0]} of {nk[2]} (without PIX_HEADSKIP: {nk[1]}); kept rows identical: {same}")
for hs in (0, 1):
    t = tot[hs] / tot[hs][3]
    print(f"HEADSKIP={hs}: bits/pixel float {t[0]:.4f} integer {t[1]:.4f}  KL {t[2]:.5f}")
