"""Integer engine (build/pix_twin) against the PyTorch model, teacher-forced on one picture per model.
    python3 -m pix.check b16C u32M u64M [pictures.u8]   (names in blob order; optional 64x64 pictures to test on)"""
import os, json, subprocess, sys, numpy as np, torch
from img.model import ImgConfig, ImgModel
names = [a for a in sys.argv[1:] if not a.endswith(".u8")]; pics = [a for a in sys.argv[1:] if a.endswith(".u8")]
pal = torch.from_numpy(np.load("img/remote/pal/p16o.npy")).float() / 255
full = torch.from_numpy(np.fromfile(pics[0], np.uint8).reshape(-1, 4096).astype(np.int64)) if pics else None
def shrink(img, S, f):       # (N, S*S) -> (N, (S/f)^2), the way training made thumbnails
    T = S // f; rgb = pal[img].view(-1, T, f, T, f, 3).mean((2, 4)).view(-1, T * T, 3)
    return torch.cdist(rgb, pal[None].expand(len(rgb), -1, -1)).argmin(-1)
for mi, name in enumerate(names):
    if os.environ.get("ONLY") and name != os.environ["ONLY"]: continue      # ONLY=b16O: just that model (names still in blob order)
    cfg = ImgConfig(**json.load(open(f"img/remote/night/{name}.json"))["config"])
    m = ImgModel(cfg, pal=pal); m.load_state_dict(torch.load(f"img/remote/night/{name}.pt", map_location="cpu")); m.eval()
    tot = np.zeros(4); n = 0
    for k in range(len(full) if full is not None else 2):
        if full is not None:
            img = full[k:k + 1]
            if cfg.S < 64: img = shrink(img, 64, 64 // cfg.S)
        else:
            g = torch.Generator().manual_seed(k)
            img = (torch.randint(0, 6, (1, 1), generator=g) * 16 + torch.randint(0, 48, (1, cfg.S ** 2), generator=g)).clamp(max=255)
        cls = torch.tensor([[14, 5, 300, 0, 0, 0, 0, 0]]) if k % 2 == 0 else torch.zeros(1, 8, dtype=torch.long)
        th = m.thumb_of(img) if cfg.F else None
        with torch.no_grad(): lp = m(img, cls, th)[0][0].numpy().astype(np.float64)          # cells, slots, 256
        ref = cls[0].numpy().astype("<u2").tobytes() + img[0].numpy().astype(np.uint8).tobytes() + (th[0].numpy().astype(np.uint8).tobytes() if th is not None else b"")
        open("build/pix_ref.bin", "wb").write(ref)
        r = subprocess.run(["build/pix_twin", "build/pix.bin", "force", str(mi), "build/pix_ref.bin"], capture_output=True, check=True)
        raw = np.frombuffer(r.stdout, np.int32).reshape(-1, 272); keep = raw[:, 0] != -0x7FFFFFFF
        s = raw.astype(np.float64) / 4096; s[~keep] = 0
        lg = s[:, :16] - np.log(np.exp(s[:, :16]).sum(1, keepdims=True)); mm = s[:, 16:].reshape(-1, 16, 16)
        lm = mm - np.log(np.exp(mm).sum(2, keepdims=True))
        li = (lg[:, :, None] + lm).reshape(-1, 256); lp = lp.reshape(-1, 256)
        px = m.cells(img)[0].reshape(-1).numpy(); ar = np.arange(len(px))
        tot += [-lp[ar, px][keep].mean() / np.log(2), -li[ar, px][keep].mean() / np.log(2), (np.exp(lp) * (lp - li)).sum(1)[keep].mean(), (lp.argmax(1) == li.argmax(1))[keep].mean()]; n += 1; skipn = skipn + (~keep).mean() if 'skipn' in dir() else (~keep).mean()
        note = r.stderr.decode().strip()
    t = tot / n
    print(f"skipped {100 * skipn / n:.1f}% ", end=""); del skipn
    print(f"{name}: bits/pixel float {t[0]:.4f} integer {t[1]:.4f}  KL {t[2]:.5f}  same top colour {t[3] * 100:.1f}%  | {note}")
