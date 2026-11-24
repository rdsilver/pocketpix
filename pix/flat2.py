"""How much more could be painted directly if 'flat' allowed nearly-equal colours? For each cell whose thumbnail window
stays within T (largest RGB channel difference, 0-255) of the window's first block: how often, and how far is a plain
copy of each pixel's own block colour from what the model painted (test pictures were painted by the model)."""
import json, numpy as np, torch
from img.model import ImgConfig, ImgModel
palb = np.load("img/remote/pal/p16o.npy").astype(np.int64)
pal = torch.from_numpy(palb).float() / 255
full = torch.from_numpy(np.fromfile("build/pix_test48.u8", np.uint8).reshape(-1, 4096).astype(np.int64))
def shrink(img, S, f):
    T = S // f; rgb = pal[img].view(-1, T, f, T, f, 3).mean((2, 4)).view(-1, T * T, 3)
    return torch.cdist(rgb, pal[None].expand(len(rgb), -1, -1)).argmin(-1)
P8 = torch.from_numpy(palb)
for name in ["u32M", "u64M"]:
    cfg = ImgConfig(**json.load(open(f"img/remote/night/{name}.json"))["config"])
    m = ImgModel(cfg, pal=pal)
    img = full if cfg.S == 64 else shrink(full, 64, 2)
    th = m.thumb_of(img)
    tv = torch.nn.functional.pad(th, (0, 1), value=0)[:, m.tw]; valid = (m.tw >= 0)[None].expand(len(img), -1, -1)
    rgb = P8[tv]                                                # N, cells, win, 3
    first = rgb.gather(2, valid.float().argmax(2, keepdim=True)[..., None].expand(-1, -1, 1, 3))
    dist = ((rgb - first).abs().max(3).values * valid).max(2).values        # N, cells
    cells = m.cells(img); own = torch.nn.functional.pad(th, (0, 1), value=0)[:, m.tp[..., 0]]   # N, cells, slots
    err = (P8[cells] - P8[own]).abs().float()                   # N, cells, slots, 3
    for T in (0, 8, 16, 24, 32, 48):
        f = dist <= T
        e = err[f]
        print(f"{name} T={T:2d}: cells {100 * f.float().mean():5.1f}%  mean |rgb error| of a plain copy {e.mean():.2f}  share of pixels off by more than 16: {100 * (e.max(-1).values > 16).float().mean():.1f}%")
