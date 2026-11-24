import json, numpy as np, torch
from img.model import ImgConfig, ImgModel
pal = torch.from_numpy(np.load("img/remote/pal/p16o.npy")).float() / 255
full = torch.from_numpy(np.fromfile("build/pix_test48.u8", np.uint8).reshape(-1, 4096).astype(np.int64))
def shrink(img, S, f):
    T = S // f; rgb = pal[img].view(-1, T, f, T, f, 3).mean((2, 4)).view(-1, T * T, 3)
    return torch.cdist(rgb, pal[None].expand(len(rgb), -1, -1)).argmin(-1)
for name in ["u32M", "u64M"]:
    cfg = ImgConfig(**json.load(open(f"img/remote/night/{name}.json"))["config"])
    m = ImgModel(cfg, pal=pal); m.load_state_dict(torch.load(f"img/remote/night/{name}.pt", map_location="cpu")); m.eval()
    img = full if cfg.S == 64 else shrink(full, 64, 2)
    th = m.thumb_of(img)
    tv = torch.nn.functional.pad(th, (0, 1), value=-1)[:, m.tw]
    valid = tv >= 0; first = tv.gather(2, valid.float().argmax(2, keepdim=True))
    flat = ((tv == first) | ~valid).all(2)
    cells = m.cells(img)
    with torch.no_grad():
        lp = torch.cat([m(img[i:i+8], torch.zeros(len(img[i:i+8]), 8, dtype=torch.long), th[i:i+8])[0] for i in range(0, len(img), 8)])
    c = first.squeeze(2)
    pc = lp.gather(3, c[:, :, None, None].expand(-1, -1, cfg.n_slots, 1)).squeeze(3).exp()
    same = (cells == c[:, :, None]).float()
    print(name, "flat-window cells: %.1f%%" % (100 * flat.float().mean()), "| per picture min/max %.0f%%/%.0f%%" % (100 * flat.float().mean(1).min(), 100 * flat.float().mean(1).max()))
    f = flat
    print("   in those: model prob of that colour: mean %.3f, share of pixels with p>0.9: %.3f, p>0.99: %.3f; actual pixels equal to it: %.3f" % (pc[f].mean(), (pc[f] > .9).float().mean(), (pc[f] > .99).float().mean(), same[f].mean()))
    tp = torch.nn.functional.pad(th, (0, 1), value=-1)[:, m.tp]
    v = tp >= 0; own = tp[..., :1]
    pf = ((tp == own) | ~v).all(3)
    pc2 = lp.gather(3, own).squeeze(3).exp()
    print("   pixel-level (own block and 4 neighbours same): %.1f%% of pixels; prob mean %.3f; p>0.9: %.3f" % (100 * pf.float().mean(), pc2[pf].mean(), (pc2[pf] > .9).float().mean()))
    pmax = lp.max(3).values.exp()
    print("   all pixels: share where the model's top colour has p>0.9: %.3f, >0.99: %.3f, >0.999: %.3f" % ((pmax > .9).float().mean(), (pmax > .99).float().mean(), (pmax > .999).float().mean()))
