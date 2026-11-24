"""Reference numbers for the browser engine test: python3 -m img.web.ref <model.pt> <result.json> <out prefix> [seed]
Writes <prefix>.ref.json (picture, prompt, thumbnail) and <prefix>.ref.f32 (log-probabilities, cells x slots x P)."""
import json, sys, numpy as np, torch
from img.model import ImgConfig, ImgModel
pt, js, out = sys.argv[1:4]; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
cfg = ImgConfig(**json.load(open(js))["config"])
m = ImgModel(cfg); m.load_state_dict(torch.load(pt, map_location="cpu")); m.eval()
g = torch.Generator().manual_seed(seed)
img = torch.randint(0, cfg.P, (1, cfg.S * cfg.S), generator=g)
img = (img // 16 * 16 * 0 + torch.randint(0, 6, (1, 1), generator=g) * 16 + img % 48).clamp(max=255)     # a few nearby groups, like a real picture
cls = torch.tensor([[14, 5, 300, 0, 0, 0, 0, 0]]) if cfg.n_words else None
if seed % 2 and cls is not None: cls = torch.zeros_like(cls)
th = torch.randint(0, cfg.P, (1, (cfg.S // cfg.F) ** 2), generator=g) if cfg.F else None
with torch.no_grad(): lp, cells, _ = m(img, cls, th)
json.dump({"img": img[0].tolist(), "cls": cls[0].tolist() if cls is not None else [], "thumb": th[0].tolist() if th is not None else None}, open(out + ".ref.json", "w"))
lp[0].numpy().astype("<f4").tofile(out + ".ref.f32"); print(out, lp.shape)
