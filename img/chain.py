"""Draw pictures through a chain of trained runs, thumbnail first: python3 -m img.chain <out.png> <run> <run> ... [temp=1.0] [n=96]
The first run draws a small picture from nothing; each later run (trained with F > 0) draws a larger one over it."""
import json, sys, torch, numpy as np
from pathlib import Path
from .model import ImgConfig, ImgModel
from .data import Pictures
from .train import route_counts

args = [a for a in sys.argv[2:] if "=" not in a]
opt = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
temp, n = float(opt.get("temp", 1.0)), int(opt.get("n", 96))
temps = [float(t) for t in opt["temps"].split(",")] if "temps" in opt else [temp] * len(args)      # one per run
gen = torch.Generator(device="cuda"); gen.manual_seed(int(opt.get("seed", 1234)))
x, sheets = None, []
for k, r in enumerate(args):
    run = Path("img/runs") / r
    cfg = ImgConfig(**json.load(open(run / "result.json"))["config"])
    data = Pictures(cfg.data)
    counts = route_counts(ImgModel(ImgConfig(D=32, L=2, DK=32, DV=16, E=2, HD=16, DH=16, HH=16, att_layers=[1], S=cfg.S, C=cfg.C)), data, cfg)
    m = ImgModel(cfg, counts, data.pal / 255.0).cuda()
    m.load_state_dict(torch.load(run / ("best.pt" if (run / "best.pt").exists() else "model.pt"), map_location="cpu")); m.eval()
    if cfg.F and x is None:                           # seed mode: start from stored small sketches of training pictures that carry the prompt's words
        tl = data.tl_all; ok = np.ones(len(tl), bool); ixs = {w: i for i, w in enumerate(data.words)}
        for w in [w for w in opt.get("words", "").split(",") if w in ixs]: ok &= (tl == ixs[w]).any(1)
        rows = np.nonzero(ok)[0]; print("seed pictures to choose from:", len(rows), flush=True)
        rows = np.sort(np.random.default_rng(int(opt.get("seed", 1234))).choice(rows, n, replace=len(rows) < n))
        pal = torch.from_numpy(data.pal).float().cuda() / 255; T = cfg.S // cfg.F
        full = torch.from_numpy(data.train[rows].astype(np.int64)).cuda()
        x = torch.cdist(pal[full].view(n, T, cfg.F, T, cfg.F, 3).mean((2, 4)).view(n, T * T, 3), pal[None].expand(n, -1, -1)).argmin(-1)
        sheets.append(data.sheet(np.repeat(np.repeat(x.view(n, T, T).cpu().numpy(), cfg.F, 1), cfg.F, 2).reshape(n, -1), cols=int(opt.get("cols", 16)), scale=max(2, 96 // cfg.S)))
    assert bool(cfg.F) == (x is not None), "first run must have no thumbnail, later ones must"
    cl = None
    if cfg.n_words:                                   # prompts: held-out captions, or words=a,b,c for every picture
        if "words" in opt:
            ix = {w: i for i, w in enumerate(data.words)}
            cl = torch.zeros(n, cfg.W, dtype=torch.long)
            for k, w in enumerate([w for w in opt["words"].split(",") if w in ix][:cfg.W]): cl[:, k] = ix[w]   # unknown words are skipped
        else:
            cl = torch.from_numpy(data.vl[:n].astype(np.int64))
            open(sys.argv[1] + ".txt", "w").write("\n".join(f"{i // 16 + 1},{i % 16 + 1}: {c}" for i, c in enumerate(data.val_caps[:n])))
        cl = cl.cuda()
    x = m.sample(n, "cuda", cl, temp=temps[k], gen=gen, thumb=x, guide=float(opt.get("guide", 1.0)))
    sheets.append(data.sheet(x.cpu().numpy(), cols=int(opt.get("cols", 16)), scale=max(2, 96 // cfg.S)))
    print(r, "drawn", flush=True)
from PIL import Image
im = Image.new("RGB", (max(s.width for s in sheets), sum(s.height for s in sheets) + 8 * (len(sheets) - 1)), (16, 19, 28))
y = 0
for s in sheets: im.paste(s, (0, y)); y += s.height + 8
im.save(sys.argv[1])
