"""Named objects: find photos whose caption says the thing is shown close up, train a judge on them, measure hit rates.
    python3 -m img.objs slice                      # -> img/data/objslice.npz, counts, a sheet of examples
    python3 -m img.objs clf                        # judges at 16/32/64 pixels, colour and grey -> img/runs/judge*.pt
    python3 -m img.objs enl <run32> <run64>        # do real 16x16 close-ups survive the enlargers?
    python3 -m img.objs hit <base> <run32> <run64> [guide=3] [n=64] [extra=closeup]   # prompted pictures, judged
Picture order: caption k of the cache is row j of train.u8 where perm[val + j] == k (the shuffle img.data used)."""
import json, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from .data import words_of
from .caps import split_caption

D16, D32, D64, CACHE, OUT = "img/data/p16s", "img/data/p32t", "img/data/photos64", "img/data/captions_cache.json", "img/data/objslice.npz"
NOUNS = """cat dog bird horse cow sheep elephant butterfly fish flower tree leaf car bus truck train boat airplane bicycle
motorcycle house church bridge tower mountain baby pizza cake apple cup chair bed clock guitar statue sign door window
bottle book shoe hat bench lamp candle mushroom pumpkin tomato bread egg""".split()
ALIAS = {"plane": "airplane", "aircraft": "airplane", "jet": "airplane", "kitten": "cat", "puppy": "dog", "bike": "bicycle",
         "ship": "boat", "sailboat": "boat", "locomotive": "train", "blossom": "flower", "rose": "flower", "tulip": "flower"}
STOPS = set("on in at with against near beside under over from by that who which is are as its their his her".split())
CUE = {"close", "closeup", "macro", "portrait", "tight", "tightly", "headshot", "filling", "fills", "detail"}


def noun_of(tok):
    for t in tok:
        t = ALIAS.get(t, t)
        if t in NOUNS: return t
        if t.endswith("s") and t[:-1] in NOUNS: return t[:-1]
        if t.endswith("es") and t[:-2] in NOUNS: return t[:-2]
    return None


def rgb(d, part, idx):
    meta = json.load(open(f"{d}/meta.json")); pal = np.load(f"{d}/palette.npy"); S = meta["S"]
    x = np.memmap(f"{d}/{part}.u8", np.uint8, mode="r").reshape(-1, S * S)
    return pal[x[np.sort(idx)]].reshape(len(idx), S, S, 3)             # idx must be sorted by the caller


def do_slice():
    caps = json.load(open(CACHE)); meta = json.load(open(f"{D16}/meta.json")); v = meta["val"]
    perm = np.random.default_rng(0).permutation(len(caps))
    rows, labs, strict = [], [], []
    for j, k in enumerate(perm[v:]):
        before, after = split_caption(caps[k])
        head = []
        for t in after[:8]:                           # the subject phrase ends at the first "on", "with", "sitting", ...
            if t in STOPS or (t.endswith("ing") and len(t) > 4): break
            head.append(t)
        n = noun_of(head)
        if n is None: continue
        rows.append(j); labs.append(NOUNS.index(n)); strict.append(any(t in CUE for t in before))
    rows, labs, strict = np.array(rows), np.array(labs), np.array(strict)
    np.savez(OUT, rows=rows, labs=labs, strict=strict)
    ix = {w: i for i, w in enumerate(json.load(open(f"{D16}/words.json")))}
    ok = np.array([NOUNS[l] in ix for l in labs]); cnt = np.bincount(labs, minlength=len(NOUNS))
    p = (1.0 + strict) / np.sqrt(cnt[labs]) * ok; p = p / p.sum()
    for d in sys.argv[2:]:                           # picture sets (same order) that should show these pictures more often
        np.savez(f"{d}/boost.npz", rows=rows[ok], word=np.array([ix[NOUNS[l]] for l in labs[ok]]), p=p[ok] / p[ok].sum())
        m = json.load(open(f"{d}/meta.json")); m["boost_share"] = 0.25; json.dump(m, open(f"{d}/meta.json", "w"))
    print("nouns that are not prompt words:", [n for n in NOUNS if n not in ix])
    print("noun: subject of a close-up photo / subject of another photo")
    for i, n in enumerate(NOUNS): print(f"  {n}: {int((strict & (labs == i)).sum())} / {int((~strict & (labs == i)).sum())}")
    from PIL import Image
    show = [NOUNS.index(n) for n in "cat dog bird horse flower car boat airplane house pizza butterfly train".split()]
    im = Image.new("RGB", (12 * 66, len(show) * 66))
    for r, c in enumerate(show):
        idx = np.sort(rows[strict & (labs == c)][:12])
        for q, p in enumerate(rgb(D64, "train", idx)): im.paste(Image.fromarray(p), (q * 66, r * 66))
    im.save("img/runs/objslice.png")


class Judge(nn.Module):
    def __init__(self, n, ch=3):
        super().__init__()
        w = [ch, 64, 128, 256, 384]
        self.c = nn.ModuleList(nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                             nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU()) for a, b in zip(w, w[1:]))
        self.o = nn.Linear(w[-1], n); self.ch = ch

    def forward(self, x):                      # x: (B, 3, s, s) in 0..1, any s; always judged at 64
        if x.shape[-1] != 64: x = F.interpolate(x, size=64, mode="nearest")
        if self.ch == 1: x = (x * torch.tensor([0.299, 0.587, 0.114], device=x.device)[None, :, None, None]).sum(1, keepdim=True)
        for i, c in enumerate(self.c): x = c(x) if i == 0 else c(F.max_pool2d(x, 2))
        return self.o(x.mean((2, 3)))


def classes(min_n=600):
    z = np.load(OUT); cnt = np.bincount(z["labs"][z["strict"]], minlength=len(NOUNS))
    return [i for i in range(len(NOUNS)) if cnt[i] >= min_n]


def slice_data(d, cls, cap=6000):
    z = np.load(OUT); keep = []
    rng = np.random.default_rng(1)
    for k, c in enumerate(cls):
        r = z["rows"][z["strict"] & (z["labs"] == c)]; r = rng.permutation(r)[:cap]
        keep += [(int(j), k) for j in r]
    keep.sort(); idx = np.array([a for a, _ in keep]); y = np.array([b for _, b in keep])
    x = torch.from_numpy(rgb(d, "train", idx)).permute(0, 3, 1, 2).contiguous()
    hold = np.random.default_rng(2).random(len(idx)) < 0.1
    return x, torch.from_numpy(y), torch.from_numpy(hold), idx


def do_clf():
    cls = classes(); names = [NOUNS[c] for c in cls]; print(len(cls), "nouns:", names)
    res = {}
    for d, s in ((D16, 16), (D32, 32), (D64, 64)):
        x, y, hold, _ = slice_data(d, cls)
        for ch in (3, 1):
            torch.manual_seed(0)
            m = Judge(len(cls), ch).cuda(); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=0.05)
            xt, yt = x[~hold].cuda(), y[~hold].cuda(); wts = (1.0 / torch.bincount(yt, minlength=len(cls)).float()); wts = wts / wts.mean()
            E, B = 12, 256; steps = E * len(xt) // B; sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
            for it in range(steps):
                i = torch.randint(0, len(xt), (B,), device="cuda"); xb = xt[i].float() / 255
                fl = torch.rand(B, device="cuda") < 0.5; xb = torch.where(fl[:, None, None, None], xb.flip(3), xb)
                p = s // 8; xb = F.pad(xb, (p, p, p, p), mode="reflect"); a, b = np.random.randint(0, 2 * p + 1, 2); xb = xb[:, :, a:a + s, b:b + s]
                loss = F.cross_entropy(m(xb), yt[i], weight=wts, label_smoothing=0.1); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            m.eval()
            with torch.no_grad(): pr = torch.cat([m(x[hold][i:i + 512].cuda().float() / 255).argmax(1).cpu() for i in range(0, int(hold.sum()), 512)])
            yh = y[hold]; rec = [float((pr[yh == k] == k).float().mean()) for k in range(len(cls))]
            res[f"{s}{'g' if ch == 1 else ''}"] = rec
            print(f"size {s} {'grey' if ch == 1 else 'colour'}: mean recall {np.mean(rec):.3f}", flush=True)
            torch.save({"sd": m.state_dict(), "cls": cls, "ch": ch}, f"img/runs/judge{s}{'g' if ch == 1 else ''}.pt")
    print("recall per noun (16, 16 grey, 32, 32 grey, 64, 64 grey); chance = %.3f" % (1 / len(cls)))
    for k, n in enumerate(names): print(f"  {n:11s}", " ".join(f"{res[c][k]:.2f}" for c in ("16", "16g", "32", "32g", "64", "64g")))
    json.dump({"names": names, "recall": res}, open("img/runs/judge.json", "w"))


def judges():
    out = []
    for f in ("judge64", "judge64g"):
        d = torch.load(f"img/runs/{f}.pt"); m = Judge(len(d["cls"]), d["ch"]).cuda(); m.load_state_dict(d["sd"]); out.append(m.eval())
    return out, d["cls"]


def load(run):
    from pathlib import Path
    from .model import ImgConfig, ImgModel
    from .data import Pictures
    from .train import route_counts
    run = Path("img/runs") / run
    cfg = ImgConfig(**json.load(open(run / "result.json"))["config"]); data = Pictures(cfg.data)
    counts = route_counts(ImgModel(ImgConfig(D=32, L=2, DK=32, DV=16, E=2, HD=16, DH=16, HH=16, att_layers=[1], S=cfg.S, C=cfg.C)), data, cfg)
    m = ImgModel(cfg, counts, data.pal / 255.0).cuda()
    m.load_state_dict(torch.load(run / ("best.pt" if (run / "best.pt").exists() else "model.pt"), map_location="cpu"))
    return m.eval(), cfg, data


def to_rgb(x, pal, S):
    return torch.from_numpy(pal).cuda()[x].view(-1, S, S, 3).permute(0, 3, 1, 2).float() / 255


def do_enl(*runs, temp=0.7, guide=1.0):
    """Real small pictures enlarged by the model. Two runs: start from real 16x16. Three runs: start from real 8x8."""
    r32, r64 = runs[-2:]
    (jc, jg), cls = judges(); names = [NOUNS[c] for c in cls]
    x16, y, hold, idx = slice_data(D16, cls)
    z = np.memmap(f"{D16}/train.u8", np.uint8, mode="r").reshape(-1, 256)
    sel = np.nonzero(hold.numpy())[0][:2048]; th = torch.from_numpy(z[idx[sel]].astype(np.int64)).cuda(); yy = y[sel]
    gen = torch.Generator(device="cuda"); gen.manual_seed(1)
    x = th
    if len(runs) == 3:                                # 8x8: each 2x2 block becomes the palette colour nearest to its mean
        pal = torch.from_numpy(np.load(f"{D16}/palette.npy")).float().cuda() / 255
        x = torch.cdist(pal[x].view(-1, 8, 2, 8, 2, 3).mean((2, 4)).view(-1, 64, 3), pal[None].expand(len(x), -1, -1)).argmin(-1)
    for r in runs:
        m, cfg, data = load(r)
        cl = None
        if cfg.n_words:
            ix = {w: i for i, w in enumerate(data.words)}; cl = torch.zeros(len(x), cfg.W, dtype=torch.long)
            cl[:, 0] = torch.tensor([ix.get(names[k], 0) for k in yy.tolist()]); cl = cl.cuda()
        x = torch.cat([m.sample(len(x[i:i + 256]), "cuda", None if cl is None else cl[i:i + 256], temp=temp, gen=gen, thumb=x[i:i + 256],
                                guide=guide if cl is not None else 1.0) for i in range(0, len(x), 256)])
        if cfg.S == 32:                               # where is the subject lost: already at 32, or only at 64?
            d = torch.load("img/runs/judge32.pt"); j = Judge(len(d["cls"]), d["ch"]).cuda(); j.load_state_dict(d["sd"]); j.eval()
            real = torch.from_numpy(rgb(D32, "train", idx[sel])).permute(0, 3, 1, 2).cuda().float() / 255
            with torch.no_grad(): print(f"at 32: real {(j(real).argmax(1).cpu() == yy).float().mean():.3f}   enlarged {(j(to_rgb(x, data.pal, 32)).argmax(1).cpu() == yy).float().mean():.3f}")
    im = to_rgb(x, data.pal, 64)
    real = torch.from_numpy(rgb(D64, "train", idx[sel])).permute(0, 3, 1, 2).cuda().float() / 255
    with torch.no_grad():
        for nm, j in (("colour", jc), ("grey", jg)):
            a = (j(im).argmax(1).cpu() == yy).float(); b = (j(real).argmax(1).cpu() == yy).float()
            print(f"{nm}: real 64x64 {b.mean():.3f}   real 16x16 enlarged by the model {a.mean():.3f}")
            print("   per noun (real / enlarged):", " ".join(f"{n} {b[yy == k].mean():.2f}/{a[yy == k].mean():.2f}" for k, n in enumerate(names)))
    data.sheet(x[:96].cpu().numpy(), cols=12, scale=2).save(f"img/runs/enl_{'_'.join(runs)}.png")


def do_hit(base, r32, r64, opt):
    (jc, jg), cls = judges(); names = [NOUNS[c] for c in cls]
    n, guide = int(opt.get("n", 64)), float(opt.get("guide", 3)); extra = [w for w in opt.get("extra", "").split(",") if w]
    ms = [load(r) for r in (base, r32, r64)]
    d = torch.load("img/runs/judge16.pt"); j16 = Judge(len(d["cls"]), 3).cuda(); j16.load_state_dict(d["sd"]); j16.eval()
    ix = {w: i for i, w in enumerate(ms[0][2].words)}
    gen = torch.Generator(device="cuda"); gen.manual_seed(7)
    hits, sheets = {}, []
    for k, nm in enumerate(names):
        x = None
        for si, (m, cfg, data) in enumerate(ms):
            cl = None
            if cfg.n_words:
                cl = torch.zeros(n, cfg.W, dtype=torch.long)
                for q, w in enumerate([w for w in [nm] + extra if w in ix][:cfg.W]): cl[:, q] = ix[w]
                cl = cl.cuda()
            x = m.sample(n, "cuda", cl, temp=1.0 if x is None else float(opt.get("temp", 0.7)), gen=gen, thumb=x, guide=guide if si == 0 else float(opt.get("eguide", 1.0)) if cl is not None else 1.0)
            if si == 0:
                with torch.no_grad(): h16 = float((j16(to_rgb(x, data.pal, 16)).argmax(1) == k).float().mean())
        im = to_rgb(x, ms[-1][2].pal, 64)
        with torch.no_grad(): hits[nm] = (float((jc(im).argmax(1) == k).float().mean()), float((jg(im).argmax(1) == k).float().mean()), h16)
        sheets.append(x[:12].cpu().numpy())
    tag = opt.get("tag", base); guide = f"{guide} enlargers {opt.get('eguide', 1.0)} temp {opt.get('temp', 0.7)}"
    ms[-1][2].sheet(np.concatenate(sheets), cols=12, scale=2).save(f"img/runs/hit_{tag}.png")
    print(f"hit rate {base} {r32} {r64} guide {guide} extra {extra}: at 64 colour {np.mean([h[0] for h in hits.values()]):.3f}  grey {np.mean([h[1] for h in hits.values()]):.3f}  at 16 {np.mean([h[2] for h in hits.values()]):.3f}  chance {1 / len(names):.3f}")
    print("  (64 colour / 64 grey / 16) " + "  ".join(f"{k} {a:.2f}/{b:.2f}/{c:.2f}" for k, (a, b, c) in hits.items()))
    json.dump(hits, open(f"img/runs/hit_{tag}.json", "w"))


if __name__ == "__main__":
    a = [x for x in sys.argv[2:] if "=" not in x]; o = dict(x.split("=") for x in sys.argv[2:] if "=" in x)
    {"slice": do_slice, "clf": do_clf, "enl": lambda: do_enl(*a, temp=float(o.get("temp", 0.7)), guide=float(o.get("guide", 1.0))), "hit": lambda: do_hit(*a, o)}[sys.argv[1]]()
