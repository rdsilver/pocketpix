"""Teach the cartridge models from clean pictures made by a big open model (FLUX.1-schnell, Apache 2.0).

  python3 -m img.distill prompts <out.jsonl> [per_word=64] [words=trial|all|a,b,c] [seed=1]
        one line per picture to make: {"i", "prompt", "seed", "noun", "colour"}
  python3 -m img.flux_gen ...                       (makes the pictures; see that file)
  python3 -m img.distill build <jobs.jsonl> <gen dir> <tag>
        -> img/data/x64<tag>, x32<tag>, x16<tag>: the same files img.train reads (palette indices, word ids)
  python3 -m img.distill train <model> <tag> <gpu> [epochs=10] [lr=2e-4] [wq_lr_mult=2] [name=..] [key=value ...]
        fine-tunes img/remote/night/<model>.pt (b16C, u32M or u64M) on img/data/x<S><tag>; run name <model>_<tag>
  python3 -m img.distill photoval                   general photos (img/data/photos64_test) in the cartridge palette
        -> img/data/pv64, pv32, pv16, to see whether the rest of what the models know gets worse
  python3 -m img.distill score <tag> <name> ...     bits per pixel of each model (night name or run name) on the FLUX
        held-out pictures (with their prompts) and on the general photos (empty prompt); lower is better
  python3 -m img.distill sheet <out.png> <base> <u32> <u64> [words=car,red;dog;-;...] [n=8] [seed=1]   (- = empty prompt)
        draws pictures through three models, each a night name (old) or a run name (new), PyTorch on the GPU
  python3 -m img.distill bundle <tag>               best weights + configs of the three runs -> bundle/x<tag>, to copy back from the GPU machine
  python3 -m img.distill contact <gen dir> <jobs.jsonl> <out.png>   a page of the made pictures, with their prompts

The palette is the cartridge's (img/remote/pal/p16o.npy), the word list the night models' (4096 words), so the new
weights drop straight into pix/export.py.
"""
from __future__ import annotations

import glob, json, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIGHT = f"{ROOT}/img/remote/night"
PAL = f"{ROOT}/img/remote/pal/p16o.npy"

# ---------------------------------------------------------------- what to draw
TRIAL = ("airplane apple ball balloon banana bear bed beer bench bicycle bird boat book bottle bowl boy bread bus butterfly "
         "cake candle car castle cat chair chicken church cherry clock coffee cow cup deer desk dog door dress drum duck eagle "
         "egg elephant face fish flag flower fountain fruit girl guitar hat helicopter horse house insect lamp laptop lemon "
         "lighthouse lion man mirror monkey motorcycle mushroom orange phone piano plate rose salad scooter sculpture sheep shoe "
         "skyscraper snake sofa spider statue strawberry taxi television tent tiger tomato tower train tree truck umbrella van "
         "vase vegetable window wine woman zebra bridge").split()
TRIAL = [w for w in TRIAL if w != "sculpture"]                       # (statue is the word the night models know)
COLOURS = "red orange yellow green blue purple pink brown white black grey".split()
PEOPLE = {"man", "woman", "girl", "boy", "face"}
# nouns whose colour can be asked for, with the colours that make sense (empty: any of COLOURS)
COLOURED = {
    **{w: [] for w in "car bus truck van boat bicycle motorcycle scooter airplane helicopter train house door cake cup chair bed clock "
                    "guitar bottle book hat lamp candle balloon umbrella ball dress tent flower rose vase flag bowl plate bench sofa "
                    "desk phone piano drum laptop mug".split()},
    "dog": ["white", "black", "brown"], "cat": ["white", "black", "grey", "orange"], "horse": ["white", "black", "brown"],
    "bird": ["blue", "red", "yellow", "white", "black"], "fish": ["orange", "blue", "yellow", "red"], "butterfly": ["blue", "yellow", "orange", "white"],
    "cow": ["brown", "black", "white"], "sheep": ["white", "black"], "chicken": ["white", "brown"], "duck": ["white", "yellow"],
    "apple": ["red", "green", "yellow"], "mushroom": ["red", "brown", "white"], "bear": ["brown", "black", "white"], "insect": ["green", "black", "red"],
    "snake": ["green", "black", "yellow"], "spider": ["black", "brown"], "fruit": [], "tree": ["green", "orange", "yellow"], "egg": ["white", "brown"],
    "shoe": [], "statue": ["white", "grey", "brown"], "tower": ["white", "grey", "red"], "lighthouse": ["white", "red"],
}
VEHICLE = set("car bus truck van boat bicycle motorcycle scooter airplane helicopter train taxi".split())
ANIMAL = set("dog cat horse bird fish butterfly cow sheep chicken duck bear lion tiger zebra elephant deer monkey snake spider insect eagle".split())
BIG = set("house castle church tower lighthouse skyscraper bridge fountain tree tent windmill".split())
BACKS = ["white", "light grey", "pale blue", "pale beige", "pale green", "pale pink", "pale yellow"]


# natural surroundings for each kind of thing (the night models' photos have real backgrounds; plain studio ones would teach
# the cartridge models to paint everything on pastel)
SETTING = {
    **{w: ["parked on a city street", "on a country road", "in a parking lot", "on a street in a small town"] for w in "car taxi van truck bus".split()},
    **{w: ["on a city street", "on a country road", "parked by a wall", "on a park path"] for w in "motorcycle scooter bicycle".split()},
    "boat": ["on a lake", "on the sea", "moored in a harbour", "on a river"], "airplane": ["flying in the sky", "on an airport runway", "flying over clouds"],
    "helicopter": ["flying in the sky", "on a helipad"], "train": ["on railway tracks in the countryside", "at a station", "crossing a bridge"],
    **{w: ["on green grass in a field", "in a park", "on a forest path", "in a garden", "on a farm"] for w in "dog cat horse cow sheep chicken deer monkey bear".split()},
    **{w: ["in the savanna", "in the wild", "on dry grassland", "at a watering hole"] for w in "lion tiger zebra elephant".split()},
    **{w: ["on a branch", "flying in the sky", "on a fence", "on the grass"] for w in "bird eagle".split()},
    **{w: ["on a flower", "on a leaf", "in a garden"] for w in "butterfly insect spider".split()},
    "snake": ["on the ground among leaves", "on a rock", "in the grass"], "fish": ["underwater in clear blue water", "in an aquarium", "in a coral reef"],
    "duck": ["on a pond", "on a river bank", "on the grass by a lake"],
    **{w: ["under a blue sky", "on a sunny day with clouds", "in the evening light", "on a grey day"] for w in "house castle church tower skyscraper bridge fountain".split()},
    "lighthouse": ["on a rocky coast", "by the sea under a blue sky"], "tree": ["in a field", "in a park", "on a hill under a blue sky"],
    "tent": ["at a campsite in the woods", "in a meadow", "by a lake"],
    **{w: ["on a wooden table", "on a kitchen counter", "on a plate on a table", "on a picnic blanket"] for w in
       "apple banana bread cake cherry egg fruit lemon orange salad strawberry tomato vegetable".split()},
    "mushroom": ["on the forest floor", "in the grass", "on moss"],
    **{w: ["on a table in a cafe", "on a wooden table", "on a bar counter", "on a table outdoors"] for w in "beer wine coffee cup".split()},
    **{w: ["in a living room", "in a bedroom", "in a room with a window", "in a cosy room"] for w in "bed chair sofa desk lamp piano television mirror clock".split()},
    **{w: ["on a desk", "on a table", "on a shelf", "on a kitchen table"] for w in "laptop phone vase book bottle bowl plate candle".split()},
    **{w: ["in a music room", "on a stage", "leaning on a wall"] for w in "drum guitar".split()},
    **{w: ["on a shelf in a shop", "on a wooden floor", "on a chair"] for w in "hat shoe".split()}, "dress": ["on a mannequin in a shop", "hanging on a rack"],
    "door": ["of a house", "of an old building", "in a brick wall"], "window": ["of a house", "in a brick wall", "of an old building"],
    "umbrella": ["on a rainy street", "on a beach", "in a park"], "balloon": ["floating in the sky", "in a park", "at a party"],
    "ball": ["on green grass", "on a playground", "on a beach"], "flag": ["on a pole against the sky", "waving in the wind"],
    "bench": ["in a park", "by a path", "in a garden"], "statue": ["in a city square", "in a park", "in a museum"],
    **{w: ["in a garden", "in a field", "in a flower bed"] for w in "flower rose".split()},
    **{w: ["on a city street", "in a park", "on a beach", "in a room"] for w in "man woman girl boy".split()},
}


def prompt_for(noun, colour, rng):
    art = "an" if (colour or noun)[0] in "aeiou" else "a"
    thing = f"{'orange-coloured' if colour == 'orange' else colour} {noun}" if colour else noun     # ("an orange bed" draws the fruit)
    where = str(rng.choice(SETTING.get(noun, ["in a natural everyday setting"])))
    if noun == "face":
        return f"a close-up portrait photo of a person's face, looking at the camera, {rng.choice(['outdoors', 'indoors', 'in a park', 'on a street'])}, natural light"
    # many viewpoints, so the small models learn the thing from every side, not one stock angle
    common = ["front view", "side view facing left", "side view facing right", "rear view", "front three-quarter view from the left",
              "front three-quarter view from the right", "rear three-quarter view", "seen from above", "low angle view from below",
              "seen from a slightly high angle", "seen from far away", "close up"]
    if noun in ANIMAL:
        view = rng.choice(common + ["walking", "running", "lying down", "turning its head", "looking at the camera", "seen from behind"])
    elif noun in VEHICLE:
        view = rng.choice(common + ["driving towards the camera", "moving away from the camera", "seen from a bird's-eye view"])
    elif noun in BIG:
        view = rng.choice(["front view", "seen from the side", "seen from a corner", "seen from far away", "low angle view from the ground",
                           "aerial view", "seen across a field", "seen at an angle"])
    else:
        view = rng.choice(common + ["tilted at an angle", "lying on its side", "seen straight from above", "at eye level"])
    if noun in PEOPLE:
        view = rng.choice(["standing, front view", "walking, side view", "seen from behind", "sitting", "standing, three-quarter view",
                           "seen from a high angle", "seen from far away", "waving"])
        return f"a photo of {art} {noun} {where}, {view}, full length, centred, natural light"
    whole = "filling the frame" if view == "close up" else f"the whole {noun} in the picture and centred"
    return f"a photo of {art} {thing} {where}, {view}, {whole}, natural light, realistic"


def prompts(out, per_word="64", words="trial", seed="1"):
    vocab = json.load(open(f"{NIGHT}/words.json")); ix = {w: i for i, w in enumerate(vocab)}
    ws = TRIAL if words == "trial" else [w for w in vocab[2:]] if words == "all" else words.split(",")
    bad = [w for w in ws if w not in ix]
    assert not bad, f"not in the cartridge's word list: {bad}"
    rng = np.random.default_rng(int(seed)); per = int(per_word); n = 0
    assert len(ws) >= 64, "at least 64 words: training draws its sample sheets from 4 held-out pictures per word"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        for noun in ws:
            for k in range(per):
                colour = ""
                if noun in COLOURED and noun not in PEOPLE and k % 2 == 1:          # half of them with a colour asked for
                    colour = str(rng.choice(COLOURED[noun] or COLOURS))
                f.write(json.dumps({"i": n, "prompt": prompt_for(noun, colour, rng), "seed": int(rng.integers(1, 2**31 - 1)),
                                    "noun": noun, "colour": colour}) + "\n"); n += 1
    print(out, n, "pictures,", len(ws), "words")


# ---------------------------------------------------------------- pictures -> training files
def read_gen(gen):
    """All shards a flux_gen run wrote: (ids, (n, 128, 128, 3) uint8), sorted by id."""
    ids, ims = [], []
    for p in sorted(glob.glob(f"{gen}/shard_??_????.npz")):            # (not a .tmp.npz left by a stopped process)
        z = np.load(p); ids.append(z["ids"]); ims.append(z["img"])
    ids, ims = np.concatenate(ids), np.concatenate(ims)
    o = np.argsort(ids, kind="stable"); ids, ims = ids[o], ims[o]
    keep = np.r_[True, ids[1:] != ids[:-1]]                          # a shard written twice (a resumed run) counts once
    return ids[keep], ims[keep]


def thumb(x, S, pal):
    """(n, S*S) palette indices -> (n, S*S/4): each 2 x 2 block becomes the palette colour nearest to its mean colour,
    the rule the night sets were made with (img/thumbs.py) and the one ImgModel.thumb_of uses for the enlargers."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = torch.from_numpy(pal).float().to(dev) / 255.0; T = S // 2; out = []
    for i in range(0, len(x), 4096):
        rgb = P[torch.from_numpy(x[i:i + 4096].astype(np.int64)).to(dev)].view(-1, T, 2, T, 2, 3).mean((2, 4)).view(-1, T * T, 3)
        out.append(torch.cdist(rgb, P[None].expand(len(rgb), -1, -1)).argmin(-1).cpu().numpy().astype(np.uint8))
    return np.concatenate(out)


def write_set(d, x, w, caps, pal, vocab, tr, va, extra):
    os.makedirs(d, exist_ok=True)
    x[tr].tofile(f"{d}/train.u8"); x[va].tofile(f"{d}/val.u8")
    np.save(f"{d}/palette.npy", pal); json.dump(vocab, open(f"{d}/words.json", "w"))
    np.save(f"{d}/train_words.npy", w[tr]); np.save(f"{d}/val_words.npy", w[va])
    open(f"{d}/val_captions.txt", "w").write("\n".join(caps[k] for k in va))
    S = int(round(x.shape[1] ** 0.5))
    meta = {"S": S, "P": len(pal), "train": int(len(tr)), "val": int(len(va)), "n_words": len(vocab), "W": 8, "transparent": False, **extra}
    json.dump(meta, open(f"{d}/meta.json", "w")); print(d, meta, flush=True)


def build(jobs, gen, tag):
    from .data import palette_lut, index_images, halve
    J = {j["i"]: j for j in map(json.loads, open(jobs))}
    ids, im128 = read_gen(gen)
    missing = len(J) - len(ids)
    print(len(ids), "pictures made,", missing, "missing", flush=True)
    rgb64 = halve(im128)
    vocab = json.load(open(f"{NIGHT}/words.json")); ix = {w: i for i, w in enumerate(vocab)}
    w = np.zeros((len(ids), 8), np.int16); caps = []
    for r, i in enumerate(ids):
        j = J[int(i)]                                                # the noun, and the colour only when the prompt asked for it
        w[r, 0] = ix[j["noun"]]
        if j["colour"] and j["colour"] != j["noun"]: w[r, 1] = ix[j["colour"]]
        caps.append(f"{j['colour']} {j['noun']}".strip())
    # held out: the first 4 pictures of every noun
    seen, val = {}, np.zeros(len(ids), bool)
    for r, i in enumerate(ids):
        n = J[int(i)]["noun"]; seen[n] = seen.get(n, 0) + 1; val[r] = seen[n] <= 4
    pal = np.load(PAL)
    x = {64: index_images(rgb64, palette_lut(pal))}                  # 64 from the picture; 32 and 16 as thumbnails, like the night sets
    x[32] = thumb(x[64], 64, pal); x[16] = thumb(x[32], 32, pal)
    rng = np.random.default_rng(0); tr = rng.permutation(np.nonzero(~val)[0]); va = np.nonzero(val)[0]
    err = float(np.abs(pal[x[64][:2000]].astype(int) - rgb64[:2000].reshape(min(2000, len(rgb64)), -1, 3)).mean())
    for S in (64, 32, 16):
        write_set(f"{ROOT}/img/data/x{S}{tag}", x[S], w, caps, pal, vocab, tr, va,
                  {"source": "FLUX.1-schnell", "jobs": os.path.abspath(jobs), "palette_mean_abs_error_64": round(err, 2)})
    # the same plus general photos with empty prompts (img/data/photos64_test's training part, not the photos `score`
    # uses), so fine-tuning keeps what the models know beyond the taught words: x<S><tag>m
    a = photos("train")
    ax = {64: index_images(a, palette_lut(pal))}; ax[32] = thumb(ax[64], 64, pal); ax[16] = thumb(ax[32], 32, pal)
    n = len(ids)
    for S in (64, 32, 16):
        write_set(f"{ROOT}/img/data/x{S}{tag}m", np.concatenate([x[S], ax[S]]), np.concatenate([w, np.zeros((len(a), 8), np.int16)]),
                  caps + [""] * len(a), pal, vocab, np.random.default_rng(1).permutation(np.r_[tr, n + np.arange(len(a))]), va,
                  {"source": "FLUX.1-schnell + general photos", "jobs": os.path.abspath(jobs), "photos": int(len(a))})
    os.makedirs(f"{ROOT}/img/runs", exist_ok=True)
    from .data import Pictures
    dd = Pictures(f"{ROOT}/img/data/x64{tag}")
    dd.sheet(dd.val[:96], cols=16, scale=2).save(f"{ROOT}/img/runs/x{tag}_val64.png")
    print("val sheet img/runs/x%s_val64.png" % tag, flush=True)


def photos(part):
    """img/data/photos64_test's train (7,692) or val (500) photos as (n, 64, 64, 3) RGB (they have their own palette)."""
    src = f"{ROOT}/img/data/photos64_test"
    a = np.fromfile(f"{src}/{part}.u8", np.uint8).reshape(-1, 4096)
    return np.load(f"{src}/palette.npy")[a].reshape(-1, 64, 64, 3)


def photoval(n="1000"):
    """General photos to check what the rest of the model still knows: img/data/photos64_test's held-out part put into
    the cartridge palette, 32 and 16 made as thumbnails; empty prompts."""
    from .data import palette_lut, index_images
    rgb = photos("val")[:int(n)]; a = rgb
    pal = np.load(PAL); vocab = json.load(open(f"{NIGHT}/words.json"))
    x = {64: index_images(rgb, palette_lut(pal))}; x[32] = thumb(x[64], 64, pal); x[16] = thumb(x[32], 32, pal)
    w = np.zeros((len(a), 8), np.int16); idx = np.arange(len(a))
    for S in (64, 32, 16):
        write_set(f"{ROOT}/img/data/pv{S}", x[S], w, [""] * len(a), pal, vocab, idx, idx, {"source": "photos64_test, empty prompts"})


def score(tag, *names):
    """Bits per pixel (lower is better) on the FLUX held-out pictures and on the general photos, for each model."""
    import torch
    from .data import Pictures
    from .train import evaluate
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'model':16s} {'FLUX held-out':>14s} {'general photos':>15s}", flush=True)
    for nm in names:
        m, cfg = load(nm, dev); r = []
        for d in (f"{ROOT}/img/data/x{cfg.S}{tag}", f"{ROOT}/img/data/pv{cfg.S}"):
            data = Pictures(d); r.append(evaluate(m, data, cfg, dev) if len(data.val) else float("nan"))
        print(f"{nm:16s} {r[0]:14.4f} {r[1]:15.4f}", flush=True)


# ---------------------------------------------------------------- fine-tuning
def train(model, tag, gpu, *over):
    cfg = json.load(open(f"{NIGHT}/{model}.json"))["config"]
    S = cfg["S"]; d = f"img/data/x{S}{tag}"
    meta = json.load(open(f"{ROOT}/{d}/meta.json"))
    o = {}
    for a in over:
        k, v = a.split("=", 1)
        try: o[k] = json.loads(v)
        except ValueError: o[k] = v
    epochs, lr = float(o.pop("epochs", 10)), float(o.pop("lr", 2e-4))
    # gentle by default: the night runs ended with lr near 0 and many ternary weights next to a rounding edge, so the
    # matrices get only twice the base rate here (the night runs used ten times)
    batch = int(o.get("batch", cfg["batch"]))
    steps = int(o.pop("steps", max(200, int(meta["train"] * epochs / batch))))
    sets = {**{k: v for k, v in cfg.items() if k not in ("name", "data", "steps", "lr", "warmup", "eval_every", "val_images", "ckpt_every")},
            "data": d, "name": f"{model}_{tag}", "steps": steps, "lr": lr, "wq_lr_mult": 2.0, "warmup": min(300, steps // 5),
            "eval_every": max(50, steps // 20), "val_images": meta["val"], "ckpt_every": max(100, steps // 4), "init": f"{NIGHT}/{model}.pt"}
    sets.update(o)
    args = [sys.executable, "-m", "img.train", "--set"] + [f"{k}={json.dumps(v)}" for k, v in sets.items()]
    print(model, "->", sets["name"], f"{steps} steps of {batch} ({steps * batch / meta['train']:.1f} passes over {meta['train']} pictures),",
          f"lr {sets['lr']}, matrices x{sets['wq_lr_mult']}", flush=True)
    os.makedirs(f"{ROOT}/img/runs", exist_ok=True)
    r = subprocess.run(args, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
    sys.exit(r.returncode)


# ---------------------------------------------------------------- looking at the result
def load(name, device):
    """A night model (img/remote/night/<name>.pt) or a run (img/runs/<name>/best.pt), with the cartridge palette."""
    import torch
    from .model import ImgConfig, ImgModel
    run = f"{ROOT}/img/runs/{name}"
    if os.path.exists(f"{run}/result.json"):
        cfg = json.load(open(f"{run}/result.json"))["config"]; w = f"{run}/best.pt" if os.path.exists(f"{run}/best.pt") else f"{run}/model.pt"
    else:
        cfg = json.load(open(f"{NIGHT}/{name}.json"))["config"]; w = f"{NIGHT}/{name}.pt"
    cfg = ImgConfig(**cfg)
    m = ImgModel(cfg, None, np.load(PAL) / 255.0)
    m.load_state_dict(torch.load(w, map_location="cpu")); return m.to(device).eval(), cfg


def sheet(out, base, u32, u64, *opts):
    import torch
    from PIL import Image, ImageDraw
    o = dict(a.split("=", 1) for a in opts)
    n, seed = int(o.get("n", 8)), int(o.get("seed", 1))
    # trained words first, then words the trial does not teach and the empty prompt (they should not get worse)
    prompts_ = [[w for w in p.split(",") if w != "-"] for p in o.get("words", "car,red;dog;cat,black;house;flower,yellow;bus;horse;apple,green;sky;beach;mountain;street;-").split(";")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = json.load(open(f"{NIGHT}/words.json")); ix = {w: i for i, w in enumerate(vocab)}
    pal = np.load(PAL)
    models = [load(m, dev) for m in (base, u32, u64)]
    # the cartridge's settings: base guidance 3 at temperature 256/284, enlargers no guidance at 256/366
    temps, guides = [256 / 284, 256 / 366, 256 / 366], [3.0, 1.0, 1.0]
    rows = []
    for p in prompts_:
        cl = torch.zeros(n, 8, dtype=torch.long)
        for k, w in enumerate(p): cl[:, k] = ix[w]
        g = torch.Generator(device=dev); g.manual_seed(seed); x = None
        with torch.no_grad():
            for (m, cfg), t, gd in zip(models, temps, guides):
                x = m.sample(n, dev, cl.to(dev), temp=t, gen=g, thumb=x, guide=gd)
        rows.append((" ".join(reversed(p)) or "(empty)", pal[x.cpu().numpy()].reshape(n, 64, 64, 3)))
    Z, gap, lab = 2, 6, 18
    im = Image.new("RGB", (gap + n * (64 * Z + gap) + 110, gap + len(rows) * (64 * Z + gap)), (16, 19, 28)); dr = ImageDraw.Draw(im)
    for r, (cap, a) in enumerate(rows):
        y = gap + r * (64 * Z + gap); dr.text((gap, y + 4), cap, fill=(230, 230, 235))
        for k in range(n): im.paste(Image.fromarray(a[k]).resize((64 * Z, 64 * Z), Image.NEAREST), (110 + k * (64 * Z + gap), y))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True); im.save(out); print(out, flush=True)


def contact(gen, jobs, out, n="240"):
    from PIL import Image
    J = {j["i"]: j for j in map(json.loads, open(jobs))}
    ids, ims = read_gen(gen)
    k = np.random.default_rng(0).permutation(len(ids))[:int(n)]; k.sort()
    cols = 20; rows = -(-len(k) // cols)
    im = Image.new("RGB", (cols * 130, rows * 130), (16, 19, 28))
    for q, r in enumerate(k): im.paste(Image.fromarray(ims[r]), ((q % cols) * 130 + 1, (q // cols) * 130 + 1))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True); im.save(out)
    open(out + ".txt", "w").write("\n".join(f"{q // cols + 1},{q % cols + 1}: {J[int(ids[r])]['prompt']}" for q, r in enumerate(k)))
    print(out, len(k), flush=True)


def bundle(tag):
    import hashlib, shutil
    d = f"{ROOT}/bundle/x{tag}"; os.makedirs(d, exist_ok=True)
    for m in ("b16C", "u32M", "u64M"):
        run = f"{ROOT}/img/runs/{m}_{tag}"
        if not os.path.exists(f"{run}/result.json"): print("missing", run); continue
        shutil.copy(f"{run}/best.pt" if os.path.exists(f"{run}/best.pt") else f"{run}/model.pt", f"{d}/{m}.pt")
        r = json.load(open(f"{run}/result.json")); r["config"]["name"] = m; json.dump(r, open(f"{d}/{m}.json", "w"), indent=1)
        for f in glob.glob(f"{run}/samples_final*.png"): shutil.copy(f, f"{d}/{m}_{os.path.basename(f)}")
    for f in glob.glob(f"{ROOT}/img/runs/x{tag}_*.png"): shutil.copy(f, d)
    names = sorted(f for f in os.listdir(d) if f not in ("SHA256SUMS", "PARTSUMS") and ".part" not in f)
    sums, parts = [], []
    for f in names:
        b = open(f"{d}/{f}", "rb").read(); sums.append(f"{hashlib.sha256(b).hexdigest()}  {f}")
        for i in range(0, max(len(b), 1), 12 << 20):
            pn = f"{f}.part{i // (12 << 20):04d}"; open(f"{d}/{pn}", "wb").write(b[i:i + (12 << 20)])
            parts.append(f"{hashlib.sha256(b[i:i + (12 << 20)]).hexdigest()}  {pn}")
    open(f"{d}/SHA256SUMS", "w").write("\n".join(sums) + "\n"); open(f"{d}/PARTSUMS", "w").write("\n".join(parts) + "\n")
    print(d, names, flush=True)


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    {"prompts": prompts, "build": build, "photoval": photoval, "score": score, "train": train, "sheet": sheet, "contact": contact, "bundle": bundle}[c](*a)
