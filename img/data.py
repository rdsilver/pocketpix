"""Turn a picture collection into training files for the picture model.

    python3 -m img.data --hf Angais/pixel-art-atlas-1.4m --caption-col caption --out img/data/atlas16 --S 16
    python3 -m img.data --hf azyg0s/pixelart-172k --caption-col text --out img/data/icons32 --S 32
    python3 -m img.data --wds data-archetype/commoncatalog_cc_by_3m_512 --shards 240 --S 64 --out img/data/photos64 --half img/data/photos32
    python3 -m img.data --hf uoft-cs/cifar10 --image-col img --out img/data/cifar10 [--limit N]
    python3 -m img.data --npy pictures.npy --out img/data/x          (uint8 array N x S x S x 3)

Output directory: train.u8 / val.u8 (N x S*S palette indices, reading order), palette.npy (P x 3, 8-bit values of
15-bit GBA colours), meta.json, and with captions: words.json (the word list; id 0 = empty) and train_words.npy /
val_words.npy (N x 8 word ids). Pictures with transparency keep it: palette index 0 means "see-through" and is
shown in the background colour.
The palette is 16 groups of 16 colours (index = group * 16 + member) so the model can pick a group, then a colour.
"""
from __future__ import annotations

import argparse, io, json, os
import numpy as np


def kmeans(x, k, iters=25, seed=0):
    """Plain k-means on float32 rows; returns (centres, assignment). Empty clusters are re-seeded on far points."""
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=len(x) < k)].copy()
    for _ in range(iters):
        d = (x * x).sum(1)[:, None] - 2 * x @ c.T + (c * c).sum(1)[None]
        a = d.argmin(1)
        far = d[np.arange(len(x)), a].argsort()[::-1]
        for j in range(k):
            m = a == j
            c[j] = x[m].mean(0) if m.any() else x[far[j]]
    d = (x * x).sum(1)[:, None] - 2 * x @ c.T + (c * c).sum(1)[None]
    return c, d.argmin(1)


def build_palette(pix, G=16, M=16, seed=0):
    """pix: (n, 3) uint8 sample of pixels. Returns (G*M, 3) uint8 colours on the 15-bit grid, grouped."""
    x = (pix >> 3).astype(np.float32)                                # 5 bits per channel, as the GBA shows them
    gc, ga = kmeans(x, G, seed=seed)
    order = (gc @ np.array([0.299, 0.587, 0.114], np.float32)).argsort()        # groups from dark to light
    pal = []
    for g in order:
        sub = x[ga == g]
        mc, _ = kmeans(sub, M, seed=seed + 1 + int(g))
        mc = mc[(mc @ np.array([0.299, 0.587, 0.114], np.float32)).argsort()]
        pal.append(np.clip(np.rint(mc), 0, 31))
    p5 = np.concatenate(pal).astype(np.uint8)
    return (p5 << 3) | (p5 >> 2)


BG = (58, 90, 165)          # what see-through pixels are shown as (the title-screen blue)


def palette_lut(pal, skip0=False):
    """Nearest palette index for each of the 32,768 15-bit colours (never index 0 when it means see-through)."""
    g = np.stack(np.meshgrid(*[np.arange(32)] * 3, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    p = (pal >> 3).astype(np.float32)
    d = (g * g).sum(1)[:, None] - 2 * g @ p.T + (p * p).sum(1)[None]
    if skip0: d[:, 0] = np.inf
    return d.argmin(1).astype(np.uint8)


def index_images(a, lut):
    """a: (n, S, S, 3) or (n, S, S, 4) uint8. With an alpha channel, pixels under half opacity become index 0."""
    q = (a[..., :3] >> 3).astype(np.int32)
    idx = lut[(q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]]
    if a.shape[-1] == 4: idx = np.where(a[..., 3] < 128, 0, idx)
    return idx.reshape(len(a), -1).astype(np.uint8)


def shrink(im, S):
    """Centre-crop to a square and shrink with area averaging."""
    from PIL import Image
    if im.mode in ("RGBA", "LA", "P"):                               # keep transparency; sprites are used at their own size
        im = im.convert("RGBA")
        if im.size != (S, S): im = im.resize((S, S), Image.NEAREST)
        return np.asarray(im)
    im = im.convert("RGB")
    w, h = im.size
    m = min(w, h)
    im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
    return np.asarray(im if m == S else im.resize((S, S), Image.BOX if m % S == 0 else Image.LANCZOS))


def from_hf(name, image_col, caption_col, S, limit, split, config=None):
    """Reads the parquet files of a hub dataset directly (only the columns needed). Returns (pictures, captions)."""
    import pyarrow.parquet as pq
    from PIL import Image
    from huggingface_hub import HfFileSystem, hf_hub_download
    fs = HfFileSystem()
    files = sorted(f for f in fs.glob(f"datasets/{name}/**/*.parquet") if f"{split}-" in f.rsplit("/", 1)[1])
    assert files, f"no parquet files for split {split!r} in {name}"
    imgs, caps = [], []
    for f in files:
        p = hf_hub_download(name, f.split(name + "/", 1)[1], repo_type="dataset")
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=4096, columns=[image_col] + ([caption_col] if caption_col else [])):
            d = b.to_pydict()
            for k, im in enumerate(d[image_col]):
                imgs.append(shrink(Image.open(io.BytesIO(im["bytes"])), S))
                if caption_col: caps.append(d[caption_col][k] or "")
            if limit and len(imgs) >= limit: break
        print(f.rsplit("/", 1)[1], len(imgs), flush=True)
        if limit and len(imgs) >= limit: break
    if limit: imgs, caps = imgs[:limit], caps[:limit]
    return np.stack(imgs), (caps if caption_col else None)


def _tar_pictures(args):
    """One webdataset shard -> (n, S, S, 3) uint8. JPEGs are decoded at reduced size, which is several times faster."""
    path, S = args
    import tarfile
    from PIL import Image
    out = []
    with tarfile.open(path) as t:
        for m in t:
            if not m.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")): continue
            try:
                im = Image.open(io.BytesIO(t.extractfile(m).read()))
                im.draft("RGB", (S * 2, S * 2))
                out.append(shrink(im.convert("RGB"), S))
            except Exception:
                pass
    return np.stack(out) if out else np.zeros((0, S, S, 3), np.uint8)


def from_wds(name, n_shards, S, workers=0):
    """Photos from the first n_shards .tar files of a hub dataset; each shard is deleted after it is read."""
    from multiprocessing import Pool
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfFileSystem, hf_hub_download
    files = sorted(f for f in HfFileSystem().glob(f"datasets/{name}/**/*.tar"))
    assert files, f"no .tar files in {name}"
    files = [files[k] for k in sorted({int(i * len(files) / n_shards) for i in range(n_shards)})] if n_shards < len(files) else files   # spread over all shapes of photo
    workers = workers or min(16, os.cpu_count() or 4)
    parts, n = [], 0
    with Pool(workers) as pool:
        for i in range(0, len(files), workers):
            with ThreadPoolExecutor(8) as tp:                    # several downloads at once: one stream is slow
                paths = list(tp.map(lambda f: hf_hub_download(name, f.split(name + "/", 1)[1], repo_type="dataset"), files[i:i + workers]))
            for a in pool.imap(_tar_pictures, [(p, S) for p in paths]):
                parts.append(a); n += len(a)
            for p in paths:
                real = os.path.realpath(p)
                os.remove(real)
                if os.path.islink(p): os.remove(p)
            print(f"{i + len(paths)}/{len(files)} shards, {n} pictures", flush=True)
    return np.concatenate(parts)


def halve(a):
    """(n, S, S, 3) -> (n, S/2, S/2, 3) by averaging 2 x 2 blocks."""
    n, S = a.shape[:2]
    return ((a.reshape(n, S // 2, 2, S // 2, 2, 3).astype(np.uint16).sum((2, 4)) + 2) // 4).astype(np.uint8)


def words_of(text):
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


STOP = {"a", "an", "the", "of", "and", "with", "in", "on", "pixel", "art", "icon", "sprite"}


def build_words(caps, n_words=8192, W=8):
    """Word list (most frequent first; id 0 = empty slot, id 1 = any other word) and an (N, W) id array."""
    from collections import Counter
    toks = [[w for w in words_of(c) if w not in STOP][:W] for c in caps]
    cnt = Counter(w for t in toks for w in t)
    vocab = ["", "<other>"] + [w for w, _ in cnt.most_common(n_words - 2)]
    ix = {w: i for i, w in enumerate(vocab)}
    ids = np.zeros((len(caps), W), np.int16)
    for i, t in enumerate(toks):
        ids[i, :len(t)] = [ix.get(w, 1) for w in t]
    return vocab, ids


def write(out, a, caps, S, val_n, seed=0):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(a))
    alpha = a.shape[-1] == 4
    sample = a[np.sort(perm[:min(len(a), 20000)])].reshape(-1, a.shape[-1])
    if alpha: sample = sample[sample[:, 3] >= 128]
    pal = build_palette(sample[rng.choice(len(sample), min(len(sample), 400000), replace=False)][:, :3])
    if alpha: pal[0] = BG
    lut = palette_lut(pal, skip0=alpha)
    idx = np.empty((len(a), S * S), np.uint8)
    for i in range(0, len(a), 20000):
        idx[i:i + 20000] = index_images(a[perm[i:i + 20000]], lut)
    idx[val_n:].tofile(f"{out}/train.u8"); idx[:val_n].tofile(f"{out}/val.u8")
    np.save(f"{out}/palette.npy", pal)
    n_words = 0
    if caps is not None:
        vocab, ids = build_words(caps)
        ids, n_words = ids[perm], len(vocab)
        json.dump(vocab, open(f"{out}/words.json", "w"))
        np.save(f"{out}/train_words.npy", ids[val_n:]); np.save(f"{out}/val_words.npy", ids[:val_n])
        open(f"{out}/val_captions.txt", "w").write("\n".join(caps[k] for k in perm[:val_n]))
    ref = a[perm[:2000]]
    err = np.abs(pal[idx[:2000]].astype(int) - ref[..., :3].reshape(2000, -1, 3))
    if alpha: err = err[ref[..., 3].reshape(2000, -1) >= 128]
    meta = {"S": S, "P": len(pal), "train": int(len(a) - val_n), "val": int(val_n), "n_words": n_words, "W": 8,
            "transparent": bool(alpha), "see_through_share": round(float((idx[:20000] == 0).mean()), 3) if alpha else 0,
            "palette_mean_abs_error": round(float(err.mean()), 2)}
    json.dump(meta, open(f"{out}/meta.json", "w"))
    print(meta)
    return meta


class Pictures:
    def __init__(self, d):
        import torch
        self.meta = json.load(open(f"{d}/meta.json"))
        n = self.meta["S"] ** 2
        self.train = np.memmap(f"{d}/train.u8", np.uint8, mode="r").reshape(-1, n)      # shared between runs
        self.val = np.fromfile(f"{d}/val.u8", np.uint8).reshape(-1, n)
        self.pal = np.load(f"{d}/palette.npy")
        W = self.meta.get("W", 8)                 # the files may hold more candidate words per picture than a prompt has slots
        self.tl_all = np.load(f"{d}/train_words.npy") if os.path.exists(f"{d}/train_words.npy") else None
        self.tl = self.tl_all[:, :W] if self.tl_all is not None else None
        self.vl = np.load(f"{d}/val_words.npy")[:, :W] if os.path.exists(f"{d}/val_words.npy") else None
        self.words = json.load(open(f"{d}/words.json")) if self.tl is not None else None
        self.val_caps = open(f"{d}/val_captions.txt").read().split("\n") if self.tl is not None else None
        self.torch = torch
        self.boost = None                             # optional: pictures of named things, shown more often, their noun always in the prompt
        if os.path.exists(f"{d}/boost.npz") and self.meta.get("boost_share"):
            z = np.load(f"{d}/boost.npz"); self.boost = (z["rows"], z["word"], z["p"], float(self.meta["boost_share"]))

    def batch(self, rng, B, flip):
        t = self.torch
        i = rng.integers(0, len(self.train), B); noun = np.zeros(B, np.int64)
        if self.boost is not None:
            rows, word, p, share = self.boost
            nb = int(rng.binomial(B, share)); k = rng.choice(len(rows), nb, p=p)
            i[:nb] = rows[k]; noun[:nb] = word[k]
        o = np.argsort(i); i, noun = i[o], noun[o]
        x = self.train[i]
        if flip:
            S = self.meta["S"]
            f = rng.random(B) < 0.5
            x = np.where(f[:, None], x.reshape(B, S, S)[:, :, ::-1].reshape(B, -1), x)
        if self.tl is None: return t.from_numpy(x.astype(np.int64)), None
        w = self.tl_all[i].astype(np.int64)
        if w.shape[1] > self.tl.shape[1] or self.meta.get("word_subsets"):     # a random 1..W of the picture's words, so short prompts are normal
            W = self.tl.shape[1]
            r = rng.random(w.shape) + 0.05 * np.arange(w.shape[1])[None] + 9.0 * (w == 0)     # earlier words (the subject) are kept more often
            o = np.argsort(r, 1)[:, :W]
            w = np.take_along_axis(w, o, 1) * (np.arange(W)[None] < rng.integers(1, W + 1, B)[:, None])
        if noun.any():
            b = noun > 0
            w[b] = np.concatenate([noun[b, None], np.where(w[b] == noun[b, None], 0, w[b])], 1)[:, :w.shape[1]]   # noun first, the rest kept
        return t.from_numpy(x.astype(np.int64)), t.from_numpy(w)

    def val_batches(self, n, B):
        t = self.torch
        for i in range(0, min(n, len(self.val)), B):
            yield t.from_numpy(self.val[i:i + B].astype(np.int64)), (t.from_numpy(self.vl[i:i + B].astype(np.int64)) if self.vl is not None else None)

    def sheet(self, idx, cols=16, scale=3, gap=2):
        from PIL import Image
        S = self.meta["S"]
        a = self.pal[np.asarray(idx)].reshape(-1, S, S, 3)
        rows = -(-len(a) // cols)
        im = Image.new("RGB", (cols * (S * scale + gap) + gap, rows * (S * scale + gap) + gap), (16, 19, 28))
        for k, p in enumerate(a):
            im.paste(Image.fromarray(p).resize((S * scale, S * scale), Image.NEAREST), (gap + (k % cols) * (S * scale + gap), gap + (k // cols) * (S * scale + gap)))
        return im


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf"); ap.add_argument("--config"); ap.add_argument("--npy"); ap.add_argument("--out", required=True)
    ap.add_argument("--wds"); ap.add_argument("--shards", type=int, default=8); ap.add_argument("--half")
    ap.add_argument("--image-col", default="image"); ap.add_argument("--caption-col")
    ap.add_argument("--split", default="train"); ap.add_argument("--S", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--val", type=int, default=4096)
    a = ap.parse_args()
    if a.wds:
        arr, labs = from_wds(a.wds, a.shards, a.S), None
    elif a.npy:
        arr, labs = np.load(a.npy), None
    else:
        arr, labs = from_hf(a.hf, a.image_col, a.caption_col, a.S, a.limit, a.split, a.config)
    write(a.out, arr, labs, a.S, a.val)
    if a.half: write(a.half, halve(arr), labs, a.S // 2, a.val)
