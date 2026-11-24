"""Close-up crops of named things from Open Images V7 (photos CC BY 2.0, boxes CC BY 4.0), added to the picture sets.
  python3 -m img.oi pick            choose boxes: big, single, not a drawing            -> img/data/oi/picks.json
  python3 -m img.oi fetch           download each photo, cut a square round the box     -> img/data/oi/crops.npy (+ labels, colours, credits)
  python3 -m img.oi build <tag>     append the crops to photos64 / p32 / p16 as img/data/p{64,32,16}<tag>, with words and boosting
Each crop's prompt is its noun plus, when one colour clearly dominates the middle of the box, that colour's name."""
import csv, io, json, os, sys, urllib.request
import numpy as np

D = "img/data/oi"
G = "https://storage.googleapis.com/openimages/"
# Open Images class name -> our prompt word (must be in the word list)
CLASSES = {"Cat": "cat", "Dog": "dog", "Bird": "bird", "Horse": "horse", "Cattle": "cow", "Sheep": "sheep", "Elephant": "elephant",
           "Butterfly": "butterfly", "Fish": "fish", "Flower": "flower", "Rose": "rose", "Tree": "tree", "Car": "car", "Bus": "bus",
           "Truck": "truck", "Train": "train", "Boat": "boat", "Airplane": "airplane", "Bicycle": "bicycle", "Motorcycle": "motorcycle",
           "House": "house", "Tower": "tower", "Cake": "cake", "Apple": "apple", "Coffee cup": "cup", "Chair": "chair", "Bed": "bed",
           "Clock": "clock", "Guitar": "guitar", "Sculpture": "statue", "Door": "door", "Window": "window", "Bottle": "bottle",
           "Book": "book", "Hat": "hat", "Lamp": "lamp", "Candle": "candle", "Mushroom": "mushroom", "Tomato": "tomato", "Bread": "bread",
           "Egg (Food)": "egg", "Human face": "face", "Bear": "bear", "Lion": "lion", "Tiger": "tiger", "Zebra": "zebra", "Duck": "duck",
           "Banana": "banana", "Strawberry": "strawberry", "Balloon": "balloon", "Umbrella": "umbrella", "Castle": "castle",
           "Lighthouse": "lighthouse", "Helicopter": "helicopter", "Laptop": "laptop", "Ball": "ball", "Dress": "dress", "Deer": "deer",
           "Monkey": "monkey", "Snake": "snake", "Spider": "spider", "Insect": "insect", "Chicken": "chicken", "Eagle": "eagle",
           "Taxi": "taxi", "Van": "van", "Tent": "tent", "Piano": "piano", "Drum": "drum", "Mobile phone": "phone", "Television": "television",
           "Bowl": "bowl", "Plate": "plate", "Salad": "salad", "Wine": "wine", "Beer": "beer", "Coffee": "coffee", "Fruit": "fruit",
           "Vegetable": "vegetable", "Skyscraper": "skyscraper", "Fountain": "fountain", "Bench": "bench", "Couch": "sofa", "Desk": "desk",
           "Mirror": "mirror", "Vase": "vase", "Flag": "flag", "Man": "man", "Woman": "woman", "Girl": "girl", "Boy": "boy"}
CAP, MIN_AREA = 12000, 0.12


def get(url, path):
    if not os.path.exists(path): urllib.request.urlretrieve(url, path + ".part"); os.rename(path + ".part", path)
    return path


def pick():
    os.makedirs(D, exist_ok=True)
    mid = {r[0]: CLASSES[r[1]] for r in csv.reader(open(get(G + "v5/class-descriptions-boxable.csv", f"{D}/classes.csv"), encoding="utf8")) if r[1] in CLASSES}
    print(len(mid), "of", len(CLASSES), "classes found; missing:", sorted(set(CLASSES.values()) - set(mid.values())), flush=True)
    ok, credit = set(), {}
    for r in csv.DictReader(open(get(G + "2018_04/train/train-images-boxable-with-rotation.csv", f"{D}/images.csv"), encoding="utf8")):
        if r["Rotation"] in ("", "0", "0.0"): ok.add(r["ImageID"]); credit[r["ImageID"]] = (r["Author"], r["License"], r["OriginalLandingURL"])
    per = {}
    for r in csv.DictReader(open(get(G + "v6/oidv6-train-annotations-bbox.csv", f"{D}/boxes.csv"), encoding="utf8")):
        w = mid.get(r["LabelName"])
        if w is None or r["ImageID"] not in ok or r["IsGroupOf"] == "1" or r["IsDepiction"] == "1" or r["IsInside"] == "1": continue
        x0, x1, y0, y1 = float(r["XMin"]), float(r["XMax"]), float(r["YMin"]), float(r["YMax"])
        area = (x1 - x0) * (y1 - y0)
        if area < MIN_AREA or r["IsTruncated"] == "1" and area < 0.3: continue
        per.setdefault(w, []).append((area - 0.2 * (r["IsOccluded"] == "1"), r["ImageID"], x0, x1, y0, y1))
    picks, rng = [], np.random.default_rng(0)
    for w, v in sorted(per.items()):
        v = [v[i] for i in rng.permutation(len(v))[:CAP]]
        picks += [(w,) + b[1:] for b in v]; print(f"{w}: {len(per[w])} boxes, kept {len(v)}", flush=True)
    json.dump({"picks": picks, "credit": {p[1]: credit[p[1]] for p in picks}}, open(f"{D}/picks.json", "w"))
    print(len(picks), "crops chosen", flush=True)


def colour_of(a):
    """Name of the colour that fills most of an RGB patch, or '' when none clearly does."""
    import colorsys
    a = a.reshape(-1, 3).astype(np.float32) / 255
    mx, mn = a.max(1), a.min(1); v = mx; s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == a[:, 0], ((a[:, 1] - a[:, 2]) / d) % 6, np.where(mx == a[:, 1], (a[:, 2] - a[:, 0]) / d + 2, (a[:, 0] - a[:, 1]) / d + 4)) * 60
    name = np.full(len(a), "", dtype=object)
    grey = s < 0.18
    name[grey & (v > 0.8)] = "white"; name[grey & (v < 0.22)] = "black"; name[grey & (v >= 0.22) & (v <= 0.8)] = "grey"
    name[~grey & (v < 0.15)] = "black"
    c = ~grey & (v >= 0.15) & (s >= 0.3)
    for lo, hi, n in ((0, 14, "red"), (14, 40, "orange"), (40, 68, "yellow"), (68, 165, "green"), (165, 260, "blue"), (260, 300, "purple"), (300, 340, "pink"), (340, 360, "red")):
        name[c & (h >= lo) & (h < hi)] = n
    name[c & (h >= 10) & (h < 45) & (v < 0.55)] = "brown"
    name[c & ((h >= 340) | (h < 14)) & (s < 0.55) & (v > 0.75)] = "pink"
    names, counts = np.unique(name, return_counts=True)
    k = counts.argmax()
    return names[k] if names[k] and counts[k] >= 0.5 * len(a) else ""


def one(p):
    from PIL import Image
    w, iid, x0, x1, y0, y1 = p
    try:
        raw = urllib.request.urlopen(f"https://s3.amazonaws.com/open-images-dataset/train/{iid}.jpg", timeout=30).read()
        im = Image.open(io.BytesIO(raw)); im.draft("RGB", (512, 512)); im = im.convert("RGB")
        W, H = im.size
        bx0, bx1, by0, by1 = x0 * W, x1 * W, y0 * H, y1 * H
        side = min(max(bx1 - bx0, by1 - by0) * 1.12, W, H)
        if side < 80: return None
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        l = min(max(cx - side / 2, 0), W - side); t = min(max(cy - side / 2, 0), H - side)
        crop = im.crop((round(l), round(t), round(l + side), round(t + side))).resize((64, 64), Image.LANCZOS)
        a = np.asarray(crop)
        # the middle half of the box, in crop coordinates, names the colour
        u0, u1 = (bx0 - l) / side * 64, (bx1 - l) / side * 64; v0, v1 = (by0 - t) / side * 64, (by1 - t) / side * 64
        qx, qy = (u1 - u0) / 4, (v1 - v0) / 4
        mid = a[max(int(v0 + qy), 0):max(int(v1 - qy), 1), max(int(u0 + qx), 0):max(int(u1 - qx), 1)]
        return a, colour_of(mid) if mid.size else ""
    except Exception:
        return None


def fetch():
    from multiprocessing import Pool
    z = json.load(open(f"{D}/picks.json")); picks = z["picks"]
    if len(sys.argv) > 2: picks = picks[:int(sys.argv[2])]
    out, lab, col, ids = [], [], [], []
    with Pool(160) as pool:
        for i, (p, r) in enumerate(zip(picks, pool.imap(one, picks, chunksize=8))):
            if r is not None: out.append(r[0]); lab.append(p[0]); col.append(r[1]); ids.append(p[1])
            if i % 20000 == 0: print(i, len(out), flush=True)
    np.save(f"{D}/crops.npy", np.stack(out)); json.dump({"noun": lab, "colour": col, "id": ids}, open(f"{D}/crops.json", "w"))
    json.dump({i: z["credit"][i] for i in sorted(set(ids))}, open(f"{D}/credits.json", "w"))
    from collections import Counter
    print(len(out), "crops;", "colours:", Counter(col).most_common(), flush=True)
    from PIL import Image
    rng = np.random.default_rng(0); k = rng.permutation(len(out))[:240]
    sheet = np.stack(out)[k].reshape(12, 20, 64, 64, 3).transpose(0, 2, 1, 3, 4).reshape(12 * 64, 20 * 64, 3)
    Image.fromarray(sheet).save("img/runs/oi_crops.png")
    open("img/runs/oi_crops.txt", "w").write("\n".join(f"{j // 20 + 1},{j % 20 + 1}: {col[i]} {lab[i]}" for j, i in enumerate(k)))


def build():
    import torch
    from .data import palette_lut, index_images
    tag = sys.argv[2]
    crops = np.load(f"{D}/crops.npy"); z = json.load(open(f"{D}/crops.json"))
    pal = np.load("img/data/photos64/palette.npy"); x64 = index_images(crops, palette_lut(pal))
    palt = torch.from_numpy(pal).float().cuda() / 255.0

    def half(x, S):                                  # same rule as img.thumbs
        T = S // 2; o = []
        for i in range(0, len(x), 8192):
            rgb = palt[torch.from_numpy(x[i:i + 8192].astype(np.int64)).cuda()].view(-1, T, 2, T, 2, 3).mean((2, 4)).view(-1, T * T, 3)
            o.append(torch.cdist(rgb, palt[None].expand(len(rgb), -1, -1)).argmin(-1).cpu().numpy().astype(np.uint8))
        return np.concatenate(o)
    x32 = half(x64, 64); x16 = half(x32, 32)
    words = json.load(open("img/data/p16s/words.json")); ix = {w: i for i, w in enumerate(words)}
    old = np.load("img/data/p16s/train_words.npy"); n0 = len(old)
    add = np.zeros((len(crops), old.shape[1]), old.dtype)
    add[:, 0] = [ix[w] for w in z["noun"]]; add[:, 1] = [0 if w in ("man", "woman", "girl", "boy", "face") else ix.get(c, 0) for w, c in zip(z["noun"], z["colour"])]
    b = np.load("img/data/p16s/boost.npz")
    nouns = add[:, 0].astype(np.int64); cnt = np.bincount(np.concatenate([b["word"], nouns]), minlength=len(words))
    rows = np.concatenate([b["rows"], n0 + np.arange(len(crops))]); word = np.concatenate([b["word"], nouns])
    strict = np.concatenate([b["p"] * 0 + 0.5, np.ones(len(crops))])       # crops count like close-ups
    p = (1 + strict) / np.sqrt(cnt[word]); p /= p.sum()
    for S, src, x in ((64, "p64s", x64), (32, "p32s", x32), (16, "p16s", x16)):
        d = f"img/data/p{S}{tag}"; os.makedirs(d, exist_ok=True)
        with open(f"{d}/train.u8", "wb") as f:
            a = np.memmap(f"img/data/{src}/train.u8", np.uint8, mode="r")
            for i in range(0, len(a), 1 << 28): f.write(a[i:i + (1 << 28)].tobytes())
            f.write(x.tobytes())
        for fn in ("val.u8", "palette.npy", "words.json", "val_words.npy", "val_captions.txt", "val_captions_full.txt"):
            if os.path.lexists(f"{d}/{fn}"): os.remove(f"{d}/{fn}")
            os.symlink(os.path.realpath(f"img/data/{src}/{fn}"), f"{d}/{fn}")
        np.save(f"{d}/train_words.npy", np.concatenate([old, add])); np.savez(f"{d}/boost.npz", rows=rows, word=word, p=p)
        m = json.load(open(f"img/data/{src}/meta.json")); m.update(train=n0 + len(crops), boost_share=float(sys.argv[3]) if len(sys.argv) > 3 else 0.3)
        json.dump(m, open(f"{d}/meta.json", "w")); print(d, m, flush=True)
    from .data import Pictures
    dd = Pictures(f"img/data/p16{tag}"); xb, wb = dd.batch(np.random.default_rng(0), 4096, True)
    nb = int((wb[:, 0] > 0).sum()); print("sample prompts:", [" ".join(words[i] for i in r if i) for r in wb[:24].tolist()])
    dd.sheet(xb[:96].numpy(), cols=16, scale=3).save(f"img/runs/oi_batch_{tag}.png")


if __name__ == "__main__":
    {"pick": pick, "fetch": fetch, "build": build}[sys.argv[1]]()
