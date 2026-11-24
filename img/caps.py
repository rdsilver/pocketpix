"""Add prompt words to picture sets already made by `python3 -m img.data --wds`: reads the same shards again in the
same order, keeps each photo's caption, and writes words.json / *_words.npy / val_captions.txt.
    python3 -m img.caps <dataset> <n_shards> <S> <out dir> [<out dir> ...]
The picture order is reproduced exactly (same shard list, same skipped files, same shuffle seed); the count is checked."""
import io, json, os, sys
from collections import Counter
import numpy as np
from .data import words_of, STOP

N_WORDS, W, CAND, MAX_SHARE = 4096, 8, 16, 0.12      # a prompt has W slots; CAND candidate words are kept per picture
# very common words that are still worth typing: colours, people, the big things in a scene
KEEP = set("""red orange yellow green blue purple pink brown black white gray grey golden silver dark bright
man woman person people child boy girl baby face sky clouds water sea tree trees grass building buildings street road
car wall window door table snow sand rock rocks mountain mountains field flowers flower light night""".split())
CACHE = "img/data/captions_cache.json"
import re
ANCHOR = re.compile(r"\b(photograph|photo|image|shot|picture|view|portrait|close-up|closeup)\b[^.]{0,60}?\b(of|shows|showing|captures|capturing|depicts|depicting|features|featuring)\b")


def split_caption(text):
    """The captions open with how the photo was taken ("A close-up, eye-level photograph of ..."). Returns the words
    before that opening ends (framing) and the words after it (the subject first)."""
    t = text.lower(); m = ANCHOR.search(t[:200])
    return (words_of(t[:m.end()]), words_of(t[m.end():])) if m else ([], words_of(t))



def _tar_caps(args):
    path, S = args
    import tarfile
    from PIL import Image
    keys, txt = [], {}
    with tarfile.open(path) as t:
        for m in t:
            k, ext = os.path.splitext(m.name)
            if ext.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                try:
                    im = Image.open(io.BytesIO(t.extractfile(m).read()))
                    im.draft("RGB", (S * 2, S * 2)); im.convert("RGB")       # same test as the picture pass: skip what it skipped
                    keys.append(k)
                except Exception:
                    pass
            elif ext == ".txt":
                txt[k] = t.extractfile(m).read().decode("utf8", "replace").strip()
    return [txt.get(k, "") for k in keys]


def main():
    name, n_shards, S, outs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    from multiprocessing import Pool
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfFileSystem, hf_hub_download
    files = sorted(f for f in HfFileSystem().glob(f"datasets/{name}/**/*.tar"))
    files = [files[k] for k in sorted({int(i * len(files) / n_shards) for i in range(n_shards)})] if n_shards < len(files) else files
    workers = min(16, os.cpu_count() or 4)
    if os.path.exists(CACHE):
        caps = json.load(open(CACHE))
    else:
        caps = []
        with Pool(workers) as pool:
            for i in range(0, len(files), workers):
                with ThreadPoolExecutor(8) as tp:
                    paths = list(tp.map(lambda f: hf_hub_download(name, f.split(name + "/", 1)[1], repo_type="dataset"), files[i:i + workers]))
                for c in pool.imap(_tar_caps, [(p, S) for p in paths]): caps += c
                for p in paths:
                    real = os.path.realpath(p); os.remove(real)
                    if os.path.islink(p): os.remove(p)
                print(f"{i + len(paths)}/{len(files)} shards, {len(caps)} captions", flush=True)
        json.dump(caps, open(CACHE, "w"))
    parts = [split_caption(c) for c in caps]
    toks = [a + b for b, a in parts]              # subject words first, framing words last
    df = Counter(w for t in toks for w in set(t))
    ok = [w for w, n in df.most_common() if (n <= MAX_SHARE * len(caps) or w in KEEP) and w not in STOP and len(w) > 2 and not w.isdigit()]
    vocab = ["", "<other>"] + ok[:N_WORDS - 2]
    ix = {w: i for i, w in enumerate(vocab)}
    ids = np.zeros((len(caps), CAND), np.int16)
    short = []
    for i, t in enumerate(toks):
        seen = []
        for w in t[:len(parts[i][1])]:                # the first listed words after the opening, in the caption's order
            if w in ix and w not in seen: seen.append(w)
            if len(seen) == CAND - 2: break
        for w in parts[i][0]:                         # then up to two framing words ("close", "wide", ...)
            if w in ix and w not in seen and w not in ("photograph", "photo", "image", "shot", "picture"): seen.append(w)
            if len(seen) == CAND: break
        ids[i, :len(seen)] = [ix[w] for w in seen]; short.append(" ".join(seen[:W]))
    perm = np.random.default_rng(0).permutation(len(caps))        # the shuffle img.data.write used
    for out in outs:
        meta = json.load(open(f"{out}/meta.json"))
        assert meta["train"] + meta["val"] == len(caps), (out, meta["train"] + meta["val"], len(caps))
        v = meta["val"]
        json.dump(vocab, open(f"{out}/words.json", "w"))
        np.save(f"{out}/train_words.npy", ids[perm][v:]); np.save(f"{out}/val_words.npy", ids[perm][:v])
        open(f"{out}/val_captions.txt", "w").write("\n".join(short[k] for k in perm[:v]))
        open(f"{out}/val_captions_full.txt", "w").write("\n".join(caps[k].replace("\n", " ") for k in perm[:v]))
        json.dump({**meta, "n_words": len(vocab), "W": W}, open(f"{out}/meta.json", "w"))
        print(out, "words", len(vocab), "empty prompts", float((ids[:, 0] == 0).mean()), flush=True)
    print("top words:", vocab[2:60])
    print("pictures per word:", {w: df[w] for w in "cat dog horse bird car boat flower tree red blue green man woman face pizza train".split()})
    print("missing:", [w for w in KEEP if w not in ix])


if __name__ == "__main__":
    main()
