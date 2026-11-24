"""Make the training pictures with FLUX.1-schnell (Apache 2.0), one process per GPU.

    python3 -m img.flux_gen <jobs.jsonl> <out dir> <gpu index> <number of gpus> [res=512] [batch=8] [chunk=2000] [steps=4]
                            [repo=unsloth/FLUX.1-schnell] [seq=64] [limit=0] [group=chunks per encoding round]

Each process takes the jobs whose i % <number of gpus> == <gpu index>, in groups of chunks: the two text encoders
turn a group's prompts into embeddings kept on disk, then the image model draws them a chunk at a time. Every picture is kept at 128 x 128 (box-filtered from res),
in out/shard_<gpu>_<chunk>.npz (ids, img). A finished chunk is never made again, so a stopped run carries on.
Log lines: pictures per second, so the full run's time can be worked out after the first chunk.
    python3 -m img.flux_gen fetch          download the weights first (an ungated copy, checked file by file against
                                           the official fingerprints in img/flux_official.json; no login needed)
Needs: pip install diffusers transformers accelerate sentencepiece protobuf.
"""
from __future__ import annotations

import gc, json, os, sys, time
import numpy as np


def main(jobs, out, gi, ng, *opts):
    import torch
    from diffusers import FluxPipeline
    from PIL import Image
    o = dict(a.split("=", 1) for a in opts)
    res, B, chunk, steps = int(o.get("res", 512)), int(o.get("batch", 8)), int(o.get("chunk", 2000)), int(o.get("steps", 4))
    repo, seq, limit = o.get("repo", MIRROR), int(o.get("seq", 64)), int(o.get("limit", 0))
    gi, ng = int(gi), int(ng)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    J = [j for j in map(json.loads, open(jobs)) if j["i"] % ng == gi]
    if limit: J = J[:limit]
    os.makedirs(out, exist_ok=True)
    todo = [(c, J[s:s + chunk]) for c, s in enumerate(range(0, len(J), chunk)) if not os.path.exists(f"{out}/shard_{gi:02d}_{c:04d}.npz")]
    print(f"gpu {gi}: {len(J)} pictures, {len(todo)} chunks to make", flush=True)
    if not todo: return
    # Groups of chunks (group= chunks, 8,000 pictures by default): the text encoders turn a group's prompts into
    # embeddings kept on disk (float16, 0.5 MB per picture at seq=64), then the image model draws them. The encoders and
    # the image model do not fit in 32 GB together, so they take turns; a group's embeddings are deleted after it.
    G = int(o.get("group", max(1, 8000 // chunk)))
    made, t_start = 0, time.time(); total = sum(len(jj) for _, jj in todo)
    for g0 in range(0, len(todo), G):
        group = todo[g0:g0 + G]
        rest = [j for _, jj in group for j in jj]
        ef, pf = f"{out}/emb_{gi:02d}.npy", f"{out}/pool_{gi:02d}.npy"
        t0 = time.time()
        txt = FluxPipeline.from_pretrained(repo, transformer=None, vae=None, torch_dtype=dt).to(dev)
        E = Pl = None
        with torch.no_grad():
            for s in range(0, len(rest), 32):
                ps = [j["prompt"] for j in rest[s:s + 32]]
                e, p, _ = txt.encode_prompt(prompt=ps, prompt_2=ps, device=dev, num_images_per_prompt=1, max_sequence_length=seq)
                if E is None:                      # (FLUX.1: seq x 4096 and 768)
                    E = np.lib.format.open_memmap(ef, "w+", np.float16, (len(rest),) + tuple(e.shape[1:]))
                    Pl = np.lib.format.open_memmap(pf, "w+", np.float16, (len(rest),) + tuple(p.shape[1:]))
                E[s:s + len(ps)] = e.float().cpu().numpy(); Pl[s:s + len(ps)] = p.float().cpu().numpy()
        E.flush(); Pl.flush()
        del txt, e, p; gc.collect()                # (gc: a reference cycle can keep the encoders on the GPU otherwise)
        if dev == "cuda": torch.cuda.empty_cache()
        print(f"gpu {gi}: {len(rest)} prompts encoded in {time.time() - t0:.0f} s", flush=True)
        pipe = FluxPipeline.from_pretrained(repo, text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None, torch_dtype=dt).to(dev)
        pipe.set_progress_bar_config(disable=True)
        r = 0
        for c, jj in group:
            t1 = time.time(); ids, ims = [], []
            for s in range(0, len(jj), B):
                part = jj[s:s + B]
                gens = [torch.Generator(device=dev).manual_seed(j["seed"]) for j in part]
                pe = torch.from_numpy(np.ascontiguousarray(E[r + s:r + s + len(part)])).to(dev, dt)
                pp = torch.from_numpy(np.ascontiguousarray(Pl[r + s:r + s + len(part)])).to(dev, dt)
                with torch.no_grad():
                    a = pipe(prompt_embeds=pe, pooled_prompt_embeds=pp, num_inference_steps=steps, guidance_scale=0.0, height=res, width=res,
                             generator=gens, output_type="np", max_sequence_length=seq).images
                for j, x in zip(part, a):
                    im = Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))
                    ims.append(np.asarray(im.resize((128, 128), Image.BOX))); ids.append(j["i"])
            r += len(jj)
            np.savez(f"{out}/shard_{gi:02d}_{c:04d}.tmp.npz", ids=np.array(ids, np.int64), img=np.stack(ims))
            os.replace(f"{out}/shard_{gi:02d}_{c:04d}.tmp.npz", f"{out}/shard_{gi:02d}_{c:04d}.npz")
            made += len(jj); t2 = time.time()
            mem = f"; GPU memory peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB (reserved {torch.cuda.max_memory_reserved() / 2**30:.1f})" if dev == "cuda" else ""
            print(f"gpu {gi} chunk {c}: {len(jj)} pictures in {t2 - t1:.0f} s = {len(jj) / max(t2 - t1, 1e-9):.2f} pictures/s; "
                  f"{made} of {total} made in {(t2 - t_start) / 60:.1f} min{mem}", flush=True)
        del E, Pl, pipe; gc.collect(); os.remove(ef); os.remove(pf)
        if dev == "cuda": torch.cuda.empty_cache()


MIRROR = "unsloth/FLUX.1-schnell"     # an ungated copy; every file is checked against the official fingerprints by verify()
PATTERNS = ["model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*", "transformer/*", "vae/*"]


def fetch(repo=MIRROR):
    """Download the diffusers files (not the 24 GB single-file copy of the same weights), then verify them."""
    from huggingface_hub import snapshot_download
    p = snapshot_download(repo, max_workers=16, allow_patterns=PATTERNS)
    print(repo, "at", p, flush=True)
    verify(p)


def verify(path):
    """Every file must match black-forest-labs/FLUX.1-schnell (img/flux_official.json): SHA-256 for the weights, the git
    blob hash for small files. Exits 1 on any difference or missing file."""
    import hashlib
    ref = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flux_official.json")))["files"]
    bad = []
    for f, r in sorted(ref.items()):
        fp = os.path.join(path, f)
        if not os.path.exists(fp): bad.append(f + " missing"); continue
        if "sha256" in r:
            h = hashlib.sha256()
            with open(fp, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
            ok = h.hexdigest() == r["sha256"]
        else:
            data = open(fp, "rb").read()
            ok = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == r["git_blob"]
        if not ok: bad.append(f + " DIFFERS")
    print(f"{len(ref) - len(bad)} of {len(ref)} files identical to the official FLUX.1-schnell", flush=True)
    if bad: print("NOT VERIFIED:", bad, flush=True); sys.exit(1)


if __name__ == "__main__":
    if sys.argv[1] == "fetch": fetch(*sys.argv[2:])
    elif sys.argv[1] == "verify": verify(sys.argv[2])
    else: main(*sys.argv[1:])
