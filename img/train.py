"""Train one picture model.

    python3 -m img.train --set name=x data=img/data/cifar10 steps=2000 ... [--device cuda]

Writes img/runs/<name>/: log.jsonl, samples_<step>.png, result.json, model.pt, best.pt; resumes from ckpt.pt.
The score is bits per pixel on held-out pictures (8.0 = guessing among 256 colours; lower is better).
"""
from __future__ import annotations

import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train.model import QLinear
from .data import Pictures
from .model import ImgConfig, ImgModel, gba_cost, config_dict

RUNS = Path(__file__).resolve().parent / "runs"


@torch.no_grad()
def evaluate(model, data, cfg, device):
    model.eval()
    tot, n = 0.0, 0
    for x, c in data.val_batches(cfg.val_images, cfg.batch):
        lp, cells, _ = model(x.to(device), c.to(device) if (c is not None and cfg.n_words) else None)
        tot += -lp.gather(-1, cells[..., None]).sum().item(); n += cells.numel()
    model.train()
    return tot / n / math.log(2)


def route_counts(model, data, cfg):
    x = torch.from_numpy(np.asarray(data.train[:20000]).astype(np.int64))
    c = torch.from_numpy(data.tl[:20000].astype(np.int64)) if (cfg.n_words and data.tl is not None) else None
    return torch.bincount(model.routes(model.cells(x), c).flatten(), minlength=cfg.n_cells * cfg.RC).float() + 1


def prompts(data, cfg, n, device):
    """Held-out captions to draw: (word ids or None, caption texts or None)."""
    if not cfg.n_words: return None, None
    return torch.from_numpy(data.vl[:n].astype(np.int64)).to(device), data.val_caps[:n]


def run(cfg: ImgConfig, device="cpu"):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    data = Pictures(cfg.data)
    assert data.meta["S"] == cfg.S and data.meta["P"] == cfg.P
    if cfg.n_words: assert cfg.n_words == data.meta["n_words"] and cfg.W == data.meta["W"]
    counts = route_counts(ImgModel(ImgConfig(D=32, L=2, DK=32, DV=16, E=2, HD=16, DH=16, HH=16, att_layers=[1], S=cfg.S, C=cfg.C)), data, cfg)
    model = ImgModel(cfg, counts, data.pal / 255.0).to(device)
    vth = (lambda n: model.thumb_of(torch.from_numpy(data.val[:n].astype(np.int64)).to(device))) if cfg.F else (lambda n: None)
    tables = {id(p) for p in model.table_params()}
    mats = [m.weight for m in model.modules() if isinstance(m, QLinear)]
    mat_ids = {id(p) for p in mats}
    other = [p for p in model.parameters() if id(p) not in tables and id(p) not in mat_ids]
    qm = cfg.wq_lr_mult if cfg.wq != "fp" else 1.0
    opt = torch.optim.AdamW([{"params": mats, "weight_decay": cfg.wd, "mult": qm},
                             {"params": model.table_params(), "weight_decay": 0.0, "mult": cfg.table_lr_mult},
                             {"params": other, "weight_decay": 0.0, "mult": 1.0}], lr=cfg.lr, betas=(0.9, 0.95))
    out = RUNS / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    ck, start, best, done_s = out / "ckpt.pt", 0, float("inf"), 0.0
    if cfg.init and not ck.exists():                                    # fine-tuning: the given weights, a fresh optimiser
        model.load_state_dict(torch.load(cfg.init, map_location="cpu"))
    if ck.exists():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        assert c["steps"] == cfg.steps
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        rng.bit_generator.state = c["rng"]; torch.set_rng_state(c["torch_rng"])
        start, best, done_s = c["step"] + 1, c["best"], c["elapsed"]
    log = open(out / "log.jsonl", "a" if start else "w")
    if cfg.init and not start:                                         # fine-tuning: the starting point's score, which also
        best = evaluate(model, data, cfg, device)                      # stays the best unless training beats it
        torch.save(model.state_dict(), out / "best.pt")
        log.write(json.dumps({"step": -1, "val_bits": round(best, 4), "init": cfg.init}) + "\n"); log.flush()
    t0, ent = time.time() - done_s, None
    gen = torch.Generator(device=device)
    for step in range(start, cfg.steps):
        p = step / cfg.steps
        lr = cfg.lr * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * p))
        for g in opt.param_groups: g["lr"] = lr * g["mult"]
        x, c = data.batch(rng, cfg.batch, cfg.flip)
        if cfg.n_words:
            c = torch.where(torch.from_numpy(rng.random(cfg.batch) < cfg.cls_drop)[:, None], torch.zeros_like(c), c).to(device)
        else:
            c = None
        lp, cells, e = model(x.to(device), c)
        loss = -lp.gather(-1, cells[..., None]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ent = e.item() if e is not None else None
        last = step == cfg.steps - 1
        do_eval = (step and step % cfg.eval_every == 0) or last
        if step % 50 == 0 or do_eval:
            rec = {"step": step, "bits": round(loss.item() / math.log(2), 4), "lr": lr, "elapsed": round(time.time() - t0, 1)}
            if do_eval:
                rec["val_bits"] = round(evaluate(model, data, cfg, device), 4)
                if rec["val_bits"] < best:
                    best = rec["val_bits"]; torch.save(model.state_dict(), out / "best.pt")
                model.eval()
                cl, _ = prompts(data, cfg, 64, device)
                gen.manual_seed(1234)                                  # same dice for every sheet: changes are the model's
                data.sheet(model.sample(64, device, cl, gen=gen, thumb=vth(64)).cpu().numpy(), cols=16, scale=96 // cfg.S).save(out / f"samples_{step:06d}.png")
                model.train()
            log.write(json.dumps(rec) + "\n"); log.flush()
        if cfg.ckpt_every and step and step % cfg.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "steps": cfg.steps,
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "best": best,
                        "elapsed": time.time() - t0}, out / "ckpt.tmp")
            os.replace(out / "ckpt.tmp", ck)
    wall = time.time() - t0
    val = evaluate(model, data, cfg, device)
    model.eval()
    ns = min(256, len(data.val))
    cl, caps = prompts(data, cfg, ns, device)
    gen.manual_seed(1234)
    data.sheet(model.sample(ns, device, cl, gen=gen, thumb=vth(ns)).cpu().numpy(), cols=16, scale=96 // cfg.S).save(out / "samples_final.png")
    if caps: (out / "samples_final.txt").write_text("\n".join(f"{i // 16 + 1},{i % 16 + 1}: {c}" for i, c in enumerate(caps)))
    if cfg.n_words:                                                    # the same dice with empty prompts, to see what the words add
        gen.manual_seed(1234)
        data.sheet(model.sample(ns, device, torch.zeros_like(cl), gen=gen, thumb=vth(ns)).cpu().numpy(), cols=16, scale=96 // cfg.S).save(out / "samples_final_noprompt.png")
    res = {"name": cfg.name, "val_bits_per_pixel": round(val, 4), "best_val_bits_per_pixel": round(min(best, val), 4),
           "steps": cfg.steps, "pictures_seen": cfg.steps * cfg.batch, "wall_s": round(wall, 1),
           "pictures_per_s": round(cfg.steps * cfg.batch / max(wall, 1e-9)), "params": model.counts(),
           "expert_usage_entropy": ent, "gba": gba_cost(cfg), "device": device, "config": config_dict(cfg)}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    torch.save(model.state_dict(), out / "model.pt")
    return res


def parse(sets):
    d = {}
    for o in sets:
        k, v = o.split("=", 1)
        assert k in ImgConfig.__dataclass_fields__, f"unknown config key {k!r}"
        try: d[k] = json.loads(v)
        except ValueError: d[k] = v
    return ImgConfig(**d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    r = run(parse(a.set), a.device)
    print(json.dumps({k: r[k] for k in ("name", "val_bits_per_pixel", "pictures_per_s", "wall_s", "expert_usage_entropy", "gba")}))
