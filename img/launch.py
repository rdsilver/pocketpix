"""Picture-model runs, one per GPU.   GPUS=0,1,2,3 python3 img/launch.py <data dir> <stage> [name prefix ...]
Stages: check (short shape check, about one pass over 1.2M pictures) and final (several passes)."""
import json, os, queue, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data, stage, only = sys.argv[1], sys.argv[2], sys.argv[3:]
meta = json.load(open(f"{data}/meta.json"))
passes = {"check": 1.5, "final": 8}[stage]
B = 128 if meta["S"] < 64 else 64
steps = int(meta["train"] * passes / B)
COMMON = {"data": data, "S": meta["S"], "batch": B, "steps": steps, "warmup": 1000, "eval_every": max(1000, steps // 10 // 50 * 50),
          "val_images": 4096, "ckpt_every": 5000, "n_words": meta["n_words"], "flip": not meta["n_words"]}
KOKO = {}                                                                     # D=256, 8 layers: Koko's body
D192 = {"D": 192, "DK": 192, "DV": 96, "HD": 96}
D160 = {"D": 160, "DK": 160, "DV": 80, "HD": 96, "L": 6, "att_layers": [1, 3, 5]}
FP = {"wq": "fp", "wq_lr_mult": 1, "row_gain": False}
C8 = {"C": 8}
if meta["S"] >= 64:        # 64 x 64: cells of 8 x 8 pixels keep the body at 64 passes; times are cartridge estimates
    RUNS = [
        ("koko_c8", {**KOKO, **C8}),               # about 35 s per picture, 23 MB
        ("d192_c8", {**D192, **C8}),               # about 29 s, 14 MB
        ("d160_l6_c8", {**D160, **C8}),            # about 25 s, 10 MB
        ("koko_c8_fp", {**KOKO, **C8, **FP}),      # full-precision twin: what 2-bit weights cost
        ("d192_c8_head128", {**D192, **C8, "DH": 128, "HH": 256}),   # the head writes 64 pixels per pass here: is it the weak part?
        ("d192_c4", {**D192}),                     # 256 body passes, about 54 s: how much better?
    ]
else:
    RUNS = [
        ("koko", KOKO),                        # 32 x 32: about 19 s per picture on the cartridge (estimate), 18 MB
        ("d192", D192),                        # about 13 s, 10.5 MB
        ("d160_l6", D160),                     # about 10 s, 7 MB
        ("koko_fp", {**KOKO, **FP}),           # full-precision twin: what 2-bit weights cost
        ("d192_head128", {**D192, "DH": 128, "HH": 256}),      # is the tiny head the weak part?
        ("d192_cell8", {**D192, **C8}),                        # 16 body passes instead of 64 (about 7 s): how much worse?
        ("d192_dense", {**D192, "mlp": "dense", "HD": 768}),   # no routed experts
    ]
GPUS = queue.Queue()
for g in os.environ.get("GPUS", "0").split(","): GPUS.put(g)


def go(r):
    name, over = r
    name = f"{stage}_{name}"
    if only and not any(name.startswith(o) or name.startswith(f"{stage}_{o}") for o in only): return name, "skipped"
    if os.path.exists(f"img/runs/{name}/result.json"): return name, "already done"
    g = GPUS.get()
    try:
        sets = [f"{k}={json.dumps(v)}" for k, v in {**COMMON, **over, "name": name}.items()]
        os.makedirs("img/runs", exist_ok=True)
        p = subprocess.run([sys.executable, "-m", "img.train", "--set", *sets], stdout=open(f"img/runs/{name}.out", "w"),
                           stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": g})
        return name, p.returncode
    finally:
        GPUS.put(g)


with ThreadPoolExecutor(GPUS.qsize()) as ex:
    for name, rc in ex.map(go, RUNS): print(name, rc, flush=True)
