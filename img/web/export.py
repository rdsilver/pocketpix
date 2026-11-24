"""Trained picture model -> one binary file the browser engine reads.
  python3 -m img.web.export <model.pt> <result.json> <out.bin> [tables=i8|f32]
File: u32 length of a JSON header, the header, then raw data (little-endian, 4-byte aligned).
  tern  matrix rows as 2 bits per weight (0, +1 = 1, -1 = 2), plus one float per row (matrix scale x row gain)
  i8    table rows as signed bytes, plus one float per row
  f32   small things as they are
Matrices are stored exactly as the model uses them (the quantised weight), so the engine needs no training code."""
import json, sys, struct, numpy as np, torch
from img.model import ImgConfig, ImgModel
from train.model import QLinear
from train import quant

def main(pt, js, out, tables="i8"):
    cfg = ImgConfig(**json.load(open(js))["config"])
    m = ImgModel(cfg); m.load_state_dict(torch.load(pt, map_location="cpu")); m.eval()
    head, blobs, off = {"config": {k: getattr(cfg, k) for k in "S C P G D L NH DK DV att_layers mlp E HD RC topk up_in n_words W cls_layers F DH HH head_blocks".split()}, "t": {}}, [], 0
    def add(a):
        nonlocal off
        b = a.tobytes(); pad = (-len(b)) % 4
        blobs.append(b + b"\0" * pad); o = off; off += len(b) + pad
        return o
    qnames = {}
    for name, mod in m.named_modules():
        if isinstance(mod, QLinear):
            with torch.no_grad():
                w = quant.weight_values(mod.weight, cfg.wq)                     # (..., out, in)
                s = w.abs().flatten(-2).amax(-1)                                # one scale per matrix
                t = torch.round(w / s[..., None, None]).to(torch.int8)
                assert torch.allclose(t.float() * s[..., None, None], w, atol=1e-7) and t.abs().max() <= 1
                rs = s[..., None].expand(w.shape[:-1]).clone()
                if mod.g is not None: rs = rs * mod.g
            t = t.numpy().reshape(-1, w.shape[-1]); n_in = w.shape[-1]
            code = np.where(t < 0, 2, t).astype(np.uint8)
            padc = (-n_in) % 4
            if padc: code = np.pad(code, ((0, 0), (0, padc)))
            c4 = code.reshape(len(code), -1, 4)
            packed = (c4[..., 0] | c4[..., 1] << 2 | c4[..., 2] << 4 | c4[..., 3] << 6).astype(np.uint8)
            head["t"][name] = {"kind": "tern", "shape": list(w.shape), "data": add(packed), "scale": add(rs.numpy().astype("<f4").reshape(-1))}
            qnames[name + ".weight"] = qnames[name + ".g"] = 1
    for k, v in m.state_dict().items():
        if k in qnames: continue
        a = v.numpy()
        if k.endswith(".route"):
            assert a.max() < 256
            head["t"][k] = {"kind": "u8", "shape": list(a.shape), "data": add(a.astype(np.uint8))}
        elif a.ndim == 2 and a.size > 60000 and tables == "i8":
            s = np.abs(a).max(1) / 127.0; s[s == 0] = 1.0
            q = np.rint(a / s[:, None]).astype(np.int8)
            head["t"][k] = {"kind": "i8", "shape": list(a.shape), "data": add(q), "scale": add(s.astype("<f4"))}
        else:
            head["t"][k] = {"kind": "f32", "shape": list(a.shape), "data": add(a.astype("<f4"))}
    hj = json.dumps(head).encode(); hj += b" " * ((-len(hj)) % 4)
    with open(out, "wb") as f:
        f.write(struct.pack("<I", len(hj))); f.write(hj)
        for b in blobs: f.write(b)
    print(out, 4 + len(hj) + off, "bytes")

if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if "=" not in x]; o = dict(x.split("=") for x in sys.argv[1:] if "=" in x)
    main(*a, **o)
