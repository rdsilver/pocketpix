"""Picture models -> the cartridge's data blob (build/pix.bin) and gen/pix_layout.h.
    python3 -m pix.export b16C u32M u64M        (names under img/remote/night)
Integer formats follow engine/DESIGN.md (in the Koko repository): Q12 everywhere, ternary matrices for the 81-entry table kernels as
"T3" words (three groups of four inputs per 32-bit word, see step_words), one Q16 multiplier per matrix (per row
where the model has a row gain).
Differences: matrices with more than 192 inputs are stored in chunks of 192 inputs (the lookup tables for one chunk
fit in fast RAM), chunk after chunk, each chunk row after row; table rows are int8 with a Q16 step per row."""
import json, os, sys, numpy as np, torch
from img.model import ImgConfig, ImgModel
from train.model import QLinear
from train import quant

Q, ONE, CH = 12, 4096, 192
AWIN = {"u32M": 64, "u64M": 64}      # attention looks back this many cells (0 = all); measured cost under 0.2% of bits per pixel
FLAT = {"u32M": 1, "u64M": 2}        # cells whose thumbnail window stays within this (5-bit steps per channel) are painted directly
TAU = {"u32M": (64, 32), "u64M": (32, 32)}     # hidden values up to this are dropped: (body, head); measured with pix/check.py
EXP_BITS, EXP_N = 7, 16 << 7
MF = ("S C D L NH DK DV HD DH HH F NW CLSL ATTMASK E HS HSH WIN NC NS KB TAUB TAUH FLATT AWIN "
      "CELLPOS CELLPOS_ST PREV PREV_ST UP UP_ST TH TH_ST CLS CLS_ST NF HIN HIN_M SLOT SLOT_ST NB NB_ST IN IN_ST HTH HTH_ST "
      "HN H1 H1M YQ SLOT_Y NB_Y IN_Y HTH_Y H2 H2M HF OUTG OUTG_M OUTM OUTM_M PERM NBIX TW TP ROPE_INC LAYERS SEC").split()
# (An approximation with one switch: empty this dict and every layer uses the model's TAU; or PIX_TAUL 0 in pix.c.)
# Later layers stand larger thresholds: per-layer body thresholds found by pix/tausearch.py (greedy, one layer at a
# time), each model's whole list costing under 0.0002 nats of KL and 0.001 bits per pixel on the 48 test pictures.
# (Merged with the first matrix's early exit, W1X, the last layers of u32M and b16O were taken back from 384, 384 and 96
# so that both approximations together stay inside +0.0003 nats of KL per model.)
TAUL = {"b16C": [16, 32, 16, 16, 16, 32, 32, 64], "u32M": [64, 64, 64, 128, 256, 256, 256, 256], "u64M": [32, 64, 128, 288, 192, 128],
        "b16O": [16, 32, 16, 16, 16, 32, 32, 64]}
LF = "N1 N2 Q K V O KG ATT DBIAS MIXC W1 W2 HDL W1M W2M ROUTE CLSL CLSL_ST W1X W1X2 TAU".split()
W1EXIT = json.load(open("pix/w1exit.json")) if os.path.exists("pix/w1exit.json") else None     # tools/w1cal.py: limits for the early exit


def step_bytes(trits):      # (the blob's format before T3, below; kept for reference: the keys made at run time are still step bytes)
    t = np.asarray(trits); assert t.shape[1] % 16 == 0, t.shape
    code = np.where(t > 0, 1, np.where(t < 0, 2, 0)).reshape(t.shape[0], -1, 4)
    idx = code[..., 0] + 3 * code[..., 1] + 9 * code[..., 2] + 27 * code[..., 3]
    st = np.empty_like(idx); st[:, 0] = idx[:, 0] + 1; st[:, 1:] = 81 + idx[:, 1:] - idx[:, :-1]
    return st.astype(np.uint8)


def step_words(trits):
    """Rows of a ternary matrix as "T3" words for the row kernels: three groups of four inputs per 32-bit word.
    Group g's table entry is at pos = 81 g + idx (words from the start of the tables); p is the entry of the group
    before the word (-1 at the start of a row). A word holds  A << 24 | B << 11 | C :
      B = pos(first group) - p    (read as [p, w >> 9]: the bits of A land 32 KB and more above, where fast RAM repeats)
      A = pos(second group) - p   (read as [p, w >> 22] with write-back)
      C = pos(third group) - pos(second group)   (the low byte, with write-back)
    A row of G groups takes (G + 2) // 3 words; when G is not a multiple of 3 the FIRST word is the short one:
    two groups in B and A (C = 0), or one group in A (B = C = 0)."""
    t = np.asarray(trits); assert t.shape[1] % 16 == 0, t.shape
    code = np.where(t > 0, 1, np.where(t < 0, 2, 0)).reshape(t.shape[0], -1, 4)
    idx = code[..., 0] + 3 * code[..., 1] + 9 * code[..., 2] + 27 * code[..., 3]
    G = idx.shape[1]; nw = (G + 2) // 3; f = G - 3 * (nw - 1)
    pos = (idx + 81 * np.arange(G)[None, :]).astype(np.int64)
    pp = np.concatenate([np.full((len(pos), 3 - f), -1, np.int64), pos], 1).reshape(len(pos), nw, 3)     # (the short word's missing groups: -1)
    prev = np.concatenate([np.full((len(pos), 1), -1, np.int64), pp[:, :-1, 2]], 1)
    B, A, C = pp[..., 0] - prev, pp[..., 1] - prev, pp[..., 2] - pp[..., 1]
    if f == 2: B[:, 0] = pp[:, 0, 1] - prev[:, 0]; A[:, 0] = pp[:, 0, 2] - prev[:, 0]; C[:, 0] = 0
    if f == 1: B[:, 0] = 0; A[:, 0] = pp[:, 0, 2] - prev[:, 0]; C[:, 0] = 0
    assert B.min() >= 0 and B.max() < 256 and A.min() >= 1 and A.max() < 256 and C.min() >= 0 and C.max() < 256
    return (A << 24 | B << 11 | C).astype(np.uint32)


def chunked(trits):
    t = np.asarray(trits)
    return np.concatenate([step_words(t[:, a:a + CH]).reshape(-1) for a in range(0, t.shape[1], CH)])


def columns(trits):
    """(rows, n_in) trits -> column lists for the sparse kernel: per input column one word (count of + rows, count of
    - rows, both in words of 4), then the + row numbers and the - row numbers, each padded to whole words with the
    number of the spare slot after the last row. Returns (bytes, offset of each column in words)."""
    t = np.asarray(trits); rows = t.shape[0]; assert rows < 255
    out, offs = [], []
    n = 0
    for c in range(t.shape[1]):
        p, m = np.nonzero(t[:, c] > 0)[0], np.nonzero(t[:, c] < 0)[0]
        pad = lambda a: np.concatenate([a, np.full((-len(a)) % 4, rows)]).astype(np.uint8)
        p, m = pad(p), pad(m)
        offs.append(n); out.append(np.array([len(p) // 4, len(m) // 4, 0, 0], np.uint8)); out.append(p); out.append(m)
        n += 1 + len(p) // 4 + len(m) // 4
    return np.concatenate(out), np.array(offs, np.uint32)


class Blob:
    def __init__(self): self.b = bytearray()
    def add(self, a, dtype):
        while len(self.b) % 4: self.b.append(0)
        o = len(self.b); self.b += np.ascontiguousarray(np.asarray(a).astype(dtype)).tobytes(); return o


def tern(mod, cfg):
    """QLinear -> (trits (..., out, in) int, matrix scale (...,) float, row gain or None)."""
    with torch.no_grad():
        w = quant.weight_values(mod.weight, cfg.wq)
        s = w.abs().flatten(-2).amax(-1)
        t = torch.round(w / s[..., None, None]).long()
        assert torch.allclose(t.float() * s[..., None, None], w, atol=1e-7)
    return t.numpy(), s.numpy().astype(np.float64), (mod.g.detach().numpy().astype(np.float64) if mod.g is not None else None)


def add_model(B, name, hs, hsh):
    d = os.environ.get("PIX_MODELS", "img/remote/night").rstrip("/") + "/"     # PIX_MODELS: another folder of <name>.pt + <name>.json
    cfg = ImgConfig(**json.load(open(d + name + ".json"))["config"])
    m = ImgModel(cfg); m.load_state_dict(torch.load(d + name + ".pt", map_location="cpu")); m.eval()
    sd = {k: v.numpy() for k, v in m.state_dict().items()}
    f = dict.fromkeys(MF, 0)
    hk = cfg.DK // cfg.NH
    f.update(S=cfg.S, C=cfg.C, D=cfg.D, L=cfg.L, NH=cfg.NH, DK=cfg.DK, DV=cfg.DV, HD=cfg.HD, DH=cfg.DH, HH=cfg.HH, F=cfg.F,
             NW=cfg.n_words, CLSL=int(cfg.cls_layers), ATTMASK=sum(1 << l for l in range(cfg.L) if m.bcfg.has_att(l)),
             E=cfg.E if cfg.mlp == "moe" else 0, HS=hs, HSH=hsh, TAUB=TAU.get(name, (16, 32))[0], TAUH=TAU.get(name, (16, 32))[1], FLATT=FLAT.get(name, 0), AWIN=AWIN.get(name, 0), NC=cfg.n_cells, NS=cfg.n_slots, KB=-(-hk // 16) * 4)
    assert cfg.head_blocks == 1 and cfg.topk == 8 and cfg.up_in and cfg.n_words and cfg.P == 256 and cfg.G == 16
    assert cfg.D % 16 == 0 and cfg.D <= CH and cfg.DK <= CH and cfg.DV % 16 == 0 and cfg.DH % 16 == 0 and cfg.HH % 16 == 0 and cfg.HD % 16 == 0

    def table(key):
        a = sd[key].astype(np.float64); s = np.abs(a).max(1) / 127.0; z = s == 0; s[z] = 1.0
        q = np.rint(a / s[:, None]); st = np.rint(s * 65536); st[z] = 0
        assert st.max() < 2 ** 31
        return B.add(q, np.int8), B.add(st, np.int32)
    for fld, key in (("CELLPOS", "cell_pos"), ("PREV", "prev_emb"), ("UP", "up_emb"), ("CLS", "cls_emb"), ("SLOT", "slot_emb"), ("NB", "nb_emb"), ("IN", "in_emb")):
        f[fld], f[fld + "_ST"] = table(key + ".weight")
    if cfg.F:
        f["TH"], f["TH_ST"] = table("th_emb.weight"); f["HTH"], f["HTH_ST"] = table("hth_emb.weight")
        f["WIN"] = m.tw.shape[1]; f["TW"] = B.add(m.tw.numpy(), np.int16); f["TP"] = B.add(m.tp.numpy(), np.int16)
    f["PERM"] = B.add(m.perm.numpy(), np.int16); f["NBIX"] = B.add(m.nb.numpy(), np.int16)
    gq = lambda k: B.add(np.rint(sd[k].astype(np.float64) * ONE), np.int32)
    freq = 1.0 / (10000 ** (np.arange(0, hk, 2) / hk))
    f["ROPE_INC"] = B.add(np.rint(freq / (2 * np.pi) * 2 ** 32), np.uint32)
    lays = []
    for l in range(cfg.L):
        o = dict.fromkeys(LF, 0)
        o["N1"], o["N2"] = gq(f"n1.{l}.g"), gq(f"n2.{l}.g")
        o["TAU"] = TAUL[name][l] if name in TAUL else f["TAUB"]
        a = m.att[l]
        if m.bcfg.has_att(l):
            tq, _, _ = tern(a.q, cfg); tk, _, kg = tern(a.k, cfg); tv, sv, _ = tern(a.v, cfg); to, so, _ = tern(a.o, cfg)
            o["Q"], o["K"], o["V"], o["O"] = B.add(chunked(tq), np.uint32), B.add(chunked(tk), np.uint32), B.add(chunked(tv), np.uint32), B.add(chunked(to), np.uint32)
            o["KG"] = B.add(np.rint(kg * ONE), np.int32)
            c = np.exp(np.clip(sd[f"att.{l}.temp"].astype(np.float64), -3, 3)) / np.sqrt(hk) * 65536
            o["ATT"] = B.add(np.rint(np.concatenate([[sv * 65536, so * 65536], c])), np.int32)
            o["DBIAS"] = B.add(np.rint(sd[f"att.{l}.dbias"].astype(np.float64) * ONE), np.int32)
        else:
            o["MIXC"] = B.add(np.rint(np.concatenate([sd[f"att.{l}.taps"], sd[f"att.{l}.avg"]]).T.astype(np.float64) * ONE), np.int32)
        t1, s1, _ = tern(m.mlp[l].w1, cfg); t2, s2, _ = tern(m.mlp[l].w2, cfg)
        t1, t2 = t1.reshape(-1, cfg.HD, cfg.D), t2.reshape(-1, cfg.D, cfg.HD)
        if cfg.mlp == "moe": o["W2"] = B.add(np.concatenate([chunked(e) for e in t2]), np.uint32)
        else:                                                # hidden units that can never fire or are never read are left out
            live = ((t1[0] != 0).any(1)) & ((t2[0] != 0).any(0)); t1 = t1[:, live]; t2 = t2[:, :, live]; o["HDL"] = int(live.sum())
            o["W2"] = B.add(step_words(t2[0].T), np.uint32)      # one row of steps per hidden unit: the matrix transposed
        o["W1"] = B.add(np.concatenate([chunked(e) for e in t1]), np.uint32)
        if W1EXIT and cfg.mlp != "moe" and name in W1EXIT["limits"]:      # rows whose sum is below these with 28, 24, ... 4 groups to go are dropped
            assert W1EXIT["rem"] == [28, 24, 20, 16, 12, 8, 4]; o["W1X"] = B.add(W1EXIT["limits"][name][l], np.int32)
        if W1EXIT and name in W1EXIT.get("limits2", {}): o["W1X2"] = B.add(W1EXIT["limits2"][name][l], np.int32)     # the same for two streams (their sum, in the tables' units)
        o["W1M"] = B.add(np.rint(np.atleast_1d(s1) * 65536), np.int32); o["W2M"] = B.add(np.rint(np.atleast_1d(s2) * 65536 * 2 ** hs), np.int32)
        if cfg.mlp == "moe": o["ROUTE"] = B.add(sd[f"mlp.{l}.route"], np.uint8)
        if cfg.cls_layers and l: o["CLSL"], o["CLSL_ST"] = table(f"cls_l.{l - 1}.weight")
        lays.append([o[k] for k in LF])
    f["LAYERS"] = B.add(lays, np.uint32); f["NF"] = gq("nf.g"); f["HN"] = gq("hn.0.g"); f["HF"] = gq("hf.g")
    t, s, _ = tern(m.h_in, cfg); f["HIN"] = B.add(chunked(t), np.uint32); f["HIN_M"] = int(round(float(s) * 65536))
    t, s, _ = tern(m.h1[0], cfg); f["H1"] = B.add(chunked(t), np.uint32); f["H1M"] = int(round(float(s) * 65536))
    # the head's first matrix applied to every table row ahead of time (it comes after a norm, which is a per-dimension
    # gain and one scalar, so matrix x gain x row can be summed instead of multiplying per pixel)
    W = t.astype(np.float64).T * float(s); gh = sd["hn.0.g"].astype(np.float64)
    def rows_q(key):
        a = sd[key].astype(np.float64); st = np.abs(a).max(1) / 127.0; st[st == 0] = 1.0
        return np.rint(a / st[:, None]) * (np.rint(st * 65536) / 65536)[:, None]
    ys = {fld: (rows_q(key + ".weight") * gh) @ W for fld, key in (("SLOT", "slot_emb"), ("NB", "nb_emb"), ("IN", "in_emb")) + ((("HTH", "hth_emb"),) if cfg.F else ())}
    ymax = max(np.abs(y).max() for y in ys.values()); yq = min(12, int(np.floor(np.log2(32767 / ymax)))); f["YQ"] = yq
    assert yq >= 7, yq
    yi = {fld: np.rint(y * 2 ** yq) for fld, y in ys.items()}
    if cfg.n_slots > 1:      # (exact) the row of the pixel painted at slot j also carries slot j + 1's row minus slot j's (the
        for j in range(cfg.n_slots - 1):      # engine starts each cell with slot 0's row and adds one row fewer per pixel: pix.c)
            yi["IN"][j * (cfg.P + 1):(j + 1) * (cfg.P + 1)] += yi["SLOT"][j + 1] - yi["SLOT"][j]
        assert np.abs(yi["IN"]).max() <= 32767, np.abs(yi["IN"]).max()
        f["YQ"] = yq | 256       # (bit 8 tells the engine)
    for fld, y in yi.items(): f[fld + "_Y"] = B.add(y, np.int16)
    t, s, _ = tern(m.h2[0], cfg); f["H2"] = B.add(step_words(t.T), np.uint32); f["H2M"] = int(round(float(s) * 65536 * 2 ** hsh))
    for fld, mod in (("OUTG", m.out_g), ("OUTM", m.out_m)):
        t, s, g = tern(mod, cfg); f[fld] = B.add(chunked(t), np.uint32)
        f[fld + "_M"] = B.add(np.rint(s * 65536 * (g if g is not None else np.ones(t.shape[0]))), np.int32)
    gb = json.load(open(d + name + ".json"))["gba"]; f["SEC"] = int(round(gb["seconds_per_picture"] * 1.35))
    return [f[k] for k in MF], cfg


def main(names, hs=4, hsh=2):
    B = Blob(); defs = {}
    defs["O_SIN"] = B.add(np.rint(np.sin(2 * np.pi * np.arange(4096) / 4096) * 16384), np.int16)
    defs["O_EXP"] = B.add(np.rint(np.exp(-np.arange(EXP_N) / (1 << EXP_BITS)) * 65535), np.uint16)
    pal = np.load("img/remote/pal/p16o.npy").astype(np.int64) >> 3
    defs["O_PAL"] = B.add(pal[:, 0] | pal[:, 1] << 5 | pal[:, 2] << 10, np.uint16)
    rgb = np.load("img/remote/pal/p16o.npy").astype(np.int64)
    for nm, c in (("BG", (14, 16, 22)), ("PANEL", (40, 46, 58)), ("FG", (236, 240, 244)), ("DIM", (130, 140, 150)), ("ACC", (98, 179, 211)), ("WARN", (224, 154, 85))):
        defs["PIX_C_" + nm] = int(((rgb - np.array(c)) ** 2).sum(1).argmin())
    words = json.load(open("img/remote/night/words.json"))
    text = b"".join(w.encode() + b"\0" for w in words); off = np.cumsum([0] + [len(w.encode()) + 1 for w in words])[:-1]
    order = np.array(sorted(range(2, len(words)), key=lambda i: words[i]))            # ids in alphabetical order, for the word picker
    defs["O_WTEXT"] = B.add(np.frombuffer(text, np.uint8), np.uint8); defs["O_WOFF"] = B.add(off, np.uint32); defs["O_WSORT"] = B.add(order, np.uint16)
    rows, cfgs = [], []
    for n in names:
        r, c = add_model(B, n, hs, hsh); rows.append(r); cfgs.append(c)
    defs["O_MODELS"] = B.add(rows, np.uint32)
    while len(B.b) % 4: B.b.append(0)
    open(os.environ.get("PIX_OUT", "build/pix.bin"), "wb").write(B.b)
    mx = lambda k: max(getattr(c, k) for c in cfgs)
    with open(os.environ.get("PIX_LAYOUT", "gen/pix_layout.h"), "w") as h:
        h.write("/* generated by pix/export.py */\n#ifndef PIX_LAYOUT_H\n#define PIX_LAYOUT_H\n")
        for k, v in defs.items(): h.write(f"#define {k} {v}u\n" if k.startswith("O_") else f"#define {k} {v}\n")
        h.write(f"#define PIX_NMODELS {len(names)}\n#define PIX_NWORDS {len(words)}\n#define PIX_NSORT {len(order)}\n#define PIX_EXP_N {EXP_N}\n#define PIX_EXP_BITS {EXP_BITS}\n")
        for k in ("D", "L", "DK", "DV", "HD", "DH", "HH", "NH"): h.write(f"#define PIX_MAX_{k} {mx(k)}\n")
        h.write(f"#define PIX_MAX_NC {max(c.n_cells for c in cfgs)}\n#define PIX_MAX_NS {max(c.n_slots for c in cfgs)}\n")
        h.write("enum { " + ", ".join("MF_" + k for k in MF) + ", MF_N };\nenum { " + ", ".join("LF_" + k for k in LF) + ", LF_N };\n")
        h.write(f"#define PIX_BLOB_BYTES {len(B.b)}u\n/* models: {' '.join(names)} */\n#endif\n")
    print("blob", len(B.b), "bytes =", round(len(B.b) / 2 ** 20, 2), "MiB", names)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if "=" not in x]; o = {k: int(v) for k, v in (x.split("=") for x in sys.argv[1:] if "=" in x)}
    main(a, **o)
