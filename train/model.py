"""Koko model, config-driven so one file covers every arm of the experiment ladders.

Decoder-only, pre-norm. Options: weight format, cheap attention keys, top-k attention, plain or
hard-routed MLP blocks with several router types, lookup tables keyed on recent tokens, a hash-keyed
memory, a copy head, a cheap mixer for layers without attention, and a two-step output head.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import quant

KEYQ = ("none", "sign", "tern34", "pq16", "rescore")
ROUTERS = ("learned", "frozen", "token", "keybits")


@dataclass
class Config:
    # shape
    V: int = 4096
    D: int = 128
    L: int = 4
    T: int = 256
    NH: int = 4
    DK: int = 64            # total key/query width
    DV: int = 64            # total value width
    mlp: str = "dense"      # dense | moe
    HD: int = 256           # hidden width of the MLP (of each expert for moe)
    E: int = 16             # experts (moe)
    router: str = "learned"  # learned | frozen (random, untrained) | token (by token id) | keybits (attention key signs)
    moe_aux: float = 0.01   # load-balancing loss weight (learned router only)
    tie_head: bool = False
    head: str = "flat"      # flat | two_level
    head_clusters: str = "kmeans"   # random | freq | kmeans  (two_level)
    head_short: int = 0     # most frequent tokens scored directly at the first step
    head_C: int = 0         # clusters (0 = about sqrt of the tail)
    att_layers: list | None = None  # layers that have attention (None = all)
    mixer: str = "none"     # what layers without attention get: none | diag
    x0_mix: bool = False    # learned per-layer scalars re-adding the input embedding
    copy_head: bool = False  # last layer's first head votes for the token that followed each attended position
    hashmem_heads: int = 0  # memory rows addressed by sign bits of fixed random projections of the stream
    hashmem_bits: int = 14
    hashmem_layers: list = field(default_factory=list)
    # constraints
    wq: str = "fp"          # fp | tern | tern_median | sherry | int8
    arenas: bool = False    # full-precision residual, faded to zero during training
    arenas_frac: float = 0.6
    int8_parts: tuple = ()   # diagnostic: parts kept 8-bit while the rest follows wq: "head", "att", "mlp"
    wq_start_frac: float = 0.0
    wq_lr_mult: float = 1.0
    row_gain: bool = False  # learned per-row gain after matrices that feed hard decisions (keys, router, head)
    aq: int = 0             # activation bits in front of quantised matrices (0 = off)
    keyq: str = "none"      # none | sign | tern34 | pq16 (learned 16-entry codebook per 4 key dims) | rescore
    rescore_m: int = 32     # rescore: sign keys pick this many candidates, exact keys rank them
    att_bias: bool = False  # learned per-head bias by distance
    topk: int = 0           # attention mixes only the k best positions (0 = all)
    inject: list = field(default_factory=list)   # [{kind: tok|bigram|trigram, rows, layers, gate, bits}]
    # training
    seed: int = 1
    opt: str = "adamw"      # adamw | muon (matrices only; everything else stays on Adam)
    muon_lr: float = 0.02
    lr: float = 2e-3
    wd: float = 0.05
    wd_off_frac: float = 1.0  # weight decay is switched off after this fraction of training
    table_lr_mult: float = 1.0
    warmup: int = 200
    steps: int = 2000
    batch: int = 32
    eval_every: int = 500
    val_seqs: int = 256
    data: str = "tinystories"   # tinystories | synthetic | mix0 (chat / facts / stories, see corpus.py)
    init: str = ""          # start from this model.pt instead of random weights (fine-tuning)
    teacher: str = ""       # run directory of a full-precision model that coaches this one (result.json + model.pt)
    kd_w: float = 0.0       # weight of the coaching loss: (1 - kd_w) * cross-entropy + kd_w * T^2 * KL(teacher || student)
    kd_t: float = 1.0       # coaching temperature T
    ckpt_every: int = 0     # save a resumable checkpoint every N steps (0 = never); a rerun resumes from it
    mix: dict = field(default_factory=lambda: {"chat": 0.5, "facts": 0.25, "stories": 0.25})  # training shares
    facts_n: int = 0        # fact-capacity probe: number of made-up people (0 = off)
    facts_frac: float = 0.25  # share of each batch that is fact text
    facts_marker: bool = False
    name: str = "run"

    def validate(self):
        assert self.wq in ("fp", "int8") + quant.WQ_MODES and self.keyq in KEYQ and self.router in ROUTERS
        assert self.DK % self.NH == 0 and self.DV % self.NH == 0 and (self.DK // self.NH) % 2 == 0
        assert self.mlp in ("dense", "moe") and self.head in ("flat", "two_level") and self.mixer in ("none", "diag")
        assert self.opt in ("adamw", "muon")
        if self.wq == "sherry" or self.keyq in ("tern34", "pq16"):
            assert self.D % 4 == 0 and self.HD % 4 == 0 and self.DV % 4 == 0 and (self.DK // self.NH) % 4 == 0
        for t in self.inject:
            assert t["kind"] in ("tok", "bigram", "trigram") and all(0 <= l < self.L for l in t["layers"])
        if self.copy_head:
            assert self.head == "flat", "copy head is only wired to the flat output head"
            assert self.has_att(self.L - 1), "copy head needs attention in the last layer"
        if self.mlp == "moe" and self.router == "keybits":
            assert self.att_layers is None, "keybits routing needs attention in every layer"
            assert 2 ** (self.E.bit_length() - 1) == self.E and self.E.bit_length() - 1 <= self.DK
        assert not (self.tie_head and self.head == "two_level")
        # with a per-row weight scale (or float weights) a per-row gain is the same function, so the arm would test nothing
        assert not (self.row_gain and self.wq in ("fp", "int8", "sherry")), "row_gain only adds anything to tern / tern_median"

    def has_att(self, l):
        return self.att_layers is None or l in self.att_layers


class QState:
    """Shared switchboard: is quantisation on yet, and how much full-precision residual is left."""
    def __init__(self):
        self.on = True
        self.lam = 0.0


class QLinear(nn.Module):
    """Bias-free linear layer whose weight is quantised per the config.

    Weight shape (..., out, in). With a 3-D weight and a 3-D input it runs one matrix per leading
    index (batched experts). `gain` adds a learned per-row gain after the matrix."""

    def __init__(self, cfg: Config, st: QState, n_in: int, n_out: int, lead: tuple = (), zero: bool = False,
                 gain: bool = False, part: str = ""):
        super().__init__()
        self.cfg, self.st = cfg, st
        self.int8 = cfg.wq == "int8" or (cfg.wq != "fp" and part in cfg.int8_parts)
        w = torch.zeros(*lead, n_out, n_in) if zero else torch.randn(*lead, n_out, n_in) / math.sqrt(n_in)
        self.weight = nn.Parameter(w)
        self.g = nn.Parameter(torch.ones(*lead, n_out)) if gain else None

    def qweight(self, w):
        if self.int8:                                  # per-row 8-bit, as in the stage-1 cartridge
            return quant.quantize_rows(w, 8)
        return quant.quantize_weight(w, self.cfg.wq)

    @staticmethod
    def _lin(x, w):
        return torch.bmm(x, w.transpose(1, 2)) if w.ndim == 3 else F.linear(x, w)

    def forward(self, x, index=None):
        w = self.weight if index is None else self.weight[index]
        if self.cfg.wq == "fp" or not self.st.on:
            y = self._lin(x, w)
        else:
            xq = quant.quantize_act(x, self.cfg.aq) if self.cfg.aq else x
            y = self._lin(xq, self.qweight(w))
            if self.cfg.arenas and self.training and self.st.lam > 0:
                y = y + self.st.lam * self._lin(x, w)
        if self.g is not None:
            g = self.g if index is None else self.g[index]
            y = y * (g[:, None, :] if w.ndim == 3 else g)
        return y


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)


def rope(x, pos):
    """x: (B, H, T, d) with d even. Rotate pairs by position, llama-style frequencies."""
    d = x.shape[-1]
    freq = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device) / d))
    ang = pos[:, None] * freq[None, :]
    c, s = ang.cos()[None, None], ang.sin()[None, None]
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2)


def _unit(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)


class Attention(nn.Module):
    def __init__(self, cfg, st):
        super().__init__()
        self.cfg = cfg
        self.q = QLinear(cfg, st, cfg.D, cfg.DK, part="att")
        self.k = QLinear(cfg, st, cfg.D, cfg.DK, gain=cfg.row_gain, part="att")
        self.v = QLinear(cfg, st, cfg.D, cfg.DV, part="att")
        self.o = QLinear(cfg, st, cfg.DV, cfg.D, zero=True, part="att")
        # quantised keys have a fixed length, so each head learns its own sharpness
        self.temp = nn.Parameter(torch.zeros(cfg.NH))
        if cfg.att_bias:
            self.dbias = nn.Parameter(torch.zeros(cfg.NH, cfg.T))
        if cfg.keyq == "pq16":
            hk = cfg.DK // cfg.NH
            pat = torch.tensor([[1.0 if (p >> b) & 1 else -1.0 for b in range(4)] for p in range(16)])
            self.codebook = nn.Parameter(pat[None, None].repeat(cfg.NH, hk // 4, 1, 1))   # (H, G, 16, 4)

    def _keys(self, q, k):
        """Returns (query, key) as scored. Quantised keys have fixed length, so the query is bounded
        too (per-head RMS) and the key gradient is passed only where the key is not saturated."""
        c = self.cfg
        if c.keyq in ("none", "rescore"):
            return q, k
        q = _unit(q)
        kc = (k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True).detach() + 1e-5)).clamp(-1, 1)
        if c.keyq == "pq16":
            B, H, T, hk = k.shape
            bits = (k.detach() >= 0).long().view(B, H, T, hk // 4, 4)
            idx = (bits * torch.tensor([1, 2, 4, 8], device=k.device)).sum(-1)              # (B, H, T, G)
            h_i = torch.arange(H, device=k.device)[None, :, None, None]
            g_i = torch.arange(hk // 4, device=k.device)[None, None, None, :]
            khat = self.codebook[h_i, g_i, idx].reshape(B, H, T, hk)                         # learned codewords
            return q, khat + (kc - kc.detach())
        return q, kc + (quant.key_values(k, c.keyq) - kc).detach()

    def forward(self, x):
        c = self.cfg
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        split = lambda t, d: t.view(B, T, c.NH, d // c.NH).transpose(1, 2)
        kraw = self.k(x)                                   # before rotation: position-free, used by keybits routing
        q = rope(split(self.q(x), c.DK), pos)
        k = rope(split(kraw, c.DK), pos)
        v = split(self.v(x), c.DV)
        scale = self.temp.clamp(-3, 3).exp()[None, :, None, None] / math.sqrt(c.DK // c.NH)
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        qs, ks = self._keys(q, k)
        s = (qs @ ks.transpose(-1, -2)) * scale
        if c.att_bias:
            i = torch.arange(T, device=x.device)
            s = s + self.dbias[:, (i[:, None] - i[None, :]).clamp(min=0)][None]
        s = s.masked_fill(~causal, float("-inf"))
        if c.keyq == "rescore" and T > c.rescore_m:        # cheap sign scores shortlist, exact scores rank
            with torch.no_grad():
                ss = (_unit(q) @ quant.hard_sign(k).transpose(-1, -2)).masked_fill(~causal, float("-inf"))
                cand = ss >= ss.topk(c.rescore_m, dim=-1).values[..., -1:]
            s = s.masked_fill(~cand, float("-inf"))
        if c.topk and T > c.topk:
            kth = s.topk(c.topk, dim=-1).values[..., -1:]
            s = s.masked_fill(s < kth, float("-inf"))
        a = s.softmax(-1)
        return self.o((a @ v).transpose(1, 2).reshape(B, T, c.DV)), a, kraw


class DiagMixer(nn.Module):
    """Cheap stand-in for attention: per-dimension weights on the previous three positions and on
    three running averages. No matrix, so it costs a few thousand cycles on the cartridge."""

    def __init__(self, cfg):
        super().__init__()
        self.taps = nn.Parameter(torch.zeros(3, cfg.D))
        self.avg = nn.Parameter(torch.zeros(3, cfg.D))
        t = torch.arange(cfg.T)
        d = (t[:, None] - t[None, :]).clamp(min=0).float()
        ks = [((1 - r) * r ** d).tril() for r in (0.5, 0.8, 0.95)]
        self.register_buffer("kern", torch.stack(ks), persistent=False)      # (3, T, T)

    def forward(self, x):
        T = x.shape[1]
        y = torch.zeros_like(x)
        for i in range(3):
            y = y + self.taps[i] * F.pad(x, (0, 0, i + 1, 0))[:, :T]
            y = y + self.avg[i] * (self.kern[i, :T, :T] @ x)
        return y


class DenseMLP(nn.Module):
    def __init__(self, cfg, st):
        super().__init__()
        self.w1 = QLinear(cfg, st, cfg.D, cfg.HD, part="mlp")
        self.w2 = QLinear(cfg, st, cfg.HD, cfg.D, zero=True, part="mlp")

    def forward(self, x, tok=None, kraw=None):
        return self.w2(F.relu(self.w1(x)).square()), x.new_zeros(()), None


def balanced_token_routes(counts, E, seed):
    """Token -> expert, spreading training-frequency mass evenly: heaviest tokens first, each to a
    random one of the experts that are still (nearly) the lightest. The random choice makes the map
    of each layer independent of the others while keeping the loads even."""
    g = torch.Generator().manual_seed(seed)
    counts = torch.as_tensor(counts).double()
    load = torch.zeros(E, dtype=torch.float64)
    route = torch.zeros(len(counts), dtype=torch.long)
    for t in counts.argsort(descending=True, stable=True).tolist():
        cand = (load <= load.min() + counts[t]).nonzero().flatten()
        e = int(cand[torch.randint(len(cand), (1,), generator=g)])
        route[t] = e
        load[e] += counts[t]
    return route


class MoEMLP(nn.Module):
    """One expert runs per token. Routers: learned (argmax of a trained matrix, gradient through the
    chosen expert's probability plus a load-balancing term), frozen (same matrix, never trained),
    token (a table indexed by the token id), keybits (sign bits of this layer's attention key)."""

    def __init__(self, cfg, st, layer, counts=None):
        super().__init__()
        self.cfg = cfg
        if cfg.router in ("learned", "frozen"):
            self.router = QLinear(cfg, st, cfg.D, cfg.E, gain=cfg.row_gain and cfg.router == "learned")
            if cfg.router == "frozen":
                self.router.weight.requires_grad_(False)
        if cfg.router == "token":
            counts = torch.ones(cfg.V) if counts is None else torch.as_tensor(counts)
            self.register_buffer("route", balanced_token_routes(counts, cfg.E, cfg.seed * 1000 + layer))
        self.w1 = QLinear(cfg, st, cfg.D, cfg.HD, lead=(cfg.E,), part="mlp")
        self.w2 = QLinear(cfg, st, cfg.HD, cfg.D, lead=(cfg.E,), zero=True, part="mlp")

    def _run(self, x, idx):
        """x (N, D), idx (N,): apply each token's expert. Tokens are grouped by expert and padded
        so all experts run in one batched product."""
        E, N = self.cfg.E, x.shape[0]
        counts = torch.bincount(idx, minlength=E)
        M = int(counts.max())
        if M * E > 32 * N + 64 * E:                     # very lopsided routing: avoid a huge padded tensor
            y = torch.zeros_like(x)
            for e in range(E):
                m = idx == e
                y[m] = self.w2(F.relu(self.w1(x[m], e)).square(), e).to(y.dtype)
            return y
        order = idx.argsort(stable=True)
        e_sorted = idx[order]
        pos = torch.arange(N, device=x.device) - (counts.cumsum(0) - counts)[e_sorted]
        pad = x.new_zeros(E, M, x.shape[1])
        pad[e_sorted, pos] = x[order]
        out = self.w2(F.relu(self.w1(pad)).square())
        y = torch.empty_like(x)
        y[order] = out[e_sorted, pos].to(y.dtype)
        return y

    def forward(self, x, tok=None, kraw=None):
        c = self.cfg
        E = c.E
        aux = x.new_zeros(())
        gate = 1.0
        if c.router == "token":
            idx = self.route[tok]
        elif c.router == "keybits":
            nb = E.bit_length() - 1
            bits = (kraw[..., :nb].detach() >= 0).long()
            idx = (bits * (2 ** torch.arange(nb, device=x.device))).sum(-1)
        else:
            p = self.router(x).softmax(-1)
            idx = p.argmax(-1)
            if c.router == "learned":
                pc = p.gather(-1, idx[..., None])
                gate = 1.0 + pc - pc.detach()
                mean_p = p.flatten(0, 1).mean(0)
                aux = (mean_p * (mean_p * E + 1e-9).log()).sum()        # KL(mean usage || uniform)
        y = self._run(x.flatten(0, 1), idx.flatten()).view_as(x)
        use = torch.bincount(idx.flatten(), minlength=E).float()
        use = use / use.sum()
        ent = -(use * (use + 1e-9).log()).sum() / math.log(E)
        return y * gate, aux, ent.detach()


def _mix32(h):
    """Avalanche a 64-bit integer tensor down to well-mixed 32 bits (multiply-xor-shift)."""
    m = 0xFFFFFFFF
    h = h & m
    h = ((h ^ (h >> 15)) * 0x2C1B3C6D) & m
    h = ((h ^ (h >> 12)) * 0x297A2D39) & m
    return h ^ (h >> 15)


class Inject(nn.Module):
    """Rows looked up by the current token or a hash of the last 2-3 tokens, added to the stream."""

    def __init__(self, cfg, spec, salt):
        super().__init__()
        self.kind, self.rows, self.gate = spec["kind"], spec["rows"], spec.get("gate", False)
        self.bits, self.salt = spec.get("bits", 0), salt
        self.v = nn.Embedding(self.rows, cfg.D)
        nn.init.zeros_(self.v.weight)
        if self.gate:
            self.k = nn.Embedding(self.rows, cfg.D)
            nn.init.normal_(self.k.weight, std=0.02)
        self.scale = 1.0 / math.sqrt(cfg.D)

    def index(self, tok):
        if self.kind == "tok":
            return tok % self.rows
        h = tok * 0x9E3779B1 + F.pad(tok, (1, 0))[:, :-1] * 0x85EBCA77 + self.salt * 0x27D4EB2F
        if self.kind == "trigram":
            h = h + F.pad(tok, (2, 0))[:, :-2] * 0xC2B2AE3D
        return _mix32(h) % self.rows

    def forward(self, tok, xn):
        i = self.index(tok)
        v = self.v(i)
        if self.bits:
            v = quant.quantize_rows(v, self.bits)
        if self.gate:
            v = v * torch.sigmoid((xn * self.k(i)).sum(-1, keepdim=True) * self.scale)
        return v


class HashMem(nn.Module):
    """Content-addressed memory with no key scoring: each head takes the sign bits of a few fixed
    random +-1 projections of the normalised stream as a row address. Similar states share rows."""

    def __init__(self, cfg, layer):
        super().__init__()
        self.H, self.nb = cfg.hashmem_heads, cfg.hashmem_bits
        g = torch.Generator().manual_seed(cfg.seed * 7919 + layer)
        proj = torch.randint(0, 2, (self.H * self.nb, cfg.D), generator=g).float() * 2 - 1
        self.register_buffer("proj", proj)
        self.table = nn.Embedding(self.H * 2 ** self.nb, cfg.D)
        nn.init.zeros_(self.table.weight)

    def forward(self, xn):
        b = (F.linear(xn.detach(), self.proj) >= 0).long().view(*xn.shape[:-1], self.H, self.nb)
        idx = (b * (2 ** torch.arange(self.nb, device=xn.device))).sum(-1)
        idx = idx + torch.arange(self.H, device=xn.device) * 2 ** self.nb
        return self.table(idx).sum(-2)


class TwoLevelHead(nn.Module):
    """P(token) = P(slot) for short-listed tokens, else P(cluster) * P(token | cluster).

    `assign[t]` is the token's first-step slot: 0..K-1 for the short-list, K + cluster otherwise.
    forward() returns exact log-probabilities over the whole vocabulary, so loss and validation are
    directly comparable with the flat head."""

    def __init__(self, cfg, st, assign):
        super().__init__()
        K = cfg.head_short
        n_root = int(assign.max()) + 1
        self.C = n_root - K
        self.root = QLinear(cfg, st, cfg.D, n_root, gain=cfg.row_gain, part="head")
        self.mem = QLinear(cfg, st, cfg.D, cfg.V, gain=cfg.row_gain, part="head")
        self.register_buffer("slot", assign.clone())
        self.register_buffer("tail", assign >= K)
        self.register_buffer("cl", torch.where(assign >= K, assign - K, torch.full_like(assign, self.C)))

    def forward(self, x):
        shp = x.shape[:-1]
        x = x.flatten(0, -2)
        N = x.shape[0]
        r = self.root(x).log_softmax(-1)                                   # (N, K + C)
        z = self.mem(x)                                                    # (N, V); short-list rows unused
        cl = self.cl[None].expand(N, -1)
        zmax = z.new_full((N, self.C + 1), -1e30).scatter_reduce(1, cl, z, "amax")
        s = z.new_zeros(N, self.C + 1).scatter_add(1, cl, (z - zmax.gather(1, cl)).exp())
        lse = s.clamp_min(1e-30).log() + zmax
        first = r.gather(1, self.slot[None].expand(N, -1))
        lp = torch.where(self.tail[None], first + z - lse.gather(1, cl), first)
        return lp.view(*shp, -1)


class Model(nn.Module):
    def __init__(self, cfg: Config, aux: dict | None = None):
        """aux: optional {"counts": token counts (V,), "assign": first-step slots (V,) for the two-step head}."""
        super().__init__()
        cfg.validate()
        aux = aux or {}
        self.cfg, self.st = cfg, QState()
        torch.manual_seed(cfg.seed)
        self.emb = nn.Embedding(cfg.V, cfg.D)
        nn.init.normal_(self.emb.weight, std=0.5)
        self.n1 = nn.ModuleList(RMSNorm(cfg.D) for _ in range(cfg.L))
        self.n2 = nn.ModuleList(RMSNorm(cfg.D) for _ in range(cfg.L))
        self.att = nn.ModuleList()
        for l in range(cfg.L):
            self.att.append(Attention(cfg, self.st) if cfg.has_att(l) else
                            (DiagMixer(cfg) if cfg.mixer == "diag" else nn.Identity()))
        self.mlp = nn.ModuleList((MoEMLP(cfg, self.st, l, aux.get("counts")) if cfg.mlp == "moe"
                                  else DenseMLP(cfg, self.st)) for l in range(cfg.L))
        self.inj = nn.ModuleDict()
        for n, spec in enumerate(cfg.inject):
            for l in spec["layers"]:
                self.inj[f"{l}_{n}"] = Inject(cfg, spec, salt=n * 16 + l + 1)
        self.hm = nn.ModuleDict({str(l): HashMem(cfg, l) for l in cfg.hashmem_layers}) if cfg.hashmem_heads else None
        if cfg.x0_mix:
            self.lam_res = nn.Parameter(torch.ones(cfg.L))
            self.lam_x0 = nn.Parameter(torch.zeros(cfg.L))
        if cfg.copy_head:
            self.copy_gain = nn.Parameter(torch.zeros(()))
        self.nf = RMSNorm(cfg.D)
        if cfg.head == "two_level":
            assign = aux.get("assign")
            if assign is None:                           # tests: a seeded random balanced assignment
                from .clusters import build_assign
                assign = build_assign(torch.ones(cfg.V), None, cfg.head_short, cfg.head_C, "random", cfg.seed)
            self.head = TwoLevelHead(cfg, self.st, torch.as_tensor(assign))
        else:
            self.head = None if cfg.tie_head else QLinear(cfg, self.st, cfg.D, cfg.V, gain=cfg.row_gain, part="head")

    def set_progress(self, p: float):
        """p = fraction of training done. Switches quantisation on and fades the residual."""
        c = self.cfg
        self.st.on = p >= c.wq_start_frac
        if c.arenas and self.st.on:
            r = min(1.0, (p - c.wq_start_frac) / max(c.arenas_frac, 1e-9))
            self.st.lam = 0.5 * (1 + math.cos(math.pi * r)) if r < 1.0 else 0.0
        else:
            self.st.lam = 0.0

    def table_params(self):
        """Lookup rows: trained with their own learning rate and no weight decay."""
        ps = list(self.emb.parameters()) + [p for m in self.inj.values() for p in m.parameters()]
        if self.hm is not None:
            ps += [m.table.weight for m in self.hm.values()]
        return ps

    def forward(self, tok):
        """Returns (scores over the vocabulary, auxiliary loss, expert-usage entropy or None).
        Scores are logits for the flat head and exact log-probabilities for the two-step head."""
        c = self.cfg
        x0 = self.emb(tok)
        x = x0
        aux, ents, a_last = x.new_zeros(()), [], None
        for l in range(c.L):
            if c.x0_mix:
                x = self.lam_res[l] * x + self.lam_x0[l] * x0
            if self.inj:
                xn = self.n1[l](x)
                for key, m in self.inj.items():
                    if key.startswith(f"{l}_"):
                        x = x + m(tok, xn)
            kraw = None
            if c.has_att(l):
                y, a_last, kraw = self.att[l](self.n1[l](x))
                x = x + y
            elif c.mixer == "diag":
                x = x + self.att[l](self.n1[l](x))
            xn2 = self.n2[l](x)
            y, a, e = self.mlp[l](xn2, tok, kraw)
            if self.hm is not None and str(l) in self.hm:
                y = y + self.hm[str(l)](xn2)
            x, aux = x + y, aux + a
            if e is not None:
                ents.append(e)
        x = self.nf(x)
        if c.head == "two_level":
            out = self.head(x)
        else:
            out = F.linear(x, self.emb.weight) if self.head is None else self.head(x)
            if c.copy_head:
                # position i attends j < i with weight w; that is a vote for the token seen at j + 1
                B, T = tok.shape
                w = a_last[:, 0].tril(-1)[:, :, :T - 1]
                votes = out.new_zeros(B, T, c.V).scatter_add_(2, tok[:, None, 1:].expand(B, T, T - 1), w.to(out.dtype))
                out = out + self.copy_gain * votes
        return out, aux, (torch.stack(ents).mean() if ents else None)

    def counts(self):
        tables = sum(p.numel() for p in self.table_params())
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "tables": tables, "matrices": total - tables}


def config_dict(cfg):
    return asdict(cfg)
