"""Two-speed picture model for the cartridge.

A picture is S x S palette pixels, cut into cells of C x C pixels, written cell by cell in reading order.
  body  one pass per cell: Koko's layer stack (ternary matrices, 1-bit-key top-k attention, running-average
        mixer, routed experts). Its input for cell t is a sum of table rows looked up by the pixels of the
        cell before it and of the cell above it, so no matrix is needed to read pixels.
  head  one tiny pass per pixel: a narrow MLP that sees the body's output for this cell, the pixels already
        written in the cell, and the neighbours left / up / up-left / up-right. It scores 16 colour groups and
        the 16 colours of each group (palette index = group * 16 + member), so only 32 rows are scored per pixel.
Everything a position sees was written earlier; img/tests.py checks that by changing pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.model import Config, QState, QLinear, RMSNorm, Attention, DiagMixer, MoEMLP, DenseMLP


@dataclass
class ImgConfig:
    S: int = 32              # picture side in pixels
    C: int = 4               # cell side in pixels
    P: int = 256             # palette colours = G groups x P/G members
    G: int = 16
    # body (same meaning as train.model.Config)
    D: int = 256
    L: int = 8
    NH: int = 4
    DK: int = 256
    DV: int = 128
    att_layers: list | None = field(default_factory=lambda: [1, 3, 5, 7])
    mlp: str = "moe"
    E: int = 128
    HD: int = 128
    RC: int = 16             # experts are picked by (cell position, colour group of the last pixel written): RC = G
    topk: int = 8
    wq: str = "tern"
    wq_lr_mult: float = 10.0
    row_gain: bool = True
    keyq: str = "sign"
    up_in: bool = True       # the body input also reads the cell above
    n_words: int = 0         # optional prompt: size of the word list (0 = no prompts). A prompt is up to W word ids
    W: int = 8               # (0 = empty slot); the rows of its words are added up, so word order does not matter
    cls_drop: float = 0.1    # share of training pictures shown with an empty prompt
    cls_layers: bool = False # the prompt is also added before every later layer, from one table per layer (ROM only)
    F: int = 0               # > 0: the picture is drawn over a finished thumbnail of side S/F (one palette colour per
                             # F x F block), made first by another model. 0 = no thumbnail
    # head
    DH: int = 64
    HH: int = 128
    head_blocks: int = 1
    # training
    seed: int = 1
    lr: float = 2e-3
    wd: float = 0.05
    table_lr_mult: float = 5.0
    warmup: int = 1000
    steps: int = 20000
    batch: int = 128
    eval_every: int = 2000
    val_images: int = 2048
    ckpt_every: int = 1000
    flip: bool = True
    data: str = "img/data/set"
    init: str = ""           # start from these weights (a state dict, e.g. a night model) instead of from scratch
    name: str = "run"

    @property
    def n_cells(self):
        return (self.S // self.C) ** 2

    @property
    def n_slots(self):
        return self.C * self.C

    def body(self) -> Config:
        return Config(V=self.n_cells * self.RC, D=self.D, L=self.L, T=self.n_cells, NH=self.NH, DK=self.DK, DV=self.DV,
                      mlp=self.mlp, HD=self.HD, E=self.E, router="token", att_layers=self.att_layers, mixer="diag",
                      wq=self.wq, wq_lr_mult=self.wq_lr_mult, row_gain=self.row_gain and self.wq.startswith("tern"),
                      keyq=self.keyq, att_bias=True, topk=self.topk, seed=self.seed)


def layout(cfg: ImgConfig):
    """Index tables. perm[t, j] = raster index of slot j of cell t. nb[t, j, n] = raster index of neighbour n
    (left, up, up-left, up-right) or -1 when it is outside the picture or not written yet."""
    S, C, n = cfg.S, cfg.C, cfg.S // cfg.C
    perm = torch.zeros(n * n, C * C, dtype=torch.long)
    nb = torch.full((n * n, C * C, 4), -1, dtype=torch.long)
    for t in range(n * n):
        cy, cx = divmod(t, n)
        for j in range(C * C):
            sy, sx = divmod(j, C)
            y, x = cy * C + sy, cx * C + sx
            perm[t, j] = y * S + x
            if x > 0: nb[t, j, 0] = y * S + x - 1
            if y > 0: nb[t, j, 1] = (y - 1) * S + x
            if x > 0 and y > 0: nb[t, j, 2] = (y - 1) * S + x - 1
            # up-right lies in the next cell (not written yet) when this is the cell's last column, unless
            # the row above belongs to the cell row above
            if y > 0 and x + 1 < S and (sx < C - 1 or sy == 0): nb[t, j, 3] = (y - 1) * S + x + 1
    return perm, nb


def thumb_layout(cfg: ImgConfig):
    """tw[t, k] = thumbnail index of entry k of the window around cell t (the cell's own blocks plus one ring),
    tp[t, j, n] = thumbnail index of pixel j's own block and the blocks right / below / left / above it. -1 = outside."""
    S, C, Fc, n = cfg.S, cfg.C, cfg.F, cfg.S // cfg.C
    Ts, m = S // Fc, max(1, C // Fc)
    tw = torch.full((n * n, (m + 2) ** 2), -1, dtype=torch.long)
    tp = torch.full((n * n, C * C, 5), -1, dtype=torch.long)
    for t in range(n * n):
        cy, cx = divmod(t, n)
        for k in range((m + 2) ** 2):
            ky, kx = divmod(k, m + 2)
            y, x = cy * C // Fc + ky - 1, cx * C // Fc + kx - 1
            if 0 <= y < Ts and 0 <= x < Ts: tw[t, k] = y * Ts + x
        for j in range(C * C):
            sy, sx = divmod(j, C)
            by, bx = (cy * C + sy) // Fc, (cx * C + sx) // Fc
            for q, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0))):
                y, x = by + dy, bx + dx
                if 0 <= y < Ts and 0 <= x < Ts: tp[t, j, q] = y * Ts + x
    return tw, tp


class ImgModel(nn.Module):
    def __init__(self, cfg: ImgConfig, route_counts=None, pal=None):
        super().__init__()
        assert cfg.P % cfg.G == 0 and cfg.RC == cfg.G and cfg.S % cfg.C == 0
        assert not cfg.F or (cfg.S % cfg.F == 0 and (cfg.C % cfg.F == 0 or cfg.F % cfg.C == 0))
        self.cfg, self.st = cfg, QState()
        b = self.bcfg = cfg.body()
        b.validate()
        torch.manual_seed(cfg.seed)
        nc, ns, P1 = cfg.n_cells, cfg.n_slots, cfg.P + 1                      # index P = "nothing there"
        perm, nb = layout(cfg)
        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("nb", nb, persistent=False)
        # ---- body
        self.cell_pos = nn.Embedding(nc, cfg.D)
        self.prev_emb = nn.Embedding(ns * P1, cfg.D)
        self.up_emb = nn.Embedding(ns * P1, cfg.D) if cfg.up_in else None
        self.cls_emb = nn.Embedding(cfg.n_words, cfg.D, padding_idx=0) if cfg.n_words else None
        for e in (self.cell_pos, self.prev_emb, self.up_emb, self.cls_emb):
            if e is not None: nn.init.normal_(e.weight, std=0.5 / math.sqrt(ns) if e is not self.cell_pos else 0.5)
        if self.cls_emb is not None:
            with torch.no_grad(): self.cls_emb.weight[0].zero_()
        self.cls_l = None
        if cfg.n_words and cfg.cls_layers:
            self.cls_l = nn.ModuleList(nn.Embedding(cfg.n_words, cfg.D, padding_idx=0) for _ in range(cfg.L - 1))
            for e in self.cls_l:
                nn.init.normal_(e.weight, std=0.1)
                with torch.no_grad(): e.weight[0].zero_()
        self.n1 = nn.ModuleList(RMSNorm(cfg.D) for _ in range(cfg.L))
        self.n2 = nn.ModuleList(RMSNorm(cfg.D) for _ in range(cfg.L))
        self.att = nn.ModuleList(Attention(b, self.st) if b.has_att(l) else DiagMixer(b) for l in range(cfg.L))
        self.mlp = nn.ModuleList((MoEMLP(b, self.st, l, route_counts) if cfg.mlp == "moe" else DenseMLP(b, self.st))
                                 for l in range(cfg.L))
        self.nf = RMSNorm(cfg.D)
        self.th_emb = self.hth_emb = None
        if cfg.F:
            tw, tp = thumb_layout(cfg)
            self.register_buffer("tw", tw, persistent=False)
            self.register_buffer("tp", tp, persistent=False)
            self.register_buffer("pal", torch.zeros(cfg.P, 3) if pal is None else torch.as_tensor(pal).float(), persistent=False)
            self.th_emb = nn.Embedding(tw.shape[1] * P1, cfg.D)
            self.hth_emb = nn.Embedding(5 * P1, cfg.DH)
            nn.init.normal_(self.th_emb.weight, std=0.5 / math.sqrt(tw.shape[1]))
            nn.init.normal_(self.hth_emb.weight, std=0.3)
        # ---- head
        self.h_in = QLinear(b, self.st, cfg.D, cfg.DH)
        self.slot_emb = nn.Embedding(ns, cfg.DH)
        self.nb_emb = nn.Embedding(4 * P1, cfg.DH)
        self.in_emb = nn.Embedding(ns * P1, cfg.DH)        # row (slot j', colour): added for every pixel already written in the cell
        for e in (self.slot_emb, self.nb_emb, self.in_emb):
            nn.init.normal_(e.weight, std=0.3)
        self.hn = nn.ModuleList(RMSNorm(cfg.DH) for _ in range(cfg.head_blocks))
        self.h1 = nn.ModuleList(QLinear(b, self.st, cfg.DH, cfg.HH) for _ in range(cfg.head_blocks))
        self.h2 = nn.ModuleList(QLinear(b, self.st, cfg.HH, cfg.DH, zero=True) for _ in range(cfg.head_blocks))
        self.hf = RMSNorm(cfg.DH)
        self.out_g = QLinear(b, self.st, cfg.DH, cfg.G, gain=b.row_gain)
        self.out_m = QLinear(b, self.st, cfg.DH, cfg.P, gain=b.row_gain)

    def set_progress(self, p):
        self.st.on, self.st.lam = True, 0.0

    def table_params(self):
        es = [self.cell_pos, self.prev_emb, self.up_emb, self.cls_emb, self.slot_emb, self.nb_emb, self.in_emb]
        es += [self.th_emb, self.hth_emb] + (list(self.cls_l) if self.cls_l is not None else [])
        return [e.weight for e in es if e is not None]

    def thumb_of(self, img):
        """(B, S*S) palette indices -> (B, (S/F)^2): per F x F block, the palette colour nearest to the block's mean."""
        c = self.cfg
        Ts = c.S // c.F
        rgb = self.pal[img].view(-1, Ts, c.F, Ts, c.F, 3).mean((2, 4)).view(-1, Ts * Ts, 3)
        return torch.cdist(rgb, self.pal[None].expand(len(rgb), -1, -1)).argmin(-1)

    # ------------------------------------------------------------------ pieces
    def cells(self, img):
        """img (B, S*S) palette indices -> (B, cells, slots)."""
        return img[:, self.perm]

    def routes(self, cells, cls=None):
        """Expert route id per cell: cell position * RC + colour group of the last pixel of the cell before.
        The first cell has no pixel before it: it uses the first prompt word instead (0 without a prompt)."""
        c = self.cfg
        g = F.pad(cells[:, :-1, -1], (1, 0)) // (c.P // c.G)
        if cls is not None:
            g = torch.cat([cls[:, :1] % c.RC, g[:, 1:]], 1)
        return torch.arange(c.n_cells, device=cells.device)[None] * c.RC + g

    def body(self, cells, cls=None, thumb=None):
        """Returns h (B, cells, D): h[:, t] depends only on cells < t (and the label)."""
        c, P1 = self.cfg, self.cfg.P + 1
        B, nc, ns = cells.shape
        n = c.S // c.C
        off = torch.arange(ns, device=cells.device) * P1
        none = torch.full_like(cells[:, :1], c.P)
        prev = torch.cat([none, cells[:, :-1]], 1)
        x = self.cell_pos.weight[None, :nc] + self.prev_emb(prev + off).sum(2)
        if self.up_emb is not None:
            up = torch.cat([none.expand(-1, n, -1), cells[:, :-n]], 1)[:, :nc]
            x = x + self.up_emb(up + off).sum(2)
        if self.th_emb is not None:
            tv = F.pad(thumb, (0, 1), value=c.P)[:, self.tw]                   # (B, cells, window)
            x = x + self.th_emb(tv + torch.arange(self.tw.shape[1], device=cells.device) * P1).sum(2)
        if self.cls_emb is not None:
            if cls is None: cls = torch.zeros(B, c.W, dtype=torch.long, device=cells.device)      # empty prompt
            x = x + self.cls_emb(cls).sum(1)[:, None]
        else:
            cls = None
        route = self.routes(cells, cls)
        words = cls if self.cls_l is not None else None
        ents = []
        for l in range(c.L):
            if words is not None and l: x = x + self.cls_l[l - 1](words).sum(1)[:, None]
            if self.bcfg.has_att(l):
                x = x + self.att[l](self.n1[l](x))[0]
            else:
                x = x + self.att[l](self.n1[l](x))
            y, _, e = self.mlp[l](self.n2[l](x), route, None)
            x = x + y
            if e is not None: ents.append(e)
        return self.nf(x), (torch.stack(ents).mean() if ents else None)

    def head_feats(self, img, cells, t=None, thumb=None):
        """Table-row part of the head input for every pixel: (B, cells, slots, DH); for one cell t: (B, slots, DH)."""
        c, P1 = self.cfg, self.cfg.P + 1
        ns = c.n_slots
        imgp = F.pad(img, (0, 1), value=c.P)                                  # index -1 reads "nothing there"
        nb = self.nb if t is None else self.nb[t]
        cells = cells if t is None else cells[:, t]
        nbv = imgp[:, nb]                                                     # (..., slots, 4)
        f = self.slot_emb.weight + self.nb_emb(nbv + torch.arange(4, device=img.device) * P1).sum(-2)
        if self.hth_emb is not None:
            tv = F.pad(thumb, (0, 1), value=c.P)[:, self.tp if t is None else self.tp[t]]
            f = f + self.hth_emb(tv + torch.arange(5, device=img.device) * P1).sum(-2)
        e = self.in_emb(cells + torch.arange(ns, device=img.device) * P1)     # (..., slots, DH)
        return f + (e.cumsum(-2) - e)                                         # pixels of this cell before slot j

    def head_scores(self, u):
        """u (..., DH) -> (group scores (..., G), member scores (..., G, P/G)), before any softmax."""
        c = self.cfg
        for n, w1, w2 in zip(self.hn, self.h1, self.h2):
            u = u + w2(F.relu(w1(n(u))).square())
        u = self.hf(u)
        return self.out_g(u), self.out_m(u).view(*u.shape[:-1], c.G, c.P // c.G)

    def head(self, u):
        """u (..., DH) -> log-probabilities over the palette (..., P)."""
        g, m = self.head_scores(u)
        return (g.log_softmax(-1)[..., None] + m.log_softmax(-1)).flatten(-2)

    def forward(self, img, cls=None, thumb=None):
        """img (B, S*S) long. Returns (log-probs (B, cells, slots, P), targets (B, cells, slots), expert entropy)."""
        cells = self.cells(img)
        if self.cfg.F and thumb is None: thumb = self.thumb_of(img)
        h, ent = self.body(cells, cls, thumb)
        u = self.h_in(h)[:, :, None] + self.head_feats(img, cells, None, thumb)
        return self.head(u), cells, ent

    @torch.no_grad()
    def sample(self, n, device, cls=None, temp=1.0, gen=None, thumb=None, guide=1.0):
        """Write n pictures cell by cell, pixel by pixel, the way the cartridge will: pick a colour group, then
        score only that group's colours. Temperature applies to each of the two picks. Returns (n, S*S)."""
        c = self.cfg
        img = torch.zeros(n, c.S * c.S, dtype=torch.long, device=device)
        assert (thumb is None) == (not c.F) and (thumb is None or thumb.shape[1] == (c.S // c.F) ** 2), "thumbnail side must be S/F"
        ar = torch.arange(n, device=device)
        for t in range(c.n_cells):
            h = self.h_in(self.body(self.cells(img), cls, thumb)[0][:, t])
            twice = guide != 1.0 and cls is not None       # prompt guidance: a second body pass with the empty prompt
            if twice: h0 = self.h_in(self.body(self.cells(img), torch.zeros_like(cls), thumb)[0][:, t])
            for j in range(c.n_slots):
                f = self.head_feats(img, self.cells(img), t, thumb)[:, j]
                g, m = self.head_scores(h + f)
                if twice:
                    g0, m0 = self.head_scores(h0 + f)
                    g = g0.log_softmax(-1) + guide * (g.log_softmax(-1) - g0.log_softmax(-1))
                    m = m0.log_softmax(-1) + guide * (m.log_softmax(-1) - m0.log_softmax(-1))
                gi = torch.multinomial((g / temp).softmax(-1), 1, generator=gen)[:, 0]
                mi = torch.multinomial((m[ar, gi] / temp).softmax(-1), 1, generator=gen)[:, 0]
                img[:, self.perm[t, j]] = gi * (c.P // c.G) + mi
        return img

    def counts(self):
        tables = sum(p.numel() for p in self.table_params())
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "tables": tables, "matrices": total - tables}


def gba_cost(cfg: ImgConfig):
    """Estimated cartridge cost per picture, from Koko's measured costs (engine/DESIGN.md in the Koko repository): a ternary row costs
    6.4 cycles per stored byte (4 weights) plus about 55 of overhead; building the lookup tables for a vector
    costs about 270 cycles per 4 inputs. Koko's body (D=256, 8 layers, half with attention): 2.80M cycles that
    scale with D^2 and 0.98M that scale with D. Earlier estimates of this kind ran 35% optimistic."""
    row = lambda n_in: n_in / 4 * 6.4 + 55
    build = lambda n_in: n_in / 4 * 270
    w, d = cfg.D / 256, cfg.L / 8
    hd = cfg.HD / 128
    body = (2.80e6 * w * (0.6 * w + 0.4 * hd) + 0.98e6 * w) * d + cfg.n_slots * 2 * cfg.D * 6      # + input row sums
    head_cell = cfg.DH * row(cfg.D) + build(cfg.D)
    pix = 0.0
    for _ in range(cfg.head_blocks):
        pix += cfg.HH * row(cfg.DH) + build(cfg.DH) + cfg.DH * row(cfg.HH) + build(cfg.HH) + cfg.DH * 40
    pix += (cfg.G + cfg.P // cfg.G) * row(cfg.DH) + build(cfg.DH) + cfg.DH * 40 + (4 + cfg.n_slots / 2) * cfg.DH * 6 + 2000
    total = cfg.n_cells * (body + head_cell) + cfg.S * cfg.S * pix
    mats = cfg.L * (cfg.E if cfg.mlp == "moe" else 1) * 2 * cfg.D * cfg.HD + (cfg.L // 2 if cfg.att_layers else cfg.L) * (
        2 * cfg.D * cfg.DK + 2 * cfg.D * cfg.DV) + cfg.D * cfg.DH + cfg.head_blocks * 2 * cfg.DH * cfg.HH + cfg.DH * (cfg.G + cfg.P)
    rows = (cfg.n_slots * (cfg.P + 1) * (2 if cfg.up_in else 1) + cfg.n_cells + cfg.n_words) * cfg.D + (cfg.n_slots + 4) * (cfg.P + 1) * cfg.DH
    return {"cycles_per_cell_body": round(body), "cycles_per_pixel_head": round(pix), "cycles_per_picture": round(total),
            "seconds_per_picture": round(total / 16.78e6, 1), "seconds_if_35pct_optimistic": round(total * 1.35 / 16.78e6, 1),
            "rom_mb": round((mats / 4 + rows * 0.75) / 2 ** 20, 2)}


def config_dict(cfg):
    return asdict(cfg)
