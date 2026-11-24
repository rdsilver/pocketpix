"""First-step slots for the two-step output head.

The K most frequent tokens get their own slot (0..K-1). The rest are split into C equal-size
clusters (slots K..K+C-1): at random, by frequency band, or by k-means on which tokens they
appear next to in the training text (tokens used in similar places share a cluster)."""
from __future__ import annotations

import math

import numpy as np
import torch


def cooccurrence_features(stream, V, n_tokens=20_000_000, dim=48, seed=0):
    """(V, 2*dim) vectors from next-token and previous-token counts, reduced by a low-rank SVD."""
    s = np.asarray(stream[:n_tokens]).astype(np.int64)
    counts = np.bincount(s[:-1] * V + s[1:], minlength=V * V).reshape(V, V).astype(np.float32)
    feats = []
    torch.manual_seed(seed)
    for m in (counts, counts.T):                         # what follows, what precedes
        m = torch.from_numpy(np.log1p(m))
        m = m / m.norm(dim=1, keepdim=True).clamp_min(1e-6)
        u, sv, _ = torch.svd_lowrank(m, q=dim, niter=4)
        feats.append(u * sv)
    f = torch.cat(feats, 1)
    return f / f.norm(dim=1, keepdim=True).clamp_min(1e-6)


def _balanced_kmeans(x, C, cap, seed, iters=25):
    g = torch.Generator().manual_seed(seed)
    cent = x[torch.randperm(len(x), generator=g)[:C]].clone()
    for _ in range(iters):
        lab = torch.cdist(x, cent).argmin(1)
        for c in range(C):
            m = lab == c
            if m.any():
                cent[c] = x[m].mean(0)
    d = torch.cdist(x, cent)
    # points with the clearest preference choose first; each takes its nearest cluster with room
    best2 = d.topk(2, largest=False).values
    order = (best2[:, 1] - best2[:, 0]).argsort(descending=True)
    room = [cap] * C
    lab = torch.empty(len(x), dtype=torch.long)
    for i in order.tolist():
        for c in d[i].argsort().tolist():
            if room[c]:
                lab[i] = c
                room[c] -= 1
                break
    return lab


def build_assign(counts, feats, K, C, mode, seed):
    """Returns a LongTensor (V,) of first-step slots."""
    counts = torch.as_tensor(counts).double()
    V = len(counts)
    g = torch.Generator().manual_seed(seed)
    order = (counts + 1e-6 * torch.rand(V, generator=g, dtype=torch.float64)).argsort(descending=True)
    tail = order[K:]
    n = len(tail)
    C = C or max(1, round(math.sqrt(n)))
    cap = math.ceil(n / C)
    if mode == "freq":
        lab = torch.arange(n) // cap
    elif mode == "random":
        lab = (torch.randperm(n, generator=g) // cap)
    elif mode == "kmeans":
        assert feats is not None, "k-means clusters need co-occurrence features"
        lab = _balanced_kmeans(feats[tail].float(), C, cap, seed)
    else:
        raise ValueError(mode)
    assign = torch.empty(V, dtype=torch.long)
    assign[order[:K]] = torch.arange(K)
    assign[tail] = K + lab
    return assign
