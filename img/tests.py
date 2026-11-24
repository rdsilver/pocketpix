"""Checks: nothing sees a pixel before it is written; log-probabilities sum to one; sampling matches the scores.
    python3 -m img.tests"""
import torch
from .model import ImgConfig, ImgModel, gba_cost

def tiny(**k):
    return ImgConfig(D=32, L=4, DK=32, DV=16, E=4, HD=16, DH=16, HH=32, att_layers=[1, 3], **k)

def make(cfg):
    torch.manual_seed(5)
    return ImgModel(cfg, pal=torch.rand(cfg.P, 3)).eval()

def randomise(m):
    torch.manual_seed(0)
    for p in m.parameters():
        if p.abs().sum() == 0: torch.nn.init.normal_(p, std=0.3)       # zero-initialised output matrices would hide leaks

def test_no_leak(cfg):
    m = make(cfg); randomise(m)
    g = torch.Generator().manual_seed(1)
    img = torch.randint(0, cfg.P, (1, cfg.S * cfg.S), generator=g)
    cls = torch.tensor([[3, 1, 0, 0, 0, 0, 0, 0]]) if cfg.n_words else None
    th = torch.randint(0, cfg.P, (1, (cfg.S // cfg.F) ** 2), generator=g) if cfg.F else None    # the thumbnail is given, not read from img
    base = m(img, cls, th)[0][0].flatten(0, 1)                             # (pixels in writing order, P)
    order = m.perm.flatten()                                           # writing order -> raster index
    bad = 0
    for gi in torch.randint(0, len(order), (60,), generator=g).tolist() + [0, 1, 15, 16, len(order) - 1]:
        im2 = img.clone(); im2[0, order[gi]] = (im2[0, order[gi]] + 1 + torch.randint(0, cfg.P - 1, (1,), generator=g)) % cfg.P
        new = m(im2, cls, th)[0][0].flatten(0, 1)
        diff = (new - base).abs().amax(-1) > 1e-5
        assert not diff[:gi + 1].any(), f"pixel {gi} leaks into positions {diff[:gi + 1].nonzero().flatten().tolist()}"
        bad += int(diff[gi + 1:].any()) if gi + 1 < len(order) else 1
    assert bad >= 60, "changing a pixel should change later predictions"

def test_probs(cfg):
    m = make(cfg); randomise(m)
    img = torch.randint(0, cfg.P, (2, cfg.S * cfg.S))
    lp = m(img)[0]
    assert torch.allclose(lp.exp().sum(-1), torch.ones_like(lp[..., 0]), atol=1e-4)

def test_sample_consistent(cfg):
    """A sampled picture, scored by the training path, must give the same distribution at every pixel that the
    sampler used: with temperature near zero the sampler must pick the argmax of the training-path scores."""
    m = make(cfg); randomise(m)
    th = torch.randint(0, cfg.P, (2, (cfg.S // cfg.F) ** 2)) if cfg.F else None
    img = m.sample(2, "cpu", temp=1e-4, thumb=th)
    lp, cells, _ = m(img, None, th)
    q = lp.view(*lp.shape[:-1], cfg.G, cfg.P // cfg.G)
    grp = q.logsumexp(-1).argmax(-1)                                   # most likely colour group ...
    mem = q.gather(-2, grp[..., None, None].expand(*grp.shape, 1, cfg.P // cfg.G))[..., 0, :].argmax(-1)   # ... then its best colour
    assert (grp * (cfg.P // cfg.G) + mem == cells).float().mean() > 0.999

if __name__ == "__main__":
    for cfg in (tiny(), tiny(up_in=False, mlp="dense"), tiny(n_words=5), tiny(n_words=5, cls_layers=True, S=16, C=1), tiny(S=16), tiny(C=8, S=32), tiny(C=8, S=64),
                tiny(F=4), tiny(C=8, F=4), tiny(S=8, C=2), tiny(S=16, C=4, F=2), tiny(S=64, C=8, F=4)):
        test_no_leak(cfg); test_probs(cfg)
    test_sample_consistent(tiny(S=16)); test_sample_consistent(tiny(S=16, wq="fp")); test_sample_consistent(tiny(S=16, F=4))
    for name, k in (("koko-size", {}), ("D192", dict(D=192, DK=192, DV=96, HD=96)), ("D160 L6", dict(D=160, DK=160, DV=80, HD=96, L=6, att_layers=[1, 3, 5]))):
        c = ImgConfig(**k); print(name, ImgModel(c).counts(), gba_cost(c))
    print("ok")
