"""Early-exit limits for the MLPs' first matrices (pix/DESIGN.md, "early exit of the first MLP matrix").
For every dense layer and every check point (REM groups of four inputs still to come) a limit t: the engine drops a
row whose partial sum is below t there. The limits are chosen on calibration runs to save as many table lookups as
possible while the rows that would have ended up active but are dropped carry, on average over the layers, at most
EPS of the layer's hidden energy (sum of squared row sums of the active rows).
    python3 tools/w1cal.py b16C=EPS,b16C2=EPS,u32M=EPS,... dumps...   (NAME2: the limits for two streams)   (dumps: twin built with -DPIX_W1PROBE, W1DUMP=file; writes pix/w1exit.json)"""
import sys, json, math; sys.path.insert(0, 'tools')
from w1lib import *
REM = [28, 24, 20, 16, 12, 8, 4]
def readdump(fns, every=1, two=0):
    recs = {}
    for fn in fns:
        raw = np.fromfile(fn, np.int32); i = 0; k = 0
        while i < len(raw):
            m, l, s, D = raw[i:i+4]
            if k % every == 0: recs.setdefault((int(m), int(l)) if not two else (int(m), int(l), int(s)), []).append(raw[i+4:i+4+D])
            i += 4 + D; k += 1
    return recs
def in0_of(b, f, l, sh=0):
    sft = 12 + f['HS']; T = (2 * f['TAUB'] + 1) << (sft - 1); p0 = math.isqrt(T - 1) + 1
    m = int(b[f['lay'][l]['W1M']:f['lay'][l]['W1M'] + 4].view(np.int32)[0]) << sh
    return -(-((p0 << 16) - 32768) // m)
def partials(X, T, G, rem=REM):
    cps = [G - 1 - r for r in rem if r < G - 1]
    P = np.cumsum(np.einsum('ngk,rgk->nrg', X.reshape(len(X), G, 4), T.reshape(-1, G, 4)), axis=2)
    return P[:, :, cps], P[:, :, -1], cps
class Layer:
    """rows of the calibration runs: P (n, checks) partial sums, v energy if active at the end (else 0)"""
    def __init__(self, P, S, in0, rem, S2=None):
        self.P = P.reshape(-1, P.shape[2]); S = S.reshape(-1); self.v = np.where(S >= in0, S.astype(np.float64) ** 2, 0)
        if S2 is not None: S2 = S2.reshape(-1); self.v = self.v + np.where(S2 >= in0, S2.astype(np.float64) ** 2, 0)      # two streams, P = the sum of both
        self.rem = np.array(rem[-P.shape[2]:], np.float64); self.tot = self.v.sum(); self.n = len(S)
        self.order = [np.argsort(self.P[:, j], kind='stable') for j in range(P.shape[2])]
    def fire(self, lim):
        ng = self.P < np.array(lim)[None, :]; ex = ng.any(1); first = ng.argmax(1)
        return ex, first
    def score(self, lim):
        ex, first = self.fire(lim)
        return np.where(ex, self.rem[first], 0).sum() / self.n, self.v[ex].sum() / self.tot    # groups saved per row, share of energy lost
    def fit(self, lam, rounds=3):
        """limits that maximise groups saved - lam * rows * (share of energy lost)"""
        J = self.P.shape[1]; lim = [-2 ** 30] * J; cost = self.v / self.tot * self.n * lam
        for _ in range(rounds):
            for j in range(J):
                l2 = list(lim); l2[j] = -2 ** 30
                ex, first = self.fire(l2)
                later = ex & (first > j); earlier = ex & (first < j)
                gain = np.where(earlier, 0, self.rem[j] - np.where(later, self.rem[first], 0) - np.where(later, 0, cost))
                o = self.order[j]; c = np.cumsum(gain[o]); k = int(c.argmax())
                if c[k] <= 0: lim[j] = -2 ** 30
                else:
                    p = self.P[o, j]; lim[j] = int(p[k]) + 1
                    if k + 1 < len(p) and p[k + 1] == p[k]: lim[j] = int(p[k])     # (ties: stay below them)
        return lim
def calibrate(layers, eps):
    lo, hi = 0.0, 1e7                                       # bisect the price of lost energy
    for _ in range(18):
        lam = math.sqrt(max(lo, 1.0) * hi) if lo else hi / 100 if hi > 1e4 else hi / 2
        lims = [L.fit(lam, 2) for L in layers]; lost = np.mean([L.score(t)[1] for L, t in zip(layers, lims)])
        if lost > eps: lo = lam
        else: hi = lam
    return [L.fit(hi) for L in layers]
if __name__ == '__main__':
    eps = dict((a.split('=')[0], float(a.split('=')[1])) for a in sys.argv[1].split(',')); b, ms = load(); names = open('gen/pix_layout.h').read().split('/* models: ')[1].split(' */')[0].split()
    recs = readdump(sys.argv[2:], 1, 1); out = {}; out2 = {}
    def stack(m, l, ss, n=400):
        xs = [x for s in ss for x in recs.get((m, l, s), [])]
        return np.array(xs[::max(1, len(xs) // n)]).astype(np.int64)
    for m, name in enumerate(names):
        f = ms[m]; G = f['D'] // 4
        if name in eps:                                      # one stream (calibrated on both kinds of run)
            layers = []
            for l in range(f['L']):
                P, S, cps = partials(stack(m, l, (0, 1, 2)), w1_nat(b, f, l), G); layers.append(Layer(P, S, in0_of(b, f, l), REM))
            lims = calibrate(layers, eps[name]); out[name] = lims
            for l, (L, t) in enumerate(zip(layers, lims)):
                s = L.score(t); print(name, l, 'limits', t, 'groups saved %.1f%%  energy lost %.4f%%' % (100 * s[0] / G, 100 * s[1]))
        if name + '2' in eps and (m, 0, 1) in recs:          # two streams: the sum of both, in the packed tables' units
            layers = []
            for l in range(f['L']):
                n = min(len(recs[(m, l, 1)]), len(recs[(m, l, 2)])); k = max(1, n // 400); T = w1_nat(b, f, l)
                X0 = (np.array(recs[(m, l, 1)][:n:k]).astype(np.int64) + 32) >> 6; X1 = (np.array(recs[(m, l, 2)][:n:k]).astype(np.int64) + 32) >> 6
                P0, S0, cps = partials(X0, T, G); P1, S1, cps = partials(X1, T, G); layers.append(Layer(P0 + P1, S0, in0_of(b, f, l, 6), REM, S1))
            lims = [[max(-4096, min(4096, t)) for t in tt] for tt in calibrate(layers, eps[name + '2'])]; out2[name] = lims
            for l, (L, t) in enumerate(zip(layers, lims)):
                s = L.score(t); print(name, l, 'two streams: limits', t, 'groups saved %.1f%%  energy lost %.4f%%' % (100 * s[0] / G, 100 * s[1]))
    json.dump({'eps': eps, 'rem': REM, 'limits': out, 'limits2': out2}, open('pix/w1exit.json', 'w'))
