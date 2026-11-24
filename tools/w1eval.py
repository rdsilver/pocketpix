"""what the limits in pix/w1exit.json do on other dumps: python3 tools/w1eval.py dumps..."""
import sys, json; sys.path.insert(0, 'tools')
from w1lib import *
from w1cal import *
b, ms = load(); names = open('gen/pix_layout.h').read().split('/* models: ')[1].split(' */')[0].split()
lim = json.load(open('pix/w1exit.json'))['limits']; recs = readdump(sys.argv[1:], 3); agg = {}      # (one-stream limits on every stream's inputs)
for (m, l), xs in sorted(recs.items()):
    if names[m] not in lim: continue
    f = ms[m]; G = f['D'] // 4; X = np.array(xs).astype(np.int64)
    P, S, cps = partials(X, w1_nat(b, f, l), G); L = Layer(P, S, in0_of(b, f, l), REM); s = L.score(lim[names[m]][l])
    a = agg.setdefault(m, []); a.append((s[0] / G, s[1], L.fire(lim[names[m]][l])[0].mean()))
for m, a in agg.items(): a = np.array(a); print(names[m], 'lookups saved %.1f%%  rows dropped early %.1f%%  energy lost %.4f%% (worst layer %.4f%%)' % (100 * a[:, 0].mean(), 100 * a[:, 2].mean(), 100 * a[:, 1].mean(), 100 * a[:, 1].max()))
