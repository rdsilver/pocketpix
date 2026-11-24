"""Summarise PPROB outputs (twin PPROB=1 SHADOW=1; called PCOPY in the head64 explorer's tree) (build/pc_*.txt): window-rule cells against body-only cells."""
import re, glob, sys
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'build/pc_*_paint*.txt')):
    d = {}
    for l in open(f):
        k = int(l.split()[1])
        m = re.search(r'pixels (\S+)\s+p\(copy\) model (\S+) sampling (\S+)\s+-ln p model (\S+) sampling (\S+)', l)
        d[k] = [float(x) if x != '-nan' else 0.0 for x in m.groups()]
    n1, n3 = d[1][0], d[3][0]
    w = lambda i: (n1 * d[1][i] + n3 * d[3][i]) / max(1, n1 + n3)
    print(f"{f:30s} window-rule cells: px {int(n1 + n3):6d} p {w(1):.4f} -ln p {w(3):.4f} | body-only cells: px {int(d[2][0]):6d} p {d[2][1]:.4f} -ln p {d[2][3]:.4f}")
