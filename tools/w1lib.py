import numpy as np, re
def load(path='build/pix.bin'):
    lay = open('gen/pix_layout.h').read()
    MF = re.search(r'enum \{ (MF_.*?), MF_N \}', lay).group(1).replace('MF_','').split(', ')
    LF = re.search(r'enum \{ (LF_.*?), LF_N \}', lay).group(1).replace('LF_','').split(', ')
    OM = int(re.search(r'O_MODELS (\d+)', lay).group(1))
    b = np.fromfile(path, np.uint8)
    u32 = lambda o, n: b[o:o+4*n].view(np.uint32)
    models = []
    for m in range(4):
        f = dict(zip(MF, u32(OM + m*len(MF)*4, len(MF)).tolist()))
        ly = u32(f['LAYERS'], f['L']*len(LF)).reshape(f['L'], len(LF))
        f['lay'] = [dict(zip(LF, r.tolist())) for r in ly]
        models.append(f)
    return b, models
def w1_idx(b, f, l):
    """group indices (rows, D/4) 0..80"""
    D = f['D']; hdl = f['lay'][l]['HDL']; o = f['lay'][l]['W1']
    G = D//4; nw = (G+2)//3; f = G-3*(nw-1)
    w = b[o:o+hdl*nw*4].view(np.uint32).reshape(hdl, nw).astype(np.int64)      # T3 words (export.py step_words)
    A, B, C = w >> 24, w >> 11 & 255, w & 255
    pos = np.empty((hdl, G), np.int64); p = np.full(hdl, -1, np.int64); g = 0
    for j in range(nw):
        k = f if j == 0 else 3
        if k >= 2: pos[:, g] = p + B[:, j]; g += 1
        p = p + A[:, j]; pos[:, g] = p; g += 1
        if k == 3: p = p + C[:, j]; pos[:, g] = p; g += 1
    idx = pos - 81*np.arange(G)[None, :]
    assert idx.min()>=0 and idx.max()<81
    return idx
def trits(idx):
    c = np.stack([idx%3, idx//3%3, idx//9%3, idx//27], -1).reshape(idx.shape[0], -1)
    return np.where(c==1, 1, np.where(c==2, -1, 0))
def w1_nat(b, f, l):
    """trits in the model's own input order (undoes LF_W1P)"""
    t = trits(w1_idx(b, f, l)); o = f['lay'][l].get('W1P', 0)
    return t
    if not o: return t
    pm = b[o:o+f['D']].astype(np.int64); n = np.empty_like(t); n[:, pm] = t
    return n
