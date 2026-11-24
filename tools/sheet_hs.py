"""Contact sheet: twin pictures with and without
PIX_HEADSKIP (top: HEADSKIP=0, bottom: the default), same seeds and prompts.
    python3 tools/sheet.py out.png"""
import os, subprocess, sys, numpy as np
from PIL import Image
pal = np.load('img/remote/pal/p16o.npy').astype(np.uint8)
jobs = [('paint', 7, '288 10'), ('paint', 21, '14 5 300'), ('paint', 33, ''), ('paint3', 5, '500 77'), ('paint3', 9, '288 10'), ('paint', 55, '901 33 7'), ('paint3', 77, ''), ('paint', 91, '1200')]
rows = []
for ex in (dict(HEADSKIP='0'), {}):
    for half in (jobs[:4], jobs[4:]):
        pics = []
        for mode, seed, words in half:
            out = subprocess.run(['build/pix_twin', 'build/pix.bin', mode, str(seed), '768', '284', '2', *words.split()], capture_output=True, env=dict(os.environ, **ex)).stdout
            pics += [np.frombuffer(out, np.uint8).reshape(-1, 64, 64)[i] for i in range(2)]
        rows.append(np.concatenate([np.pad(pal[p], ((2, 2), (2, 2), (0, 0))) for p in pics], 1))
img = np.concatenate(rows[:2] + [np.full((8, rows[0].shape[1], 3), 255, np.uint8)] + rows[2:], 0)
Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(sys.argv[1])
