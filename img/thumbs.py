"""Thumbnail set from a picture set: python3 -m img.thumbs <src dir> <F> <out dir>
Each F x F block becomes the palette colour nearest to its mean; same palette as the source."""
import json, os, shutil, sys
import numpy as np, torch

src, Fc, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
meta = json.load(open(f"{src}/meta.json"))
S, Ts = meta["S"], meta["S"] // Fc
dev = "cuda" if torch.cuda.is_available() else "cpu"
pal = torch.from_numpy(np.load(f"{src}/palette.npy")).float().to(dev) / 255.0
os.makedirs(out, exist_ok=True)
for part in ("train", "val"):
    a = np.memmap(f"{src}/{part}.u8", np.uint8, mode="r").reshape(-1, S * S)
    with open(f"{out}/{part}.u8", "wb") as f:
        for i in range(0, len(a), 8192):
            x = torch.from_numpy(np.asarray(a[i:i + 8192]).astype(np.int64)).to(dev)
            rgb = pal[x].view(-1, Ts, Fc, Ts, Fc, 3).mean((2, 4)).view(-1, Ts * Ts, 3)
            f.write(torch.cdist(rgb, pal[None].expand(len(rgb), -1, -1)).argmin(-1).cpu().numpy().astype(np.uint8).tobytes())
shutil.copy(f"{src}/palette.npy", f"{out}/palette.npy")
json.dump({**meta, "S": Ts, "thumb_of": src, "F": Fc}, open(f"{out}/meta.json", "w"))
print(out, Ts, flush=True)
