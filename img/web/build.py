"""img/web/page.html + the cartridge's engine compiled to WebAssembly + the cartridge's model file -> two builds of the same page.
  python3 img/web/build.py [credits-openimages.txt]

The engine is pix/pix.c (the cartridge's own C code) with web_wasm/pixwasm.c, compiled with clang for wasm32 and
embedded in the page; the models are build/pix.bin (python3 -m pix.export ...), shipped in two halves (pix0, pix1)
because the artifact host takes at most 15 MB per file. The browser paints exactly what the native twin paints.

  img/web/artifact/   picture.html (a page fragment) + the models wrapped as data-only .wasm files, each under 15 MB
  img/web/itch/       index.html (a whole document) + the models as plain .bin + CREDITS.txt + credits-openimages.txt
  img/web/pocketpix-web.zip   the itch folder zipped with index.html at the top level, ready to upload to itch.io

The models are build/pix.bin (made by `python3 -m pix.export b16C u32M u64M b16O`), split in two halves; the
per-photo credits file comes from `python3 -I img/web/credits.py`
(if it is not given, the one already in img/web/itch/ is kept). Cartridge timings come from gba_times.json.

Uploading to itch.io: kind of project "HTML", upload pocketpix-web.zip and tick "This file will be played in the
browser". Embed options: viewport 960 x 640 or larger, "Mobile friendly" on, "Fullscreen button" on. The itch build
scrolls inside itself, so it works whether or not "Enable scrollbars" is ticked."""
import json, os, sys, zipfile, numpy as np
import base64, subprocess, tempfile
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(os.path.dirname(here)); credits_src = sys.argv[1] if len(sys.argv) > 1 else None
night = f'{here}/../remote/night'
MODELS = ['pix0', 'pix1']                          # the two halves of build/pix.bin (all four models, as on the cartridge)
CREDITS_OI = 'credits-openimages.txt'

def leb(n):
    out = bytearray()
    while True:
        b = n & 0x7F; n >>= 7; out.append(b | (0x80 if n else 0))
        if not n: return bytes(out)
sec = lambda i, body: bytes([i]) + leb(len(body)) + body
def data_module(d):
    return (b'\x00asm\x01\x00\x00\x00' + sec(5, leb(1) + b'\x00' + leb((len(d) + 65535) >> 16))
            + sec(7, leb(1) + leb(1) + b'm' + b'\x02' + leb(0)) + sec(11, leb(1) + b'\x00' + b'\x41\x00\x0b' + leb(len(d)) + d))

words = json.load(open(f'{night}/words.json'))
assert len(words) == 4096 and all(isinstance(w, str) and (w.isascii() and w.isalnum() and w == w.lower()) for w in words[2:])
gba = json.load(open(f'{here}/gba_times.json')); gba.pop('_comment', None)
template = open(f'{here}/page.html').read()
def wasm_engine():
    with tempfile.TemporaryDirectory() as t:
        cf = ['clang', '--target=wasm32', '-O3', '-msimd128', '-ffreestanding', '-nostdlib', f'-I{root}/gen', f'-I{root}/pix', '-Wall']
        subprocess.run(cf + ['-c', f'{root}/pix/pix.c', '-o', f'{t}/a.o'], check=True)
        subprocess.run(cf + ['-c', f'{root}/web_wasm/pixwasm.c', '-o', f'{t}/b.o'], check=True)
        mb = 64 * 2 ** 20
        subprocess.run(['wasm-ld', '--no-entry', '-z', 'stack-size=1048576', f'--initial-memory={mb}', f'--max-memory={mb}', f'{t}/a.o', f'{t}/b.o', '-o', f'{t}/e.wasm'], check=True)
        return open(f'{t}/e.wasm', 'rb').read()
eng = 'const PIXWASM = "' + base64.b64encode(wasm_engine()).decode() + '";'
pal = np.load(f'{here}/../remote/pal/p16o.npy').reshape(-1).tolist(); assert len(pal) == 768
for bad in ('http://', 'https://', '//fonts.', '@import'):          # nothing may be fetched from anywhere else
    assert bad not in template.replace('http://localhost:8000', ''), bad
blob = open(f'{root}/build/pix.bin', 'rb').read(); half = (len(blob) // 2 + 15) & ~15
data = {'pix0': blob[:half], 'pix1': blob[half:]}; OFF = {'pix0': 0, 'pix1': half}

def page(files, cfg):
    p = template
    for k, v in (('/*ENGINE*/', eng), ('/*WORDS*/', json.dumps(words, separators=(',', ':'))), ('/*PAL*/', json.dumps(pal, separators=(',', ':'))),
                 ('/*FILES*/', json.dumps(files)), ('/*CFG*/', json.dumps(cfg)), ('/*GBA*/', json.dumps(gba))):
        assert p.count(k) == 1, k; p = p.replace(k, v)
    return p

# (a) the claude.ai artifact
art = f'{here}/artifact'; os.makedirs(art, exist_ok=True); files = {}
for f in os.listdir(art):                                                             # models the cartridge dropped, stale test pages
    if f != 'picture.html' and f not in [n + '.wasm' for n in MODELS] and os.path.isfile(f'{art}/{f}'): os.remove(f'{art}/{f}')
for n in MODELS:
    w = data_module(data[n]); assert len(w) < 15 * 2 ** 20, n
    open(f'{art}/{n}.wasm', 'wb').write(w)
    files[n] = {'url': n + '.wasm', 'size': len(data[n]), 'wasm': True, 'off': OFF[n]}
open(f'{art}/picture.html', 'w', encoding='utf-8').write(page(files, {'target': 'artifact', 'credits': None}))

# (b) itch.io: a plain static folder
CREDITS = f"""PocketPix (browser version): credits and licences
==================================================

PocketPix is a small text-to-picture model made to fit on a Game Boy Advance cartridge.
This folder holds its browser version: index.html and the cartridge's model file in two halves
(pix0.bin, pix1.bin). Everything runs in the browser; nothing is sent anywhere.
The page uses no outside code, fonts or services.

What the models were trained on
-------------------------------
* CommonCatalog, CC-BY subset (https://huggingface.co/datasets/common-canvas/commoncatalog-cc-by),
  the photograph collection of the CommonCanvas project (Gokaslan et al., "CommonCanvas: An Open
  Diffusion Model Trained with Creative-Commons Images", 2023). These are photographs their authors
  published on Flickr under Creative Commons Attribution (CC BY) licences. They were read from a
  512-pixel repackaging of that subset (data-archetype/commoncatalog_cc_by_3m_512 on Hugging Face).
  Which photographs: every photograph in 240 shards of that repackaging (about 0.98 million); the
  shards are listed in img/commoncatalog_shards.txt at https://github.com/rdsilver/pocketpix
  Who took them: the Flickr users recorded for each photograph in CommonCatalog (columns unickname,
  photoid, pageurl, licensename, licenseurl; Flickr's "Attribution License" is CC BY 2.0). A photo's
  name in the shards is the SHA-256 of its JPEG bytes in CommonCatalog, which links the two.
  A per-photograph list is not included yet: no table linking the two sets of names is published,
  and building one means hashing about 21 TB of images. It will be added if such a table appears.
* Open Images V7 (https://storage.googleapis.com/openimages/web/index.html): close-up crops of
  named things. The photographs are licensed CC BY 2.0 (https://creativecommons.org/licenses/by/2.0/);
  the object boxes used to cut the crops are by Google LLC, licensed CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/).
  The author and page of every Open Images photograph used are listed in {CREDITS_OI}.

How the photographs were used: each was cropped, shrunk to 64x64 pixels (and to 32x32 and 16x16) and
reduced to a fixed palette of 256 colours before training. No photograph is included in this download;
the model files hold only the trained numbers. The licensors do not endorse this project.

Not affiliated
--------------
Game Boy Advance is a trademark of Nintendo. This is an unofficial hobby project and is not made or
endorsed by Nintendo. No Nintendo code or data is included.

Open Images credits: where an author's recorded name was an e-mail address, only the part before
the @ is given.

Cartridge times quoted on the page were measured in {gba['measured_in']}.
"""
itch = f'{here}/itch'; os.makedirs(itch, exist_ok=True); files = {}
if credits_src: open(f'{itch}/{CREDITS_OI}', 'wb').write(open(credits_src, 'rb').read())
assert os.path.exists(f'{itch}/{CREDITS_OI}'), 'give the credits file made by img/web/credits.py as the second argument'
for n in MODELS:
    open(f'{itch}/{n}.bin', 'wb').write(data[n]); files[n] = {'url': n + '.bin', 'size': len(data[n]), 'wasm': False, 'off': OFF[n]}
open(f'{itch}/CREDITS.txt', 'w', encoding='utf-8', newline='\n').write(CREDITS)
doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
       + page(files, {'target': 'itch', 'credits': [['CREDITS.txt', 'training data and licences'], [CREDITS_OI, 'every Open Images photograph']]}))
open(f'{itch}/index.html', 'w', encoding='utf-8').write(doc)
keep = ['index.html'] + [n + '.bin' for n in MODELS] + ['CREDITS.txt', CREDITS_OI]
for f in os.listdir(itch):
    if f not in keep: os.remove(f'{itch}/{f}')
zpath = f'{here}/pocketpix-web.zip'
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for f in keep:
        zi = zipfile.ZipInfo(f, (2026, 1, 1, 0, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED; zi.external_attr = 0o644 << 16
        z.writestr(zi, open(f'{itch}/{f}', 'rb').read(), compresslevel=9)
for d, fs in ((art, ['picture.html'] + [n + '.wasm' for n in MODELS]), (itch, keep)):
    for f in fs: print(f'{os.path.getsize(f"{d}/{f}"):>10}  {os.path.relpath(f"{d}/{f}", here)}')
print(f'{os.path.getsize(zpath):>10}  {os.path.relpath(zpath, here)}')
print('itch.io: HTML project, upload the zip, "played in the browser", embed 960 x 640 or larger, mobile friendly + fullscreen button on')
