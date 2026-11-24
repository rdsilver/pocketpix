// node img/web/test_engine.js <model.bin> <prefix of .ref.json/.ref.f32> : compare the engine with PyTorch
const fs = require('fs'), E = require('./engine.js');
const [bin, pre] = process.argv.slice(2);
const b = fs.readFileSync(bin), m = E.loadModel(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
const ref = JSON.parse(fs.readFileSync(pre + '.ref.json')), rb = fs.readFileSync(pre + '.ref.f32');
const lp = new Float32Array(rb.buffer.slice(rb.byteOffset, rb.byteOffset + rb.length)), P = m.cfg.P;
const p = new E.Painter(m, { words: ref.cls, thumb: ref.thumb ? Uint8Array.from(ref.thumb) : null });
p.full = new Float32Array(P);
let maxd = 0, kl = 0, n = 0, bitsRef = 0, bitsJs = 0;
p.onFull = (t, j, full) => {
  const o = (t * m.ns + j) * P; let k = 0;
  for (let i = 0; i < P; i++) { const d = Math.abs(full[i] - lp[o + i]); if (lp[o + i] > -12 && d > maxd) maxd = d; k += Math.exp(lp[o + i]) * (lp[o + i] - full[i]); }
  kl += k; n++; const px = ref.img[m.perm[t * m.ns + j]]; bitsRef -= lp[o + px] / Math.LN2; bitsJs -= full[px] / Math.LN2;
};
const t0 = Date.now();
while (!p.done) p.step(ref.img);
console.log(bin.split('/').pop(), 'max |dlogp| (p > e^-12):', maxd.toExponential(2), ' mean KL:', (kl / n).toExponential(2), ' bits/pixel ref', (bitsRef / n).toFixed(4), 'js', (bitsJs / n).toFixed(4), ' ms', Date.now() - t0);
