/* Picture model engine: the same arithmetic as img/model.py, one cell at a time, with the earlier cells' keys,
   values and running averages kept so each cell is computed once. Works in a page, a worker and node. */
(function (root) {
  'use strict';

  function loadModel(buf) {
    const dv = new DataView(buf), hl = dv.getUint32(0, true);
    const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
    const base = 4 + hl, t = {};
    for (const [name, e] of Object.entries(head.t)) {
      const n = e.shape.reduce((a, b) => a * b, 1), cols = e.shape[e.shape.length - 1], rows = n / cols;
      if (e.kind === 'tern') {
        const bytesPerRow = Math.ceil(cols / 4), src = new Uint8Array(buf, base + e.data, rows * bytesPerRow);
        const w = new Int8Array(n), dec = [0, 1, -1, 0];
        for (let r = 0; r < rows; r++)
          for (let c = 0; c < cols; c++) w[r * cols + c] = dec[(src[r * bytesPerRow + (c >> 2)] >> ((c & 3) * 2)) & 3];
        t[name] = { w, scale: new Float32Array(buf.slice(base + e.scale, base + e.scale + rows * 4)), rows, cols, shape: e.shape };
      } else if (e.kind === 'i8') {
        t[name] = { w: new Int8Array(buf, base + e.data, n), scale: new Float32Array(buf.slice(base + e.scale, base + e.scale + rows * 4)), rows, cols };
      } else if (e.kind === 'u8') {
        t[name] = new Uint8Array(buf, base + e.data, n);
      } else {
        t[name] = { f: new Float32Array(buf.slice(base + e.data, base + e.data + n * 4)), rows, cols };
      }
    }
    return new Model(head.config, t);
  }

  /* y[r] = scale[r] * sum_c w[r0 + r, c] * x[c] for `rows` rows starting at row r0 of a ternary matrix */
  function mat(m, x, y, r0, rows) {
    const w = m.w, cols = m.cols, sc = m.scale;
    for (let r = 0; r < rows; r++) {
      let o = (r0 + r) * cols, s = 0;
      for (let c = 0; c < cols; c++) s += w[o + c] * x[c];
      y[r] = s * sc[r0 + r];
    }
  }
  /* dst += row of a table */
  function addRow(tab, row, dst) {
    const cols = tab.cols, o = row * cols;
    if (tab.f) { const f = tab.f; for (let c = 0; c < cols; c++) dst[c] += f[o + c]; }
    else { const w = tab.w, s = tab.scale[row]; for (let c = 0; c < cols; c++) dst[c] += w[o + c] * s; }
  }
  function rms(g, x, y) {
    const n = x.length; let s = 0;
    for (let i = 0; i < n; i++) s += x[i] * x[i];
    const k = 1 / Math.sqrt(s / n + 1e-5), gf = g.f;
    for (let i = 0; i < n; i++) y[i] = gf[i] * x[i] * k;
  }
  function logSoftmax(v, n, out) {
    let mx = -Infinity;
    for (let i = 0; i < n; i++) if (v[i] > mx) mx = v[i];
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.exp(v[i] - mx);
    const l = mx + Math.log(s);
    for (let i = 0; i < n; i++) out[i] = v[i] - l;
  }

  class Model {
    constructor(cfg, t) {
      this.cfg = cfg; this.t = t;
      const S = cfg.S, C = cfg.C, n = S / C, ns = C * C, nc = n * n;
      this.n = n; this.ns = ns; this.nc = nc; this.P1 = cfg.P + 1;
      this.perm = new Int32Array(nc * ns); this.nb = new Int32Array(nc * ns * 4).fill(-1);
      for (let c = 0; c < nc; c++) {
        const cy = Math.floor(c / n), cx = c % n;
        for (let j = 0; j < ns; j++) {
          const sy = Math.floor(j / C), sx = j % C, y = cy * C + sy, x = cx * C + sx, o = (c * ns + j) * 4;
          this.perm[c * ns + j] = y * S + x;
          if (x > 0) this.nb[o] = y * S + x - 1;
          if (y > 0) this.nb[o + 1] = (y - 1) * S + x;
          if (x > 0 && y > 0) this.nb[o + 2] = (y - 1) * S + x - 1;
          if (y > 0 && x + 1 < S && (sx < C - 1 || sy === 0)) this.nb[o + 3] = (y - 1) * S + x + 1;
        }
      }
      if (cfg.F) {
        const Fc = cfg.F, Ts = S / Fc, m = Math.max(1, Math.floor(C / Fc)), win = (m + 2) * (m + 2);
        this.Ts = Ts; this.win = win;
        this.tw = new Int32Array(nc * win).fill(-1); this.tp = new Int32Array(nc * ns * 5).fill(-1);
        const d5 = [[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]];
        for (let c = 0; c < nc; c++) {
          const cy = Math.floor(c / n), cx = c % n;
          for (let k = 0; k < win; k++) {
            const ky = Math.floor(k / (m + 2)), kx = k % (m + 2);
            const y = Math.floor(cy * C / Fc) + ky - 1, x = Math.floor(cx * C / Fc) + kx - 1;
            if (y >= 0 && y < Ts && x >= 0 && x < Ts) this.tw[c * win + k] = y * Ts + x;
          }
          for (let j = 0; j < ns; j++) {
            const sy = Math.floor(j / C), sx = j % C, by = Math.floor((cy * C + sy) / Fc), bx = Math.floor((cx * C + sx) / Fc);
            for (let q = 0; q < 5; q++) {
              const y = by + d5[q][0], x = bx + d5[q][1];
              if (y >= 0 && y < Ts && x >= 0 && x < Ts) this.tp[(c * ns + j) * 5 + q] = y * Ts + x;
            }
          }
        }
      }
      this.hasAtt = l => cfg.att_layers == null || cfg.att_layers.includes(l);
      const hk = cfg.DK / cfg.NH; this.hk = hk;
      this.freq = new Float64Array(hk / 2);
      for (let i = 0; i < hk / 2; i++) this.freq[i] = 1 / Math.pow(10000, (2 * i) / hk);
    }
  }

  /* One stream of the body (a prompt, or the empty prompt for guidance). */
  class Body {
    constructor(m, words, amp) {
      this.m = m; const c = m.cfg, D = c.D;
      this.words = (words || []).filter(w => w > 0).slice(0, c.W);
      this.first = words && words.length ? words[0] : 0;      // the first slot as given (0 = empty) picks the first cell's expert
      this.x = new Float32Array(D); this.xn = new Float32Array(D); this.y = new Float32Array(D);
      this.hid = new Float32Array(c.HD); this.h = new Float32Array(D);
      this.q = new Float32Array(c.DK); this.k = new Float32Array(c.DK); this.v = new Float32Array(c.DV); this.mix = new Float32Array(c.DV);
      this.L = [];
      for (let l = 0; l < c.L; l++)
        this.L.push(m.hasAtt(l) ? { keys: new Int8Array(m.nc * c.DK), vals: new Float32Array(m.nc * c.DV) }
          : { hist: [new Float32Array(D), new Float32Array(D), new Float32Array(D)], avg: [new Float32Array(D), new Float32Array(D), new Float32Array(D)] });
      this.sc = new Float32Array(m.nc);
      this.cw = new Float32Array(D * c.L);        // prompt rows summed once: [0] for the input, [l] before layer l
      if (c.n_words) {
        for (const w of this.words) {
          addRow(m.t['cls_emb.weight'], w, this.cw.subarray(0, D));
          if (c.cls_layers) for (let l = 1; l < c.L; l++) addRow(m.t[`cls_l.${l - 1}.weight`], w, this.cw.subarray(l * D, (l + 1) * D));
        }
      }
      if (amp && amp !== 1) for (let i = 0; i < this.cw.length; i++) this.cw[i] *= amp;      // experiment: a stronger prompt in one pass
    }
    /* img: pixels written so far (Uint8Array S*S); thumb: Uint8Array or null. Returns h (D) for cell t. */
    cell(t, img, thumb) {
      const m = this.m, c = m.cfg, T = m.t, D = c.D, ns = m.ns, P1 = m.P1, n = m.n, x = this.x, xn = this.xn, y = this.y;
      x.set(T['cell_pos.weight'].f.subarray(t * D, (t + 1) * D));
      for (let j = 0; j < ns; j++) {
        addRow(T['prev_emb.weight'], j * P1 + (t > 0 ? img[m.perm[(t - 1) * ns + j]] : c.P), x);
        if (c.up_in) addRow(T['up_emb.weight'], j * P1 + (t >= n ? img[m.perm[(t - n) * ns + j]] : c.P), x);
      }
      if (c.F) for (let k = 0; k < m.win; k++) { const i = m.tw[t * m.win + k]; addRow(T['th_emb.weight'], k * P1 + (i < 0 ? c.P : thumb[i]), x); }
      for (let i = 0; i < D; i++) x[i] += this.cw[i];
      const g = t > 0 ? img[m.perm[(t - 1) * ns + ns - 1]] >> Math.log2(c.P / c.G) : (c.n_words ? this.first % c.RC : 0);
      const route = t * c.RC + g;
      for (let l = 0; l < c.L; l++) {
        if (c.cls_layers && l) { const o = l * D; for (let i = 0; i < D; i++) x[i] += this.cw[o + i]; }
        rms(T[`n1.${l}.g`], x, xn);
        const st = this.L[l];
        if (m.hasAtt(l)) this.attend(l, t, st); else this.mixer(l, st);
        for (let i = 0; i < D; i++) x[i] += y[i];
        rms(T[`n2.${l}.g`], x, xn);
        const w1 = T[`mlp.${l}.w1`], w2 = T[`mlp.${l}.w2`], hid = this.hid;
        const e = c.mlp === 'moe' ? T[`mlp.${l}.route`][route] : 0;
        mat(w1, xn, hid, e * c.HD, c.HD);
        for (let i = 0; i < c.HD; i++) { const a = hid[i] > 0 ? hid[i] : 0; hid[i] = a * a; }
        mat(w2, hid, y, e * D, D);
        for (let i = 0; i < D; i++) x[i] += y[i];
      }
      rms(T['nf.g'], x, this.h);
      return this.h;
    }
    mixer(l, st) {
      const T = this.m.t, D = this.m.cfg.D, xn = this.xn, y = this.y, taps = T[`att.${l}.taps`].f, avg = T[`att.${l}.avg`].f, R = [0.5, 0.8, 0.95];
      y.fill(0);
      for (let i = 0; i < 3; i++) {
        const h = st.hist[i], a = st.avg[i], r = R[i];
        for (let d = 0; d < D; d++) {
          a[d] = (1 - r) * xn[d] + r * a[d];
          y[d] += taps[i * D + d] * h[d] + avg[i * D + d] * a[d];
        }
      }
      const last = st.hist.pop(); last.set(xn); st.hist.unshift(last);     // hist[i] = input of i + 1 cells ago
    }
    attend(l, t, st) {
      const m = this.m, c = m.cfg, T = m.t, hk = m.hk, dvh = c.DV / c.NH, q = this.q, k = this.k, v = this.v, sc = this.sc;
      mat(T[`att.${l}.q`], this.xn, q, 0, c.DK); mat(T[`att.${l}.k`], this.xn, k, 0, c.DK); mat(T[`att.${l}.v`], this.xn, v, 0, c.DV);
      for (let h = 0; h < c.NH; h++)
        for (let i = 0; i < hk / 2; i++) {
          const ang = t * m.freq[i], cs = Math.cos(ang), sn = Math.sin(ang), o = h * hk + 2 * i;
          let a = q[o], b = q[o + 1]; q[o] = a * cs - b * sn; q[o + 1] = a * sn + b * cs;
          a = k[o]; b = k[o + 1];
          st.keys[t * c.DK + o] = Math.fround(a * cs - b * sn) >= 0 ? 1 : -1; st.keys[t * c.DK + o + 1] = Math.fround(a * sn + b * cs) >= 0 ? 1 : -1;
        }
      st.vals.set(v, t * c.DV);
      const temp = T[`att.${l}.temp`].f, dbias = T[`att.${l}.dbias`], mix = this.mix;
      for (let h = 0; h < c.NH; h++) {
        let s2 = 0;
        for (let i = 0; i < hk; i++) s2 += q[h * hk + i] * q[h * hk + i];
        const u = 1 / Math.sqrt(s2 / hk + 1e-5), scale = Math.exp(Math.max(-3, Math.min(3, temp[h]))) / Math.sqrt(hk) * u;
        for (let j = 0; j <= t; j++) {
          let s = 0; const o = j * c.DK + h * hk;
          for (let i = 0; i < hk; i++) s += q[h * hk + i] * st.keys[o + i];
          sc[j] = s * scale + (dbias ? dbias.f[h * dbias.cols + (t - j)] : 0);
        }
        let floor = -Infinity;
        if (c.topk && m.nc > c.topk && t + 1 > c.topk) {              // keep the best topk (ties with the last one stay)
          const top = new Float32Array(c.topk).fill(-Infinity);
          for (let j = 0; j <= t; j++) {
            const s = sc[j];
            if (s > top[c.topk - 1]) { let p = c.topk - 1; while (p > 0 && top[p - 1] < s) { top[p] = top[p - 1]; p--; } top[p] = s; }
          }
          floor = top[c.topk - 1];
        }
        let mx = -Infinity;
        for (let j = 0; j <= t; j++) if (sc[j] >= floor && sc[j] > mx) mx = sc[j];
        let z = 0;
        for (let i = 0; i < dvh; i++) mix[h * dvh + i] = 0;
        for (let j = 0; j <= t; j++) {
          if (sc[j] < floor) continue;
          const a = Math.exp(sc[j] - mx); z += a;
          const o = j * c.DV + h * dvh;
          for (let i = 0; i < dvh; i++) mix[h * dvh + i] += a * st.vals[o + i];
        }
        for (let i = 0; i < dvh; i++) mix[h * dvh + i] /= z;
      }
      mat(T[`att.${l}.o`], mix, this.y, 0, c.D);
    }
  }

  /* Draws one picture. words: prompt word ids (may be empty); thumb: the smaller picture this one is drawn over. */
  class Painter {
    constructor(m, opt) {
      this.m = m; const c = m.cfg; opt = opt || {};
      this.temp = opt.temp == null ? 1 : opt.temp; this.guide = opt.guide == null ? 1 : opt.guide;
      this.rand = opt.rand || Math.random; this.thumb = opt.thumb || null; this.guideCells = opt.guideCells;
      if (!!c.F !== !!this.thumb) throw new Error('this model ' + (c.F ? 'needs' : 'does not take') + ' a smaller picture to draw over');
      const words = c.n_words ? (opt.words || []) : [];
      this.twice = this.guide !== 1 && c.n_words > 0 && words.some(w => w > 0);
      this.a = new Body(m, words, opt.amp); this.b = this.twice ? new Body(m, []) : null;
      this.img = new Uint8Array(c.S * c.S); this.t = 0;
      const DH = c.DH;
      this.ha = new Float32Array(DH); this.hb = new Float32Array(DH); this.f = new Float32Array(DH); this.cellsum = new Float32Array(DH);
      this.u = new Float32Array(DH); this.z = new Float32Array(DH); this.hh = new Float32Array(c.HH); this.d = new Float32Array(DH);
      this.M = c.P / c.G;
      this.g = [new Float32Array(c.G), new Float32Array(c.G)]; this.mm = [new Float32Array(this.M), new Float32Array(this.M)];
      this.full = null;      // set to a Float32Array(P) to also get every colour's log-probability (tests)
    }
    get done() { return this.t >= this.m.nc; }
    /* group scores for stream s into this.g[s] (log-probabilities); leaves the head state in this.u */
    scoreGroups(hin, s) {
      const m = this.m, c = m.cfg, T = m.t, DH = c.DH, u = this.u, z = this.z, hh = this.hh, d = this.d;
      for (let i = 0; i < DH; i++) u[i] = hin[i] + this.f[i];
      for (let b = 0; b < c.head_blocks; b++) {
        rms(T[`hn.${b}.g`], u, z);
        mat(T[`h1.${b}`], z, hh, 0, c.HH);
        for (let i = 0; i < c.HH; i++) { const a = hh[i] > 0 ? hh[i] : 0; hh[i] = a * a; }
        mat(T[`h2.${b}`], hh, d, 0, DH);
        for (let i = 0; i < DH; i++) u[i] += d[i];
      }
      rms(T['hf.g'], u, z);
      const us = s ? (this.ub || (this.ub = new Float32Array(DH))) : (this.ua || (this.ua = new Float32Array(DH)));
      us.set(z);
      mat(T['out_g'], z, this.g[s], 0, c.G);
      logSoftmax(this.g[s], c.G, this.g[s]);
    }
    scoreMembers(s, gi) {
      mat(this.m.t['out_m'], s ? this.ub : this.ua, this.mm[s], gi * this.M, this.M);
      logSoftmax(this.mm[s], this.M, this.mm[s]);
    }
    pick(lp, n) {
      const t = this.temp; let mx = -Infinity;
      for (let i = 0; i < n; i++) if (lp[i] > mx) mx = lp[i];
      let z = 0; const p = new Float64Array(n);
      for (let i = 0; i < n; i++) { p[i] = Math.exp((lp[i] - mx) / t); z += p[i]; }
      let r = this.rand() * z;
      for (let i = 0; i < n; i++) { r -= p[i]; if (r < 0) return i; }
      return n - 1;
    }
    /* Draw the next cell. force: optional pixel values to write instead of sampling (tests). */
    step(force) {
      const m = this.m, c = m.cfg, T = m.t, ns = m.ns, P1 = m.P1, t = this.t, img = this.img, DH = c.DH, G = c.G, M = this.M;
      if (this.twice && this.guideCells != null && t >= this.guideCells) this.twice = false;      // experiment: guidance only for the top of the picture
      mat(T['h_in'], this.a.cell(t, img, this.thumb), this.ha, 0, DH);
      if (this.twice) mat(T['h_in'], this.b.cell(t, img, this.thumb), this.hb, 0, DH);
      this.cellsum.fill(0);
      for (let j = 0; j < ns; j++) {
        const f = this.f, o = (t * ns + j) * 4;
        f.set(T['slot_emb.weight'].f.subarray(j * DH, (j + 1) * DH));
        for (let q = 0; q < 4; q++) { const i = m.nb[o + q]; addRow(T['nb_emb.weight'], q * P1 + (i < 0 ? c.P : img[i]), f); }
        if (c.F) for (let q = 0; q < 5; q++) { const i = m.tp[(t * ns + j) * 5 + q]; addRow(T['hth_emb.weight'], q * P1 + (i < 0 ? c.P : this.thumb[i]), f); }
        for (let i = 0; i < DH; i++) f[i] += this.cellsum[i];
        this.scoreGroups(this.ha, 0);
        let g = this.g[0];
        if (this.twice) {
          this.scoreGroups(this.hb, 1);
          const g0 = this.g[1]; for (let i = 0; i < G; i++) g0[i] = g0[i] + this.guide * (g[i] - g0[i]);
          g = g0;
        }
        if (this.full) {                                   // tests: every colour, without guidance
          const lg = Float32Array.from(this.g[0]);
          for (let gi = 0; gi < G; gi++) { this.scoreMembers(0, gi); for (let k = 0; k < M; k++) this.full[gi * M + k] = lg[gi] + this.mm[0][k]; }
          if (this.onFull) this.onFull(t, j, this.full);
        }
        let px;
        if (force) px = force[m.perm[t * ns + j]];
        else {
          const gi = this.pick(g, G);
          this.scoreMembers(0, gi);
          let mm = this.mm[0];
          if (this.twice) { this.scoreMembers(1, gi); const m0 = this.mm[1]; for (let i = 0; i < M; i++) m0[i] = m0[i] + this.guide * (mm[i] - m0[i]); mm = m0; }
          px = gi * M + this.pick(mm, M);
        }
        img[m.perm[t * ns + j]] = px;
        addRow(T['in_emb.weight'], j * P1 + px, this.cellsum);
      }
      this.t++;
      return this.t < m.nc;
    }
  }

  function rng(seed) {           // mulberry32
    let a = seed >>> 0;
    return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }

  const api = { loadModel, Painter, rng };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.PicEngine = api;
})(typeof self !== 'undefined' ? self : globalThis);
