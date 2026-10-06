// core.js - GA-FeCl3 photo analysis, pure functions (browser and Node).
// Mirrors code/08c_app_model.py (features) and the ROI rules of code/02a_roi_detect.py (v3), simplified:
// circle instead of ellipse, no satellite "core" step. All processing is on the device; nothing is uploaded.

export const VERSION = '0.1.1';

// ---------- colour ----------
function srgbToLinear(c) { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
// sRGB (0-255) -> CIELAB, D65, matching skimage.color.rgb2lab
export function rgbToLab(r, g, b) {
  const R = srgbToLinear(r), G = srgbToLinear(g), B = srgbToLinear(b);
  let x = 0.412453 * R + 0.357580 * G + 0.180423 * B;
  let y = 0.212671 * R + 0.715160 * G + 0.072169 * B;
  let z = 0.019334 * R + 0.119193 * G + 0.950227 * B;
  x /= 0.95047; z /= 1.08883;
  const f = t => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(x), fy = f(y), fz = f(z);
  const L = 116 * fy - 16;   // skimage convention (f() linear branch below 0.008856)
  return [L, 500 * (fx - fy), 200 * (fy - fz)];
}
// CIEDE2000 (Sharma, Wu, Dalal 2005), kL = kC = kH = 1
export function ciede2000(lab1, lab2) {
  const [L1, a1, b1] = lab1, [L2, a2, b2] = lab2, rad = Math.PI / 180, deg = 180 / Math.PI;
  const C1 = Math.hypot(a1, b1), C2 = Math.hypot(a2, b2), Cb = (C1 + C2) / 2;
  const G = 0.5 * (1 - Math.sqrt(Math.pow(Cb, 7) / (Math.pow(Cb, 7) + Math.pow(25, 7))));
  const a1p = (1 + G) * a1, a2p = (1 + G) * a2;
  const C1p = Math.hypot(a1p, b1), C2p = Math.hypot(a2p, b2);
  const hp = (a, b) => (a === 0 && b === 0 ? 0 : ((Math.atan2(b, a) * deg) + 360) % 360);
  const h1p = hp(a1p, b1), h2p = hp(a2p, b2);
  const dLp = L2 - L1, dCp = C2p - C1p;
  let dhp = 0;
  if (C1p * C2p !== 0) { dhp = h2p - h1p; if (dhp > 180) dhp -= 360; else if (dhp < -180) dhp += 360; }
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin((dhp / 2) * rad);
  const Lbp = (L1 + L2) / 2, Cbp = (C1p + C2p) / 2;
  let hbp = h1p + h2p;
  if (C1p * C2p !== 0) { if (Math.abs(h1p - h2p) <= 180) hbp /= 2; else hbp = (h1p + h2p < 360) ? (h1p + h2p + 360) / 2 : (h1p + h2p - 360) / 2; }
  const T = 1 - 0.17 * Math.cos((hbp - 30) * rad) + 0.24 * Math.cos(2 * hbp * rad) + 0.32 * Math.cos((3 * hbp + 6) * rad) - 0.20 * Math.cos((4 * hbp - 63) * rad);
  const dTheta = 30 * Math.exp(-Math.pow((hbp - 275) / 25, 2));
  const RC = 2 * Math.sqrt(Math.pow(Cbp, 7) / (Math.pow(Cbp, 7) + Math.pow(25, 7)));
  const SL = 1 + (0.015 * Math.pow(Lbp - 50, 2)) / Math.sqrt(20 + Math.pow(Lbp - 50, 2));
  const SC = 1 + 0.045 * Cbp, SH = 1 + 0.015 * Cbp * T, RT = -Math.sin(2 * dTheta * rad) * RC;
  return Math.sqrt(Math.pow(dLp / SL, 2) + Math.pow(dCp / SC, 2) + Math.pow(dHp / SH, 2) + RT * (dCp / SC) * (dHp / SH));
}

// ---------- features (identical definitions to 08c_app_model.py) ----------
export function features(drop, bg) {
  const cl = v => Math.min(255, Math.max(0, v));
  const d = drop.map(cl), b = bg.map(cl);
  const A = (x, y) => -Math.log10((x + 0.5) / (y + 0.5));
  const gray = v => 0.299 * v[0] + 0.587 * v[1] + 0.114 * v[2];
  return [A(d[0], b[0]), A(d[1], b[1]), A(d[2], b[2]), A(gray(d), gray(b)),
          ciede2000(rgbToLab(b[0], b[1], b[2]), rgbToLab(d[0], d[1], d[2]))];
}

// ---------- random forest (sklearn export: f, t, l, r, v per tree) ----------
export function predictForest(model, x) {
  let s = 0;
  for (const t of model.trees) {
    let n = 0;
    while (t.l[n] !== -1) n = (x[t.f[n]] <= t.t[n]) ? t.l[n] : t.r[n];
    s += t.v[n];
  }
  return s / model.trees.length;
}

// ---------- interpretation ----------
// Bands follow Phase 5 (<=20 / 100 / 500 / >=1000 mg/L); cut points are geometric midpoints.
export const BANDS = [
  { max: Math.log10(44.7), key: 'nd', label: 'Not detected', detail: 'Below about 50 mg/L. Gallic acid may be absent or below what this method can see (detectable from about 100 mg/L).' },
  { max: Math.log10(223.6), key: 'b100', label: 'About 100 mg/L', detail: 'Low level, near the detection limit of the method.' },
  { max: Math.log10(707.1), key: 'b500', label: 'About 500 mg/L', detail: 'Middle of the measuring range.' },
  { max: Infinity, key: 'sat', label: '1000 mg/L or more', detail: 'The colour is saturated. Dilute the sample 10 times with water, test again and multiply the result by 10.' },
];
export function interpret(model, log10pred) {
  const band = BANDS.find(b => log10pred <= b.max);
  const est = Math.pow(10, log10pred), f80 = Math.pow(10, model.abs_err_q80);
  const showEstimate = band.key === 'b100' || band.key === 'b500';
  return { band, estimate_mgL: est, range80_mgL: [est / f80, est * f80], showEstimate };
}

// ---------- image helpers ----------
export function toGray(rgba, n) {
  const g = new Float32Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) g[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
  return g;
}
export function boxBlur(src, w, h, r) {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h), k = 2 * r + 1;
  for (let y = 0; y < h; y++) { let s = 0; for (let x = -r; x <= r; x++) s += src[y * w + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) { tmp[y * w + x] = s / k; s += src[y * w + Math.min(w - 1, x + r + 1)] - src[y * w + Math.max(0, x - r)]; } }
  for (let x = 0; x < w; x++) { let s = 0; for (let y = -r; y <= r; y++) s += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) { out[y * w + x] = s / k; s += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x]; } }
  return out;
}
function bilinear(img, w, h, x, y) {
  if (x < 0 || y < 0 || x > w - 1.001 || y > h - 1.001) return NaN;
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0, i = y0 * w + x0;
  return img[i] * (1 - fx) * (1 - fy) + img[i + 1] * fx * (1 - fy) + img[i + w] * (1 - fx) * fy + img[i + w + 1] * fx * fy;
}
export function median(arr) {
  if (!arr.length) return NaN;
  const a = Float64Array.from(arr).sort(), m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}
function fitCircle(pts) { // algebraic (Kasa) least squares
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, sxz = 0, syz = 0, sz = 0; const n = pts.length;
  for (const [x, y] of pts) { const z = x * x + y * y; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y; sxz += x * z; syz += y * z; sz += z; }
  const M = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]], v = [sxz, syz, sz];
  const det = m => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const D = det(M); if (Math.abs(D) < 1e-9) return null;
  const sol = [0, 1, 2].map(k => det(M.map((row, i) => row.map((c, j) => (j === k ? v[i] : c)))) / D);
  const cx = sol[0] / 2, cy = sol[1] / 2, r = Math.sqrt(sol[2] + cx * cx + cy * cy);
  return { cx, cy, r };
}
function robustCircle(pts) {
  let fit = fitCircle(pts), use = pts;
  for (let it = 0; it < 2 && fit; it++) {
    const res = use.map(([x, y]) => Math.abs(Math.hypot(x - fit.cx, y - fit.cy) - fit.r));
    const mad = Math.max(1, 1.4826 * median(res.map(e => e)));
    const keep = use.filter((p, i) => res[i] <= 2.5 * mad);
    if (keep.length < 12) break;
    use = keep; fit = fitCircle(use);
  }
  return fit ? { ...fit, inliers: use.length } : null;
}
// strongest edge along each of nRay rays between r0 and r1
function rayEdges(sm, w, h, cx, cy, r0, r1, nRay) {
  const pts = [];
  for (let k = 0; k < nRay; k++) {
    const a = (2 * Math.PI * k) / nRay, c = Math.cos(a), s = Math.sin(a);
    let best = 0, br = -1;
    for (let r = Math.max(3, r0); r <= r1; r += 0.5) {
      const g = bilinear(sm, w, h, cx + (r + 2) * c, cy + (r + 2) * s) - bilinear(sm, w, h, cx + (r - 2) * c, cy + (r - 2) * s);
      if (Number.isNaN(g)) break;
      if (Math.abs(g) > best) { best = Math.abs(g); br = r; }
    }
    if (br > 0 && best >= 3) pts.push([cx + br * c, cy + br * s]);
  }
  return pts;
}

// ---------- ROI + measurement ----------
// rgba: Uint8ClampedArray (w*h*4); (tx, ty): the user's tap on the drop.
export function analyse(rgba, w, h, tx, ty, opt = {}) {
  const nRay = 90, n = w * h, warn = [];
  const gray = toGray(rgba, n), sm = boxBlur(gray, w, h, 2);
  const rmax = Math.round(opt.rmax || 0.25 * Math.min(w, h));
  // Prior: two taps (centre, then edge) give centre and approximate radius; with one tap only, the strongest
  // edge on rays out to rmax is used (unreliable on real photos: highlights, film edges and cards attract it).
  let c;
  if (opt.edge) {
    const r0 = Math.hypot(opt.edge[0] - tx, opt.edge[1] - ty);
    if (r0 < 6) return { ok: false, error: 'Tap the centre of the drop, then tap its edge.' };
    c = { cx: tx, cy: ty, r: r0 };
    for (let it = 0; it < 2; it++) {   // refine between 0.6 and 1.5 radii, as in 02a
      const ci = robustCircle(rayEdges(sm, w, h, c.cx, c.cy, Math.max(4, 0.6 * c.r), 1.5 * c.r, nRay));
      if (!ci || ci.r < 0.5 * r0 || ci.r > 2 * r0) return { ok: false, error: 'The drop edge could not be found. Retake the photo in focus, from directly above, and tap the centre and then the edge of the drop.' };
      c = ci;
    }
  } else {
    c = robustCircle(rayEdges(sm, w, h, tx, ty, 6, rmax, nRay));
    if (!c || c.r < 6) return { ok: false, error: 'No drop edge found around the tapped point. Tap the centre of the drop.' };
    const c2 = robustCircle(rayEdges(sm, w, h, c.cx, c.cy, Math.max(4, 0.6 * c.r), Math.min(rmax * 1.5, 1.5 * c.r), nRay));
    if (c2 && c2.r > 6) c = c2;
  }
  const inlierFrac = c.inliers / nRay;
  if (Math.hypot(c.cx - tx, c.cy - ty) > c.r) warn.push('The detected drop is not where you tapped. Check the outline and tap the drop centre again.');
  const R = c.r, rIn = 0.70 * R, rB0 = 1.6 * R, rB1 = 2.3 * R;
  // film ring
  const ringIdx = []; let ringTotal = 0;
  const x0 = Math.floor(c.cx - rB1), x1 = Math.ceil(c.cx + rB1), y0 = Math.floor(c.cy - rB1), y1 = Math.ceil(c.cy + rB1);
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const d = Math.hypot(x - c.cx, y - c.cy);
    if (d < rB0 || d > rB1) continue; ringTotal++;
    if (x < 0 || y < 0 || x >= w || y >= h) continue; ringIdx.push(y * w + x);
  }
  if (ringIdx.length < 0.7 * ringTotal) warn.push('Too little film is visible around the drop. Keep at least three drop-widths of film around it.');
  const rg = ringIdx.map(i => gray[i]), rmed = median(rg), rmad = 1.4826 * median(rg.map(v => Math.abs(v - rmed))) || 1;
  const ring = ringIdx.filter((i, k) => Math.abs(rg[k] - rmed) <= 3 * rmad);
  // drop disc minus highlights (gray > ring median + 10, or any channel >= 245), dilated by 2 px
  const dx0 = Math.floor(c.cx - rIn) - 2, dy0 = Math.floor(c.cy - rIn) - 2, dw = Math.ceil(2 * rIn) + 5, dh = dw;
  const hl = new Uint8Array(dw * dh);
  for (let yy = 0; yy < dh; yy++) for (let xx = 0; xx < dw; xx++) {
    const x = dx0 + xx, y = dy0 + yy; if (x < 0 || y < 0 || x >= w || y >= h) continue;
    const i = y * w + x, j = 4 * i;
    if (gray[i] > rmed + 10 || rgba[j] >= 245 || rgba[j + 1] >= 245 || rgba[j + 2] >= 245) hl[yy * dw + xx] = 1;
  }
  const dropIdx = []; let discN = 0;
  for (let yy = 0; yy < dh; yy++) for (let xx = 0; xx < dw; xx++) {
    const x = dx0 + xx, y = dy0 + yy;
    if (Math.hypot(x - c.cx, y - c.cy) > rIn || x < 0 || y < 0 || x >= w || y >= h) continue; discN++;
    let bad = 0;
    for (let a = -2; a <= 2 && !bad; a++) for (let b = -2; b <= 2; b++) { const u = xx + b, v = yy + a; if (u >= 0 && v >= 0 && u < dw && v < dh && hl[v * dw + u]) { bad = 1; break; } }
    if (!bad) dropIdx.push(y * w + x);
  }
  const hlFrac = discN ? 1 - dropIdx.length / discN : 1;
  const chMed = idx => [0, 1, 2].map(k => median(idx.map(i => rgba[4 * i + k])));
  const drop = chMed(dropIdx), bg = chMed(ring);
  // quality checks (thresholds provisional; to be tuned on the room-light photo set)
  if (dropIdx.length < 300) warn.push('The drop is too small in the photo. Move the phone closer (the drop should be at least one-tenth of the picture width).');
  if (inlierFrac < 0.6) warn.push('The drop edge is unclear. Retake the photo in focus, from directly above.');
  if (hlFrac > 0.3) warn.push('Strong glare on the drop. Tilt the light or the phone slightly and retake.');
  if (Math.max(...bg) >= 250) warn.push('The film is overexposed (pure white). Reduce brightness or move away from direct light.');
  if (0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2] < 60) warn.push('The photo is too dark. Use more light.');
  const q = [0, 0, 0, 0].map(() => []);
  for (const i of ring) { const x = i % w, y = (i / w) | 0; q[(x >= c.cx ? 1 : 0) + (y >= c.cy ? 2 : 0)].push(gray[i]); }
  const qm = q.filter(a => a.length > 20).map(median);
  const unev = qm.length ? Math.max(...qm) - Math.min(...qm) : 0;
  if (unev > 15) warn.push('Uneven light or a shadow around the drop. Make sure no shadow of the phone or hand falls on the drop or the film.');
  return { ok: true, circle: { cx: c.cx, cy: c.cy, r: R }, inlierFrac, hlFrac, nDrop: dropIdx.length, nRing: ring.length,
           drop, bg, ringUnevenness: unev, x: features(drop, bg), warnings: warn };
}
