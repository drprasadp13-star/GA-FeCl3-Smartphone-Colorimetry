// Node test: node test/run_tests.mjs   (from the app folder)
import fs from 'fs';
import { fileURLToPath } from 'url';
import { rgbToLab, ciede2000, features, predictForest, analyse } from '../core.js';
const here = fileURLToPath(new URL('.', import.meta.url));
const model = JSON.parse(fs.readFileSync(here + '../model.json'));
let fails = 0; const ok = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) fails++; };
// 1. colour vs skimage
const cv = JSON.parse(fs.readFileSync(here + 'colour_vectors.json'));
let mL = 0, mE = 0;
cv.a.forEach((a, i) => { const l = rgbToLab(...a); mL = Math.max(mL, ...l.map((v, k) => Math.abs(v - cv.La[i][k])));
  mE = Math.max(mE, Math.abs(ciede2000(cv.La[i], cv.Lb[i]) - cv.de[i])); });
ok(mL < 1e-6, `sRGB->Lab vs skimage, 300 colours: max |diff| ${mL.toExponential(2)}`);
ok(mE < 1e-6, `CIEDE2000 vs skimage, 300 pairs: max |diff| ${mE.toExponential(2)}`);
// Sharma et al. 2005 test pairs 1, 7, 17, 25
const sh = [[[50, 2.6772, -79.7751], [50, 0, -82.7485], 2.0425], [[50, 0, 0], [50, -1, 2], 2.3669], [[50, 2.5, 0], [73, 25, -18], 27.1492], [[60.2574, -34.0099, 36.2677], [60.4626, -34.1751, 39.4387], 1.2644]];
ok(sh.every(([p, q, d]) => Math.abs(ciede2000(p, q) - d) < 1e-4), 'CIEDE2000 matches 4 published Sharma et al. (2005) test pairs to 1e-4');
// 2. features + model vs Python
const pv = JSON.parse(fs.readFileSync(here + 'parity_vectors.json'));
let mf = 0, mp = 0;
pv.forEach(v => { const x = features(v.drop, v.bg); mf = Math.max(mf, ...x.map((q, k) => Math.abs(q - v.feats[k]))); mp = Math.max(mp, Math.abs(predictForest(model, x) - v.pred)); });
ok(mf < 1e-6, `features vs Python, 40 frames: max |diff| ${mf.toExponential(2)}`);
ok(mp < 1e-4, `RF prediction vs Python, 40 frames: max |diff| ${mp.toExponential(2)} log10 units`);
// 3. ROI on synthetic photos (known truth)
function synth({ w = 900, h = 700, cx, cy, r, drop, film = [205, 200, 195], noise = 3, grad = 0, glare = false, sat = false, seed = 1 }) {
  let s = seed; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const px = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const d = Math.hypot(x - cx, y - cy), t = Math.min(1, Math.max(0, (r - d) / 1.5 + 0.5));
    const g = 1 + grad * (x / w - 0.5);
    let c = [0, 1, 2].map(k => (film[k] * (1 - t) + drop[k] * t) * g + (rnd() - 0.5) * 2 * noise);
    if (glare && Math.hypot(x - cx + 0.3 * r, y - cy + 0.3 * r) < 0.12 * r) c = [252, 252, 252];
    if (sat && Math.hypot(x - cx - 2.6 * r, y - cy) < 0.3 * r) c = c.map(v => v * 0.6);
    const j = 4 * (y * w + x); px[j] = c[0]; px[j + 1] = c[1]; px[j + 2] = c[2]; px[j + 3] = 255;
  }
  return px;
}
const cases = [
  { name: 'dark drop', cx: 400, cy: 330, r: 45, drop: [40, 35, 45] },
  { name: 'mid drop', cx: 500, cy: 300, r: 60, drop: [120, 100, 110] },
  { name: 'faint drop (low GA)', cx: 420, cy: 360, r: 40, drop: [185, 180, 172] },
  { name: 'dark drop + glare + light gradient', cx: 380, cy: 340, r: 50, drop: [50, 45, 55], glare: true, grad: 0.15 },
  { name: 'dark drop + satellite droplet near', cx: 450, cy: 350, r: 45, drop: [45, 40, 50], sat: true },
];
for (const cs of cases) {
  const w = 900, h = 700, px = synth(cs);
  const res = analyse(px, w, h, cs.cx + 0.2 * cs.r, cs.cy - 0.1 * cs.r, { edge: [cs.cx - 1.2 * cs.r, cs.cy] });   // centre tap off by 0.2 R, edge tap 20% too far
  if (!res.ok) { ok(false, cs.name + ': ' + res.error); continue; }
  const dc = Math.hypot(res.circle.cx - cs.cx, res.circle.cy - cs.cy), dr = res.circle.r / cs.r - 1;
  const dd = Math.max(...res.drop.map((v, k) => Math.abs(v - cs.drop[k] * (1 + (cs.grad || 0) * (cs.cx / w - 0.5)))));
  ok(dc < 2 && Math.abs(dr) < 0.05 && dd < 4, `ROI ${cs.name}: centre error ${dc.toFixed(2)} px, radius error ${(100 * dr).toFixed(1)}%, drop colour error ${dd.toFixed(1)} counts, warnings: ${res.warnings.length ? res.warnings.join(' | ') : 'none'}`);
}
console.log(fails ? `${fails} FAILED` : 'ALL PASSED');
process.exit(fails ? 1 : 0);
