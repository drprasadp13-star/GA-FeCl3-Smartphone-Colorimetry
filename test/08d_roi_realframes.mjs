// 08d_roi_realframes.mjs: runs the app's ROI + features + model (core.js) on one real frame (t = 12.0 s) of each
// 0.5% FeCl3 video, decoded by ffmpeg (BT.709 limited -> full range, auto-rotation), tapping 0.3 R off the Python
// ROI centre (one-tap mode) or with two taps (centre 0.2 R off, edge at 0.8, 1.0 or 1.25 R). Output: validation/js_realframes.csv for comparison with the Python ROI (02a) and Python features.
// Run locally next to the videos:  node webapp/test/08d_roi_realframes.mjs (writes validation/js_realframes.csv)
import fs from 'fs'; import { execFileSync } from 'child_process';
import { analyse, predictForest } from '../core.js';
const root = (process.env.GA_ROOT || '.') + '/';   // project folder with raw/ and outputs/ (set GA_ROOT)
const model = JSON.parse(fs.readFileSync(new URL('../model.json', import.meta.url)));
const csv = f => { const [h, ...r] = fs.readFileSync(f, 'utf8').trim().split('\n'); const H = h.split(','); return r.map(l => Object.fromEntries(l.split(',').map((v, i) => [H[i], v]))); };
const meta = Object.fromEntries(csv(root + 'outputs/00_audit/metadata_master.csv').map(r => [r.video_id, r]));
const roi = csv(root + 'outputs/02_roi/roi_log.csv').filter(r => r.video_id.startsWith('F0.5'));
const out = ['mode,video_id,py_cx,py_cy,py_Req,js_cx,js_cy,js_r,inlier,hl,nDrop,nRing,uneven,dR,dG,dB,bR,bG,bB,A_R,A_G,A_B,A_gray,dE00,pred_log10,n_warn,warnings'];
for (const r of roi) {
  const m = meta[r.video_id], vp = root + m.rel_path;
  const probe = execFileSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', vp]).toString().trim().split(',').map(Number);
  let [w, h] = probe; if (Math.abs(+m.rotation) === 90) [w, h] = [h, w];
  const buf = execFileSync('ffmpeg', ['-v', 'error', '-ss', '12.0', '-i', vp, '-frames:v', '1', '-vf', 'scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgba', '-f', 'rawvideo', '-'], { maxBuffer: 1 << 26 });
  if (buf.length !== w * h * 4) { out.push(`${r.video_id},size mismatch ${buf.length} vs ${w}x${h}`); continue; }
  const cx = +r.cx, cy = +r.cy, R = +r.R_eq;
  const px = new Uint8ClampedArray(buf);
  for (const [mode, k] of [['two_tap_0.8R', 0.8], ['two_tap_1.0R', 1.0], ['two_tap_1.25R', 1.25]]) {
  const tx = cx + 0.2 * R, ty = cy - 0.1 * R;
  const res = analyse(px, w, h, tx, ty, { edge: [tx - k * R, ty] });
  if (!res.ok) { out.push(`${mode},${r.video_id},${cx},${cy},${R},,,,,,,,,,,,,,,,,,,,,,FAIL`); continue; }
  const p = predictForest(model, res.x);
  out.push([mode, r.video_id, cx, cy, R, res.circle.cx.toFixed(2), res.circle.cy.toFixed(2), res.circle.r.toFixed(2), res.inlierFrac.toFixed(3), res.hlFrac.toFixed(3), res.nDrop, res.nRing, res.ringUnevenness.toFixed(2), ...res.drop, ...res.bg, ...res.x.map(v => v.toFixed(5)), p.toFixed(4), res.warnings.length, '"' + res.warnings.map(s => s.split('.')[0]).join(' | ') + '"'].join(','));
  }
  console.log(r.video_id);
}
fs.writeFileSync('validation/js_realframes.csv', out.join('\n') + '\n');
