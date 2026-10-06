// app.js - user interface. All computation is in core.js and runs on the device.
import { VERSION, analyse, predictForest, interpret } from './core.js';

const $ = id => document.getElementById(id);
const MAXDIM = 1600;           // photos are scaled so the longer side is at most 1600 px
let model = null, img = null, centre = null;  // img: {data, w, h, src}; centre: first tap

$('ver').textContent = VERSION; $('ver2').textContent = VERSION;
fetch('model.json').then(r => r.json()).then(m => { model = m; $('mver').textContent = `${m.version} (${m.trained_on})`; });

document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('on', t === b));
  $('view-test').hidden = b.dataset.view !== 'test'; $('view-about').hidden = b.dataset.view !== 'about';
}));

// 15 s timer
let tid = null;
$('timerBtn').addEventListener('click', () => {
  clearInterval(tid); let t = 0; const el = $('timer');
  const tick = () => { t++; el.textContent = t < 10 ? `${t} s, wait` : t <= 15 ? `${t} s, take the photo now` : `${t} s, too late, repeat with a fresh drop`; if (t > 20) clearInterval(tid); };
  el.textContent = '0 s'; tid = setInterval(tick, 1000);
});

$('file').addEventListener('change', async e => {
  const f = e.target.files[0]; if (!f) return;
  let bmp;
  try { bmp = await createImageBitmap(f, { imageOrientation: 'from-image' }); }
  catch { bmp = await createImageBitmap(f); }
  const s = Math.min(1, MAXDIM / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * s), h = Math.round(bmp.height * s);
  const oc = document.createElement('canvas'); oc.width = w; oc.height = h;
  const ox = oc.getContext('2d', { willReadFrequently: true, colorSpace: 'srgb' });
  ox.drawImage(bmp, 0, 0, w, h);
  img = { data: ox.getImageData(0, 0, w, h).data, w, h, src: oc };
  const cv = $('cv'); cv.width = w; cv.height = h; cv.getContext('2d').drawImage(oc, 0, 0);
  $('stage').hidden = false; $('result').hidden = true; centre = null; $('hint').textContent = 'Step 1 of 2: tap the centre of the drop.';
  e.target.value = '';
});

$('cv').addEventListener('click', ev => {
  if (!img || !model) return;
  const cv = $('cv'), rect = cv.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * img.w / rect.width, y = (ev.clientY - rect.top) * img.h / rect.height;
  const ctx = cv.getContext('2d'); ctx.drawImage(img.src, 0, 0);
  const mk = Math.max(6, img.w / 120);
  if (!centre) {               // first tap: centre
    centre = [x, y]; ctx.strokeStyle = '#ffab00'; ctx.lineWidth = Math.max(2, img.w / 400);
    ctx.beginPath(); ctx.moveTo(x - mk, y); ctx.lineTo(x + mk, y); ctx.moveTo(x, y - mk); ctx.lineTo(x, y + mk); ctx.stroke();
    $('hint').textContent = 'Step 2 of 2: tap the edge of the drop.'; return;
  }
  const res = analyse(img.data, img.w, img.h, centre[0], centre[1], { edge: [x, y] });
  centre = null;
  if (!res.ok) { showError(res.error); $('hint').textContent = 'Step 1 of 2: tap the centre of the drop.'; return; }
  const { cx, cy, r } = res.circle, lw = Math.max(2, img.w / 400);
  ctx.lineWidth = lw;
  ctx.strokeStyle = '#00c853'; ctx.beginPath(); ctx.arc(cx, cy, 0.70 * r, 0, 2 * Math.PI); ctx.stroke();
  ctx.strokeStyle = '#2979ff'; [1.6, 2.3].forEach(k => { ctx.beginPath(); ctx.arc(cx, cy, k * r, 0, 2 * Math.PI); ctx.stroke(); });
  ctx.setLineDash([lw * 3, lw * 3]); ctx.strokeStyle = '#ffab00'; ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
  $('hint').textContent = 'Green: area measured. Blue: film reference ring. To redo, tap the centre and then the edge again.';
  const lp = predictForest(model, res.x), it = interpret(model, lp);
  $('band').textContent = it.band.label;
  const fmt = v => v >= 100 ? Math.round(v / 10) * 10 : Math.round(v);
  $('est').textContent = it.showEstimate ? `Estimate about ${fmt(it.estimate_mgL)} mg/L (likely between ${fmt(it.range80_mgL[0])} and ${fmt(it.range80_mgL[1])} mg/L)` : '';
  $('detail').textContent = it.band.detail;
  $('warn').innerHTML = ''; res.warnings.forEach(w => { const li = document.createElement('li'); li.textContent = w; $('warn').appendChild(li); });
  const rows = [['Drop median RGB', res.drop.map(Math.round).join(', ')], ['Film median RGB', res.bg.map(Math.round).join(', ')],
    ['A_R, A_G, A_B, A_grey', res.x.slice(0, 4).map(v => v.toFixed(3)).join(', ')], ['CIEDE2000 drop vs film', res.x[4].toFixed(2)],
    ['Model output (log10 mg/L)', lp.toFixed(3)], ['Drop radius (px)', r.toFixed(1)], ['Pixels used, drop / film', `${res.nDrop} / ${res.nRing}`],
    ['Edge points fitted', `${Math.round(100 * res.inlierFrac)}%`], ['Glare removed', `${Math.round(100 * res.hlFrac)}%`],
    ['Light unevenness around drop (grey levels)', res.ringUnevenness.toFixed(1)], ['App / model version', `${VERSION} / ${model.version}`]];
  $('tech').innerHTML = rows.map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join('');
  $('result').hidden = false; $('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
});

function showError(msg) {
  $('band').textContent = 'No result'; $('est').textContent = ''; $('detail').textContent = msg;
  $('warn').innerHTML = ''; $('tech').innerHTML = ''; $('result').hidden = false;
}

if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
