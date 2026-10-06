// Offline cache. Bump CACHE when any file changes.
const CACHE = 'ga-photo-test-0.1.1';
const FILES = ['./', 'index.html', 'styles.css', 'app.js', 'core.js', 'model.json', 'manifest.webmanifest', 'icons/icon-192.png', 'icons/icon-512.png'];
self.addEventListener('install', e => e.waitUntil(caches.open(CACHE).then(c => c.addAll(FILES)).then(() => self.skipWaiting())));
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(k => Promise.all(k.filter(x => x !== CACHE).map(x => caches.delete(x)))).then(() => self.clients.claim())));
self.addEventListener('fetch', e => { if (e.request.method === 'GET') e.respondWith(caches.match(e.request).then(r => r || fetch(e.request))); });
