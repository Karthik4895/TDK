const CACHE = 'mt-v1';
const STATIC = [
  '/',
  '/static/manifest.json',
  '/static/icon-192.svg',
  '/static/icon-512.svg',
  'https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;900&family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap',
  'https://checkout.razorpay.com/v1/checkout.js',
  'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(STATIC)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  // Network-only for API calls and Firebase
  if (url.pathname.startsWith('/ask') ||
      url.pathname.startsWith('/orders') ||
      url.pathname.startsWith('/create-order') ||
      url.pathname.startsWith('/verify-payment') ||
      url.pathname.startsWith('/cod-order') ||
      url.pathname.startsWith('/returns') ||
      url.pathname.startsWith('/wishlist') ||
      url.pathname.startsWith('/reviews') ||
      url.pathname.startsWith('/loyalty') ||
      url.pathname.startsWith('/referral') ||
      url.pathname.startsWith('/coupons') ||
      url.pathname.startsWith('/contact') ||
      url.pathname.startsWith('/admin') ||
      url.pathname.startsWith('/history') ||
      url.hostname.includes('firebase') ||
      url.hostname.includes('googleapis.com') ||
      url.hostname.includes('razorpay') ||
      url.hostname.includes('resend') ||
      url.hostname.includes('tawk')) {
    e.respondWith(fetch(e.request));
    return;
  }

  // Cache-first for static assets (fonts, CDN scripts, icons)
  if (url.pathname.startsWith('/static/') ||
      url.hostname.includes('fonts.g') ||
      url.hostname.includes('cdn.jsdelivr') ||
      url.hostname.includes('gstatic.com') ||
      url.hostname.includes('ui-avatars.com')) {
    e.respondWith(
      caches.match(e.request).then(cached => {
        if (cached) return cached;
        return fetch(e.request).then(res => {
          const clone = res.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
          return res;
        });
      })
    );
    return;
  }

  // Network-first with cache fallback for HTML pages
  e.respondWith(
    fetch(e.request)
      .then(res => {
        const clone = res.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone));
        return res;
      })
      .catch(() => caches.match(e.request).then(cached => cached || caches.match('/')))
  );
});
