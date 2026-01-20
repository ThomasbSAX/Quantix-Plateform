/*
  Confidentialité Quantix — purge session/datasets
  Objectif: ne jamais conserver les données utilisateur entre pages.
*/

(function () {
  const CLEAR_URL = '/api/session/clear';

  function clearSession() {
    try {
      if (navigator.sendBeacon) {
        const blob = new Blob(['{}'], { type: 'application/json' });
        navigator.sendBeacon(CLEAR_URL, blob);
        return;
      }
    } catch (_) {}

    try {
      fetch(CLEAR_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
        keepalive: true,
      }).catch(() => {});
    } catch (_) {}
  }

  // À la fermeture/refresh/navigation
  window.addEventListener('pagehide', clearSession);
  window.addEventListener('beforeunload', clearSession);

  // Au clic sur un lien interne
  document.addEventListener(
    'click',
    (e) => {
      const a = e.target && e.target.closest ? e.target.closest('a') : null;
      if (!a) return;
      try {
        const url = new URL(a.href);
        if (url.origin === window.location.origin) clearSession();
      } catch (_) {
        // ignore
      }
    },
    true
  );
})();
