/* ============================================================
   NAVIGATION GLOBALE QUANTIX — version stylée
   ============================================================ */

const pageOrder = ["index", "fonctionnalites","confidentialite", "essayer"];

function getCurrentPage() {
  // Récupère le nom de la page depuis l'URL Flask (ex: /fonctionnalites -> fonctionnalites)
  const path = window.location.pathname;
  if (path === "/" || path === "") return "index";
  return path.substring(1); // Enlève le "/" du début
}

function markActiveLink() {
  const current = getCurrentPage();
  document.querySelectorAll(".nav-link, .btn-get-started").forEach(link => {
    const isActive = link.dataset.page === current;
    link.classList.toggle("active", isActive);
    if (isActive) link.classList.add("nav-pulse");
  });
}

/* --- Effet de feedback visuel --- */
function showNavigationFeedback(direction) {
  const feedback = document.createElement("div");
  feedback.className = `nav-feedback ${direction}`;
  feedback.innerHTML = direction === "next"
    ? "<i class='fa-solid fa-arrow-right'></i>"
    : "<i class='fa-solid fa-arrow-left'></i>";
  document.body.appendChild(feedback);
  requestAnimationFrame(() => feedback.classList.add("show"));
  setTimeout(() => feedback.classList.remove("show"), 450);
  setTimeout(() => feedback.remove(), 900);
}

/* --- Effet de transition de page --- */
function applyPageTransition() {
  const overlay = document.createElement("div");
  overlay.className = "page-transition";
  document.body.appendChild(overlay);
  requestAnimationFrame(() => overlay.classList.add("fade-in"));
  return new Promise(res => setTimeout(() => {
    overlay.classList.add("fade-out");
    setTimeout(() => { overlay.remove(); res(); }, 400);
  }, 200));
}

/* --- Navigation circulaire --- */
async function navigate(offset) {
  const idx = pageOrder.indexOf(getCurrentPage());
  if (idx === -1) return;
  const newIdx = (idx + offset + pageOrder.length) % pageOrder.length;

  showNavigationFeedback(offset > 0 ? "next" : "prev");
  await applyPageTransition();

  // Navigation vers les routes Flask
  window.location.href = "/" + pageOrder[newIdx];
}

/* --- Initialisation --- */
document.addEventListener("DOMContentLoaded", () => {
  markActiveLink();

  // Navigation clavier
  window.addEventListener("keydown", e => {
    if (e.key === "ArrowRight") navigate(1);
    if (e.key === "ArrowLeft") navigate(-1);
  });

  // Navigation tactile
  let startX = 0;
  window.addEventListener("touchstart", e => startX = e.touches[0].clientX);
  window.addEventListener("touchend", e => {
    const dx = e.changedTouches[0].clientX - startX;
    if (dx > 50) navigate(-1);
    if (dx < -50) navigate(1);
  });
});
