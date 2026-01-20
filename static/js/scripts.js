/**
 * QUANTIX - JavaScript Central
 * Gestion header/footer + effets généraux + pages
 */

/* =========================
   HEADER / FOOTER
   ========================= */
function loadHeader(pageName = null) {
  return fetch("header.html")
    .then(res => res.text())
    .then(html => {
      document.getElementById("header-placeholder").innerHTML = html;

      // Marquer lien actif
      if (pageName) {
        document.querySelectorAll(".nav-link, .btn-get-started").forEach(link => {
          const isActive = link.dataset.page === pageName;
          link.classList.toggle("active", isActive);
        });
      }
    })
    .catch(err => {
      console.error("Erreur chargement header:", err);
    });
}

function loadFooter() {
  return fetch("footer.html")
    .then(res => res.text())
    .then(html => {
      document.getElementById("footer-placeholder").innerHTML = html;
      const year = document.getElementById("year");
      if (year) year.textContent = new Date().getFullYear();
    })
    .catch(err => {
      console.error("Erreur chargement footer:", err);
    });
}

/* =========================
   UTILITAIRES
   ========================= */
const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));

/* =========================
   EFFETS GÉNÉRAUX
   ========================= */
function initSmoothScrolling() {
  $$('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener("click", e => {
      e.preventDefault();
      const href = anchor.getAttribute("href");
      if (!href || href === "#") return;
      let target = null;
      try {
        target = $(href);
      } catch (_) {
        target = null;
      }
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
}

function initFadeInAnimations() {
  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.transition = "opacity 1s ease, transform 1s ease";
        entry.target.style.opacity = 1;
        entry.target.style.transform = "translateY(0)";
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.2 });

  $$(".fade-in").forEach(el => {
    el.style.opacity = 0;
    el.style.transform = "translateY(20px)";
    io.observe(el);
  });
}

/* =========================
   UPLOAD FICHIERS
   ========================= */
function initFileUpload() {
  const fileInput = $("#fileInput");
  const fileSelect = $("#fileSelectBtn");
  const uploadZone = $("#uploadZone");
  const fileListDiv = $("#fileList");
  const uploadForm = $("#uploadForm");

  if (!fileInput || !fileSelect || !uploadZone || !fileListDiv || !uploadForm) return;

  fileSelect.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", updateFileList);

  function updateFileList() {
    if (fileInput.files.length === 0) {
      fileListDiv.innerHTML = '<p style="color:var(--muted);font-size:0.9rem;text-align:center">Aucun fichier sélectionné</p>';
      return;
    }
    const items = [...fileInput.files].map(
      f => `<li style="margin:6px 0;padding:8px 12px;background:rgba(255,255,255,0.05);border-radius:8px">
              <i class="fas fa-file" style="margin-right:8px;color:var(--accent)"></i>
              ${f.name} <span style="color:var(--muted);font-size:0.8rem">(${(f.size / 1024).toFixed(1)} Ko)</span>
            </li>`
    ).join("");
    fileListDiv.innerHTML = `<ul style="list-style:none;padding:0;margin:0">${items}</ul>`;
  }

  uploadZone.addEventListener("dragover", e => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });
  uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
  uploadZone.addEventListener("drop", e => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      updateFileList(e.dataTransfer.files);
    }
  });

  uploadForm.addEventListener("submit", e => {
    e.preventDefault();
    if (fileInput.files.length === 0) {
      alert("Aucun fichier sélectionné.");
      return;
    }

    const formData = new FormData(uploadForm);
    fetch("/upload", { method: "POST", body: formData })
      .then(res => res.json())
      .then(data => alert(`${data.message}\n${data.path || "Fichier stocké"}`))
      .catch(err => alert("Erreur : " + err));
  });
}

/* =========================
   INIT PAR PAGE
   ========================= */
function initBasePage(pageName = null) {
  Promise.all([loadHeader(pageName), loadFooter()]).then(() => {
    initSmoothScrolling();
  });
}

function initIndexPage() {
  initBasePage("index");
}

function initFonctionnalitesPage() {
  initBasePage("fonctionnalites");
}

function initIaPage() {
  initBasePage("ia");
  initFadeInAnimations();
}

function initEssayerPage() {
  initBasePage("essayer");
  initFileUpload();
}

function initHeaderPage() {
  // Initialisation pour les pages avec header personnalisé
  console.log("Header page initialized");
}

function initDocsPage() {
  initBasePage("docs");
  // Smooth scrolling pour la page docs
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if(target) target.scrollIntoView({behavior:'smooth',block:'start'});
    });
  });
}

function initIaPageAnimations() {
  // Animation pour les cartes de fonctionnalités
  document.addEventListener("DOMContentLoaded", () => {
    // on active un mode JS dans le <html> (utile pour fallback CSS)
    document.documentElement.classList.add("js");

    // observer chaque carte
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
        }
      });
    }, { threshold: 0.2 });

    document.querySelectorAll(".feature-card").forEach(card => {
      observer.observe(card);
    });
  });
}
