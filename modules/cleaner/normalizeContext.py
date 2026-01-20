import re
from typing import Optional


def normalize_context(
    s: Optional[str],
    *,
    preserve_latex: bool = True,
    latex_token: str = "<latex>",
    replace_urls_with_token: bool = True,
    replace_emails_with_token: bool = True
) -> Optional[str]:
    """
    Nettoyage contextuel avancé avec PRÉSERVATION des formules.

    - conserve les formules LaTeX ($...$, $$...$$, \\(...\\), \\[...\\])
      soit verbatim, soit sous forme de token
    - supprime balises HTML
    - nettoie commandes LaTeX non mathématiques (\\textbf, etc.)
    - conserve URLs / emails (option : tokenisation)

    Objectif : ne jamais perdre l'information mathématique.
    """
    if s is None:
        return None
    s = str(s)

    # ============================================================
    # 1) Extraction des formules LaTeX
    # ============================================================
    formulas = []

    def _stash(m):
        formulas.append(m.group(0))
        return f"__LATEX_{len(formulas)-1}__"

    latex_patterns = [
        r"\$\$.*?\$\$",          # $$ ... $$
        r"\$.*?\$",              # $ ... $
        r"\\\(.+?\\\)",          # \( ... \)
        r"\\\[.+?\\\]",          # \[ ... \]
    ]

    if preserve_latex:
        for pat in latex_patterns:
            s = re.sub(pat, _stash, s, flags=re.S)

    # ============================================================
    # 2) Suppression balises HTML
    # ============================================================
    s = re.sub(r"<[^>]+>", " ", s)

    # ============================================================
    # 3) Nettoyage LaTeX non mathématique
    # ============================================================
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)

    # ============================================================
    # 4) URLs / emails (conservés ou tokenisés)
    # ============================================================
    if replace_urls_with_token:
        s = re.sub(r"https?://\S+|www\.\S+", "<url>", s)

    if replace_emails_with_token:
        s = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "<email>", s)

    # ============================================================
    # 5) Réinjection des formules
    # ============================================================
    if preserve_latex:
        for i, f in enumerate(formulas):
            replacement = latex_token if latex_token else f
            s = s.replace(f"__LATEX_{i}__", replacement)

    # ============================================================
    # 6) Espaces propres
    # ============================================================
    s = re.sub(r"\s+", " ", s).strip()
    return s
