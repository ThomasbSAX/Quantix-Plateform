from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import unquote

import networkx as nx


_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_HREF_RE = re.compile(r"href=\"([^\"]+)\"", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s)>\"]+", re.IGNORECASE)

# DOI: ex 10.1145/3377811.3380369
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)

# arXiv: arXiv:2301.01234v2
_ARXIV_RE = re.compile(r"\barxiv\s*:\s*(\d{4}\.\d{4,5})(v\d+)?\b", re.IGNORECASE)

# LaTeX/BibTeX: \cite{key1,key2}
_LATEX_CITE_RE = re.compile(r"\\cite[a-zA-Z*]*\{([^}]+)\}")

# Pandoc: [@key1; @key2]
_PANDOC_CITE_RE = re.compile(r"\[@([^\]]+)\]")


def _strip_trailing_punct(s: str) -> str:
    return s.rstrip("\t\r\n .,;:!?)]}\"")


def _normalize_ref(ref: str) -> str:
    ref = ref.strip().strip("\"'")
    ref = ref.replace("\\", "/")
    ref = unquote(ref)
    # Retirer ancres
    ref = ref.split("#", 1)[0]
    # Retirer query string
    ref = ref.split("?", 1)[0]
    ref = _strip_trailing_punct(ref)
    return ref


def _extract_refs(text: str, *, modes: Sequence[str]) -> List[str]:
    refs: List[str] = []
    t = text or ""

    if "markdown_html" in modes:
        for m in _MD_LINK_RE.findall(t):
            refs.append(_normalize_ref(m))
        for m in _HREF_RE.findall(t):
            refs.append(_normalize_ref(m))

    if "raw_url" in modes:
        for m in _URL_RE.findall(t):
            refs.append(_normalize_ref(m))

    if "doi" in modes:
        for m in _DOI_RE.findall(t):
            doi = _normalize_ref(m)
            refs.append(f"doi:{doi}")

    if "arxiv" in modes:
        for mid, ver in _ARXIV_RE.findall(t):
            aid = f"{mid}{ver or ''}"
            refs.append(f"arxiv:{aid}")

    if "latex_cite" in modes:
        for block in _LATEX_CITE_RE.findall(t):
            keys = [k.strip() for k in block.split(",") if k.strip()]
            for k in keys:
                refs.append(f"cite:{k}")

    if "pandoc_cite" in modes:
        for block in _PANDOC_CITE_RE.findall(t):
            # exemple: "@a; @b, p. 1"
            parts = re.split(r"[;,]", block)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if p.startswith("@"):  # @key
                    key = p[1:].strip()
                    if key:
                        refs.append(f"cite:{key}")

    return [r for r in refs if r]


def _make_external_id(ref: str) -> str:
    # Ref: doi:xxx / arxiv:xxx / cite:xxx / url/path
    if ref.startswith("doi:"):
        return f"external:doi:{ref[4:]}"
    if ref.startswith("arxiv:"):
        return f"external:arxiv:{ref[6:]}"
    if ref.startswith("cite:"):
        return f"external:cite:{ref[5:]}"
    if ref.startswith("http://") or ref.startswith("https://"):
        return f"external:{ref}"
    # chemins relatifs et autres identifiants
    return f"external:{ref}"


def _basename_variants(doc_id: str) -> List[str]:
    base = os.path.basename(doc_id)
    base = unquote(base)
    stem = os.path.splitext(base)[0]
    out = []
    for s in {base, stem}:
        s = s.strip().lower()
        if s and len(s) >= 3:
            out.append(s)
    # version alnum seulement
    def _alnum(x: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()

    for s in {base, stem}:
        a = _alnum(s)
        if a and len(a) >= 6:
            out.append(a)
    # dédupe
    return list(dict.fromkeys(out))


def build_document_citation_graph(
    documents: Mapping[str, str],
    *,
    directed: bool = True,
    weight_attr: str = "weight",
    match: str = "both",  # basename | path | both
    allow_external: bool = False,
    detect_mentions: bool = True,
    mention_min_len: int = 6,
    modes: Optional[Sequence[str]] = None,
) -> nx.DiGraph:
    """Construit un graphe de citations/liens entre documents.

    Heuristiques supportées (sans NLP):
    - Liens Markdown: [texte](cible)
    - Liens HTML: href="cible"
    - Matching par basename et/ou par path exact.

    Poids = nombre de fois où un doc cite l'autre.
    """
    if match not in {"basename", "path", "both"}:
        raise ValueError("match must be one of: basename, path, both")
    if mention_min_len < 1:
        raise ValueError("mention_min_len must be >= 1")

    if modes is None:
        # Modes courants (sans dépendances):
        # - markdown_html: liens [txt](url) et href="url"
        # - raw_url: URLs brutes dans le texte
        # - doi/arxiv: identifiants académiques
        # - latex_cite/pandoc_cite: citations bibliographiques
        modes = ["markdown_html", "raw_url", "doi", "arxiv", "latex_cite", "pandoc_cite"]

    G = nx.DiGraph() if directed else nx.Graph()

    # Index des documents
    ids = [str(k) for k in documents.keys()]
    for doc_id in ids:
        G.add_node(doc_id)

    by_path: Dict[str, str] = {}
    by_base: Dict[str, List[str]] = {}
    for doc_id in ids:
        by_path[doc_id] = doc_id
        base = os.path.basename(doc_id)
        by_base.setdefault(base, []).append(doc_id)

    # Pré-calcul des variantes (basename/stem) pour mention matching doc->doc.
    mention_variants: Dict[str, List[str]] = {}
    if detect_mentions:
        for doc_id in ids:
            vars_ = _basename_variants(doc_id)
            # Éviter de matcher des choses trop courtes/génériques.
            vars_ = [v for v in vars_ if len(v) >= mention_min_len]
            if vars_:
                mention_variants[doc_id] = vars_

    for src_id, text in documents.items():
        if not text:
            continue
        src_id = str(src_id)
        text_str = str(text)
        refs = _extract_refs(text_str, modes=modes)

        counts = Counter(refs)
        for ref, c in counts.items():
            targets: List[str] = []

            # DOI/arXiv/cite: toujours external
            if ref.startswith(("doi:", "arxiv:", "cite:")):
                if allow_external:
                    targets = [_make_external_id(ref)]
                else:
                    continue

            # matching path exact
            if match in {"path", "both"}:
                if ref in by_path:
                    targets.append(by_path[ref])
                # cas zip pseudo-path: on autorise matching suffixe (dossier relatif)
                if not targets:
                    suffix_matches = [doc for doc in ids if doc.endswith(ref)]
                    targets.extend(suffix_matches)

            # si ref est une URL, tenter de matcher par basename du chemin
            if not targets and (ref.startswith("http://") or ref.startswith("https://")):
                base = os.path.basename(ref)
                if base:
                    targets.extend(by_base.get(base, []))

            # matching basename
            if match in {"basename", "both"}:
                base = os.path.basename(ref)
                targets.extend(by_base.get(base, []))

            # dedupe
            targets = list(dict.fromkeys(t for t in targets if t))

            if not targets:
                if allow_external:
                    # Créer un noeud externe
                    ext_id = _make_external_id(ref)
                    if ext_id not in G:
                        G.add_node(ext_id, external=True)
                    targets = [ext_id]
                else:
                    continue

            for dst_id in targets:
                if src_id == dst_id:
                    continue
                if G.has_edge(src_id, dst_id):
                    G[src_id][dst_id][weight_attr] = float(G[src_id][dst_id].get(weight_attr, 0.0)) + float(c)
                else:
                    G.add_edge(src_id, dst_id, **{weight_attr: float(c)})

        # Détection "mentions": basenames/stems d'autres docs dans le texte.
        # Utile pour du texte extrait de PDF où les liens ne sont pas conservés.
        if detect_mentions and mention_variants:
            lowered = text_str.lower()
            for dst_id, variants in mention_variants.items():
                if dst_id == src_id:
                    continue
                hit = False
                for v in variants:
                    if v in lowered:
                        hit = True
                        break
                if hit:
                    if G.has_edge(src_id, dst_id):
                        G[src_id][dst_id][weight_attr] = float(G[src_id][dst_id].get(weight_attr, 0.0)) + 1.0
                    else:
                        G.add_edge(src_id, dst_id, **{weight_attr: 1.0})

    return G
