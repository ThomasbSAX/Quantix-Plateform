from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import networkx as nx


_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_SENT_RE = re.compile(r"[.!?]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _style_features(text: str) -> np.ndarray:
    """Features de style simples, sans dépendances externes.

    Idée: capturer des signaux de "stylographie" (ponctuation, longueur, richesse lexicale).
    """
    if not text:
        return np.zeros(14, dtype=float)

    n_chars = len(text)
    n_spaces = text.count(" ")
    n_newlines = text.count("\n")
    n_digits = sum(ch.isdigit() for ch in text)

    punct = {
        ",": text.count(","),
        ";": text.count(";"),
        ":": text.count(":"),
        "!": text.count("!"),
        "?": text.count("?"),
        "-": text.count("-"),
        "\"": text.count('"') + text.count("“") + text.count("”"),
        "('": text.count("(") + text.count(")"),
    }

    tokens = _tokenize(text)
    n_tokens = len(tokens)
    uniq = len(set(tokens))
    avg_word_len = _safe_div(sum(len(t) for t in tokens), n_tokens)

    # Approximations de phrases
    sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    n_sent = len(sentences)
    avg_sent_len = _safe_div(n_tokens, n_sent)

    # Ratio majuscules
    n_upper = sum(ch.isupper() for ch in text)

    feats = np.array(
        [
            math.log1p(n_chars),
            _safe_div(n_spaces, n_chars),
            _safe_div(n_newlines, n_chars),
            _safe_div(n_digits, n_chars),
            _safe_div(n_upper, n_chars),
            _safe_div(punct[","], n_chars),
            _safe_div(punct[";"], n_chars),
            _safe_div(punct[":"], n_chars),
            _safe_div(punct["!"], n_chars),
            _safe_div(punct["?"], n_chars),
            _safe_div(punct["-"], n_chars),
            _safe_div(punct["\""], n_chars),
            _safe_div(punct["('"], n_chars),
            _safe_div(uniq, n_tokens),  # type-token ratio
        ],
        dtype=float,
    )

    # Ajouter quelques ratios lexicales
    feats = np.concatenate(
        [
            feats,
            np.array([avg_word_len, avg_sent_len], dtype=float),
        ]
    )
    return feats


def _tfidf_matrix(texts: Sequence[str], *, min_df: int = 1, max_features: int = 20000) -> np.ndarray:
    """TF-IDF minimal (bag-of-words) pour similarité de contenu."""
    docs = [Counter(_tokenize(t)) for t in texts]
    df = Counter()
    for c in docs:
        for w in c.keys():
            df[w] += 1

    vocab = [w for w, d in df.items() if d >= min_df]
    vocab.sort(key=lambda w: (-df[w], w))
    if len(vocab) > max_features:
        vocab = vocab[:max_features]
    idx = {w: i for i, w in enumerate(vocab)}

    n = len(docs)
    m = len(vocab)
    if m == 0:
        return np.zeros((n, 0), dtype=float)

    # idf
    idf = np.zeros(m, dtype=float)
    for w, i in idx.items():
        idf[i] = math.log((1.0 + n) / (1.0 + df[w])) + 1.0

    X = np.zeros((n, m), dtype=float)
    for i, c in enumerate(docs):
        for w, tf in c.items():
            j = idx.get(w)
            if j is None:
                continue
            X[i, j] = float(tf)

    # tf-idf + l2 normalize
    X *= idf
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)
    return X


def _cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return np.zeros((X.shape[0], X.shape[0]), dtype=float)
    S = X @ X.T
    np.fill_diagonal(S, 1.0)
    return S


def build_document_similarity_graph(
    documents: Mapping[str, str],
    *,
    mode: str = "combined",  # "content" | "style" | "combined"
    k: Optional[int] = 5,
    threshold: Optional[float] = 0.25,
    directed: bool = False,
    self_loops: bool = False,
    weight_attr: str = "weight",
    min_chars: int = 30,
    content_min_df: int = 1,
    content_max_features: int = 20000,
    style_weight: float = 0.35,
) -> nx.Graph:
    """Construit un graphe de similarité entre documents.

    - Noeuds: documents (clé = id)
    - Poids: similarité cosinus (dans [0,1] approx.)
    - Arêtes: union de top-k voisins et/ou seuil

    `mode`:
      - content: TF-IDF minimal
      - style: features de style
      - combined: (1-style_weight)*content + style_weight*style
    """
    if mode not in {"content", "style", "combined"}:
        raise ValueError("mode must be one of: content, style, combined")
    if k is not None and (not isinstance(k, int) or k <= 0):
        raise ValueError("k must be a positive int or None")
    if threshold is not None and not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be in [0,1] or None")
    if min_chars < 0:
        raise ValueError("min_chars must be >= 0")

    # Filtrer documents trop petits
    ids: List[str] = []
    texts: List[str] = []
    for doc_id, text in documents.items():
        if text is None:
            continue
        text = str(text)
        if len(text.strip()) < min_chars:
            continue
        ids.append(str(doc_id))
        texts.append(text)

    n = len(ids)
    G: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    for doc_id in ids:
        G.add_node(doc_id)
    if n < 2:
        return G

    S = None
    if mode in {"content", "combined"}:
        Xc = _tfidf_matrix(texts, min_df=content_min_df, max_features=content_max_features)
        Sc = _cosine_sim_matrix(Xc)
        S = Sc

    if mode in {"style", "combined"}:
        Xs = np.stack([_style_features(t) for t in texts], axis=0)
        # Normaliser features style
        norms = np.linalg.norm(Xs, axis=1, keepdims=True)
        Xs = Xs / np.maximum(norms, 1e-12)
        Ss = _cosine_sim_matrix(Xs)
        S = Ss if S is None else ((1.0 - style_weight) * S + style_weight * Ss)

    assert S is not None
    if not self_loops:
        np.fill_diagonal(S, -np.inf)

    # construction edges
    edges = set()
    if k is not None:
        k_eff = min(k, n - 1)
        for i in range(n):
            idx = np.argpartition(-S[i], kth=k_eff - 1)[:k_eff]
            for j in idx:
                if not self_loops and i == j:
                    continue
                edges.add((i, int(j)))

    if threshold is not None:
        ii, jj = np.where(S >= float(threshold))
        for i, j in zip(ii.tolist(), jj.tolist()):
            if not self_loops and i == j:
                continue
            edges.add((int(i), int(j)))

    for i, j in edges:
        u, v = ids[i], ids[j]
        w = float(S[i, j])
        if directed:
            G.add_edge(u, v, **{weight_attr: w})
        else:
            if G.has_edge(u, v):
                G[u][v][weight_attr] = max(float(G[u][v].get(weight_attr, w)), w)
            else:
                G.add_edge(u, v, **{weight_attr: w})

    return G
