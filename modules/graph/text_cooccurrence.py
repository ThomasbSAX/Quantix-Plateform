from __future__ import annotations

import re
from collections import Counter
from typing import List, Sequence

import networkx as nx


_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def build_cooccurrence_graph(
    texts: Sequence[str],
    *,
    window: int = 3,
    min_node_count: int = 2,
    min_edge_count: int = 2,
    weight_attr: str = "weight",
) -> nx.Graph:
    """Graphe de cooccurrence de mots (simple, sans dépendances externes).

    - Noeuds: mots (filtrés par fréquence globale >= min_node_count)
    - Arêtes: cooccurrences dans une fenêtre glissante de taille `window` (>= min_edge_count)
    - Poids: nombre de cooccurrences
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    if not texts:
        raise ValueError("texts is empty")

    tokens_per_doc = [_tokenize(t) for t in texts]
    counts = Counter(t for toks in tokens_per_doc for t in toks)
    vocab = {t for t, c in counts.items() if c >= min_node_count}

    edge_counts = Counter()
    for toks in tokens_per_doc:
        toks = [t for t in toks if t in vocab]
        for i in range(len(toks)):
            w = toks[i]
            jmax = min(len(toks), i + window)
            for j in range(i + 1, jmax):
                u = toks[j]
                if u == w:
                    continue
                a, b = (w, u) if w < u else (u, w)
                edge_counts[(a, b)] += 1

    G = nx.Graph()
    for t in vocab:
        G.add_node(t, count=int(counts[t]))

    for (u, v), c in edge_counts.items():
        if c >= min_edge_count:
            G.add_edge(u, v, **{weight_attr: int(c)})

    return G
