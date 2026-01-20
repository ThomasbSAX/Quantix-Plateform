from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import networkx as nx


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


def _as_2d_float_array(x: ArrayLike) -> np.ndarray:
    X = np.asarray(x, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"embeddings must be 2D (n,d), got shape={X.shape}")
    if X.shape[0] < 2:
        raise ValueError("need at least 2 nodes")
    if not np.isfinite(X).all():
        raise ValueError("embeddings contain NaN/Inf")
    return X


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def build_cosine_similarity_graph(
    embeddings: ArrayLike,
    *,
    labels: Optional[Sequence[str]] = None,
    k: Optional[int] = 10,
    threshold: Optional[float] = None,
    directed: bool = False,
    self_loops: bool = False,
    add_node_attrs: Optional[Mapping[str, Sequence[Any]]] = None,
    weight_attr: str = "weight",
) -> nx.Graph:
    """
    Construit un graphe de similarité à partir d'embeddings.
    Arête (i,j) si:
      - mode k-NN: j est dans les k plus proches voisins cosinus de i (k>0), et/ou
      - mode seuil: cos(i,j) >= threshold.
    Le poids de l'arête = similarité cosinus (dans [-1,1]).

    Règle: si k et threshold sont tous deux fournis, on prend l'union des arêtes.
    """
    X = _l2_normalize_rows(_as_2d_float_array(embeddings))
    n = X.shape[0]

    if labels is None:
        labels = [str(i) for i in range(n)]
    if len(labels) != n:
        raise ValueError(f"labels length={len(labels)} must match n={n}")

    if k is not None:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive int or None")
        if k >= n:
            k = n - 1

    if threshold is not None and not (-1.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be in [-1,1] or None")

    G: nx.Graph = nx.DiGraph() if directed else nx.Graph()

    for i, lab in enumerate(labels):
        G.add_node(lab)

    if add_node_attrs:
        for attr_name, values in add_node_attrs.items():
            if len(values) != n:
                raise ValueError(f"node attr '{attr_name}' length must be n={n}")
            for i, lab in enumerate(labels):
                G.nodes[lab][attr_name] = values[i]

    S = X @ X.T  # cosine similarities
    if not self_loops:
        np.fill_diagonal(S, -np.inf)

    edges = set()

    if k is not None:
        for i in range(n):
            idx = np.argpartition(-S[i], kth=k - 1)[:k]
            for j in idx:
                if not self_loops and i == j:
                    continue
                edges.add((i, int(j)))

    if threshold is not None:
        ii, jj = np.where(S >= threshold)
        for i, j in zip(ii.tolist(), jj.tolist()):
            if not self_loops and i == j:
                continue
            edges.add((int(i), int(j)))

    for i, j in edges:
        u, v = labels[i], labels[j]
        w = float(S[i, j])
        if directed:
            G.add_edge(u, v, **{weight_attr: w})
        else:
            if G.has_edge(u, v):
                G[u][v][weight_attr] = max(G[u][v][weight_attr], w)
            else:
                G.add_edge(u, v, **{weight_attr: w})

    return G


# Compat: l'ancien `graph_builder.py` importait `build_cosine_graph`.
def build_cosine_graph(
    embeddings: ArrayLike,
    **kwargs: Any,
) -> nx.Graph:
    return build_cosine_similarity_graph(embeddings, **kwargs)

