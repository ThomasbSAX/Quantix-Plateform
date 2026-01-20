from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple
import numpy as np
import networkx as nx


def build_knn_graph(
    X: Sequence[Sequence[float]],
    *,
    labels: Optional[Sequence[Any]] = None,
    k: int = 10,
    metric: str = "euclidean",
    directed: bool = False,
    self_loops: bool = False,
    weight_attr: str = "weight",
    node_attrs: Optional[Mapping[str, Sequence[Any]]] = None,
) -> nx.Graph:
    """
    Construit un graphe k-plus-proches-voisins.
    Arête (i,j) si j est dans les k plus proches voisins de i selon la métrique.
    Poids = distance (ou similarité négative si métrique=cosine).

    metric ∈ {"euclidean", "cosine"}.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must be 2D with at least 2 rows")

    n = X.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels length mismatch")

    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive int")
    if k >= n:
        k = n - 1

    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine'")

    G: nx.Graph = nx.DiGraph() if directed else nx.Graph()

    for lab in labels:
        G.add_node(lab)

    if node_attrs:
        for name, values in node_attrs.items():
            if len(values) != n:
                raise ValueError(f"node attr '{name}' length mismatch")
            for i, lab in enumerate(labels):
                G.nodes[lab][name] = values[i]

    if metric == "euclidean":
        # distances pairwise
        diff = X[:, None, :] - X[None, :, :]
        D = np.linalg.norm(diff, axis=2)
        if not self_loops:
            np.fill_diagonal(D, np.inf)
        score = D  # smaller is better
    else:
        # cosine distance = 1 - cosine similarity
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = X / np.maximum(norms, 1e-12)
        S = Xn @ Xn.T
        if not self_loops:
            np.fill_diagonal(S, -np.inf)
        score = 1.0 - S  # smaller is better

    for i in range(n):
        idx = np.argpartition(score[i], kth=k - 1)[:k]
        for j in idx:
            if not self_loops and i == j:
                continue
            u, v = labels[i], labels[j]
            w = float(score[i, j])
            if directed:
                G.add_edge(u, v, **{weight_attr: w})
            else:
                if G.has_edge(u, v):
                    G[u][v][weight_attr] = min(G[u][v][weight_attr], w)
                else:
                    G.add_edge(u, v, **{weight_attr: w})

    return G
