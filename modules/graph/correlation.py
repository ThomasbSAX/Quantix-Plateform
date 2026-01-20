from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence
import numpy as np
import networkx as nx


def build_correlation_graph(
    X: Sequence[Sequence[float]],
    *,
    labels: Optional[Sequence[Any]] = None,
    threshold: float = 0.5,
    method: str = "pearson",
    absolute: bool = True,
    directed: bool = False,
    self_loops: bool = False,
    weight_attr: str = "weight",
    node_attrs: Optional[Mapping[str, Sequence[Any]]] = None,
) -> nx.Graph:
    """
    Construit un graphe de corrélation entre variables.
    - X: matrice (n_samples, n_features)
    - Arête (i,j) si |corr(i,j)| >= threshold (ou corr(i,j) >= threshold si absolute=False)
    - method ∈ {"pearson", "spearman"}
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("X must be 2D with at least 2 features")

    n_feat = X.shape[1]
    if labels is None:
        labels = [str(i) for i in range(n_feat)]
    if len(labels) != n_feat:
        raise ValueError("labels length mismatch")

    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")

    if method == "pearson":
        C = np.corrcoef(X, rowvar=False)
    else:
        from scipy.stats import rankdata
        Xr = np.apply_along_axis(rankdata, 0, X)
        C = np.corrcoef(Xr, rowvar=False)

    if not self_loops:
        np.fill_diagonal(C, 0.0)

    G: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    for lab in labels:
        G.add_node(lab)

    if node_attrs:
        for name, values in node_attrs.items():
            if len(values) != n_feat:
                raise ValueError(f"node attr '{name}' length mismatch")
            for i, lab in enumerate(labels):
                G.nodes[lab][name] = values[i]

    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            w = float(C[i, j])
            cond = abs(w) >= threshold if absolute else w >= threshold
            if cond:
                u, v = labels[i], labels[j]
                if directed:
                    G.add_edge(u, v, **{weight_attr: w})
                else:
                    G.add_edge(u, v, **{weight_attr: w})

    return G
