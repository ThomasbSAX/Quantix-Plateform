from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple
import networkx as nx


def build_dag_from_dependencies(
    nodes: Sequence[Any],
    dependencies: Iterable[Tuple[Any, Any]],
    *,
    allow_cycles: bool = False,
    weight_attr: Optional[str] = None,
    weights: Optional[Iterable[float]] = None,
    node_attrs: Optional[Mapping[str, Sequence[Any]]] = None,
) -> nx.DiGraph:
    """
    Construit un graphe orienté acyclique (DAG) à partir de dépendances.
    - nodes: liste des nœuds
    - dependencies: arêtes (u, v) signifiant u -> v (u précède v)
    - allow_cycles: si False, lève une erreur si un cycle est détecté
    - weights: poids optionnels alignés sur dependencies
    """
    G = nx.DiGraph()

    for n in nodes:
        G.add_node(n)

    if node_attrs:
        for name, values in node_attrs.items():
            if len(values) != len(nodes):
                raise ValueError(f"node attr '{name}' length mismatch")
            for n, val in zip(nodes, values):
                G.nodes[n][name] = val

    if weights is not None:
        for (u, v), w in zip(dependencies, weights):
            if u not in G or v not in G:
                raise ValueError(f"unknown node in edge {(u, v)}")
            if weight_attr is None:
                G.add_edge(u, v)
            else:
                G.add_edge(u, v, **{weight_attr: float(w)})
    else:
        for u, v in dependencies:
            if u not in G or v not in G:
                raise ValueError(f"unknown node in edge {(u, v)}")
            G.add_edge(u, v)

    if not allow_cycles:
        try:
            _ = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            raise ValueError("cycle detected: graph is not a DAG")

    return G
