from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple
import networkx as nx


def build_bipartite_graph(
    left_nodes: Sequence[Any],
    right_nodes: Sequence[Any],
    edges: Iterable[Tuple[Any, Any, Optional[float]]],
    *,
    left_attr: str = "left",
    right_attr: str = "right",
    weight_attr: str = "weight",
    allow_isolated: bool = True,
    directed: bool = False,
    node_attrs_left: Optional[Mapping[str, Sequence[Any]]] = None,
    node_attrs_right: Optional[Mapping[str, Sequence[Any]]] = None,
) -> nx.Graph:
    """
    Construit un graphe biparti.
    - left_nodes: liste des nœuds de la partition gauche
    - right_nodes: liste des nœuds de la partition droite
    - edges: itérable de (u, v, w) avec u∈left, v∈right, w optionnel (poids)
    - Attributs de partition: left_attr=True pour gauche, right_attr=True pour droite
    """
    G: nx.Graph = nx.DiGraph() if directed else nx.Graph()

    for u in left_nodes:
        G.add_node(u, **{left_attr: True, right_attr: False})
    for v in right_nodes:
        G.add_node(v, **{left_attr: False, right_attr: True})

    if node_attrs_left:
        for name, values in node_attrs_left.items():
            if len(values) != len(left_nodes):
                raise ValueError(f"left attr '{name}' length mismatch")
            for u, val in zip(left_nodes, values):
                G.nodes[u][name] = val

    if node_attrs_right:
        for name, values in node_attrs_right.items():
            if len(values) != len(right_nodes):
                raise ValueError(f"right attr '{name}' length mismatch")
            for v, val in zip(right_nodes, values):
                G.nodes[v][name] = val

    for e in edges:
        if len(e) == 2:
            u, v = e
            w = None
        else:
            u, v, w = e

        if u not in G or v not in G:
            raise ValueError(f"edge references unknown node: {(u, v)}")

        if not (G.nodes[u].get(left_attr) and G.nodes[v].get(right_attr)):
            if not (G.nodes[v].get(left_attr) and G.nodes[u].get(right_attr)):
                raise ValueError("edge must connect left to right")

        if w is None:
            G.add_edge(u, v)
        else:
            G.add_edge(u, v, **{weight_attr: float(w)})

    if not allow_isolated:
        isolates = [n for n, d in G.degree() if d == 0]
        G.remove_nodes_from(isolates)

    return G
