from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import json

try:
    import igraph as ig  # type: ignore
except ImportError:  # pragma: no cover
    ig = None

def build_igraph_from_edgelist(
    edgelist: Sequence[Sequence[Any]],
    directed: bool = False,
):
    """
    Construit un graphe igraph à partir d'une liste d'arêtes [(source, target, poids)]
    """
    if ig is None:
        raise ImportError("igraph n'est pas installé. Installez-le avec 'pip install igraph'.")

    if not edgelist:
        g = ig.Graph(directed=directed)
        g.vs["name"] = []
        g.es["weight"] = []
        return g

    # Supporte des identifiants non-numériques (str, etc.) en les mappant sur des indices.
    nodes: List[Any] = []
    index: Dict[Any, int] = {}

    def _idx(x: Any) -> int:
        if x in index:
            return index[x]
        index[x] = len(nodes)
        nodes.append(x)
        return index[x]

    edges: List[Tuple[int, int]] = []
    weights: List[float] = []
    for e in edgelist:
        if len(e) < 2:
            raise ValueError("each edge must have at least (source, target)")
        u, v = e[0], e[1]
        w = e[2] if len(e) > 2 else 1.0
        edges.append((_idx(u), _idx(v)))
        weights.append(float(w))

    g = ig.Graph(n=len(nodes), edges=edges, directed=directed)
    g.vs["name"] = [str(n) for n in nodes]
    g.es["weight"] = weights
    return g

def export_igraph_to_cytoscape_data(g) -> Dict[str, Any]:
    """
    Exporte un igraph au format dict compatible Cytoscape.js
    """
    if ig is None:
        raise ImportError("igraph n'est pas installé. Installez-le avec 'pip install igraph'.")

    nodes = [
        {
            "data": {
                "id": str(v.index),
                "label": str(v["name"]) if "name" in v.attributes() else str(v.index),
            }
        }
        for v in g.vs
    ]
    edges = [
        {
            "data": {
                "source": str(e.source),
                "target": str(e.target),
                "weight": float(e["weight"]) if "weight" in e.attributes() else 1.0,
            }
        }
        for e in g.es
    ]
    return {"elements": {"nodes": nodes, "edges": edges}}


def export_igraph_to_cytoscape_json(g) -> str:
    return json.dumps(export_igraph_to_cytoscape_data(g))

