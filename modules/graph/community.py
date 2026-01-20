from __future__ import annotations

from typing import Any, Dict

import networkx as nx

try:
    import igraph as ig  # type: ignore
except ImportError:  # pragma: no cover
    ig = None

def detect_communities_nx(graph: nx.Graph) -> Dict[Any, int]:
    """
    Détecte les communautés dans un graphe NetworkX (méthode de Louvain si disponible, sinon Girvan-Newman)
    """
    try:
        # IMPORTANT: éviter la collision avec ce fichier `community.py`.
        import community.community_louvain as community_louvain  # type: ignore

        # Louvain fonctionne en pratique sur des graphes non-orientés
        G = graph.to_undirected(as_view=False) if graph.is_directed() else graph
        partition = community_louvain.best_partition(G)
        return partition
    except ImportError:
        # Fallback Girvan-Newman
        from networkx.algorithms.community import girvan_newman
        G = graph.to_undirected(as_view=False) if graph.is_directed() else graph
        comp = girvan_newman(G)
        first_level = next(comp)
        partition = {}
        for i, community in enumerate(first_level):
            for node in community:
                partition[node] = i
        return partition

def detect_communities_igraph(g) -> Dict[Any, int]:
    """
    Détecte les communautés dans un igraph (méthode Louvain)
    """
    if ig is None:
        raise ImportError("igraph n'est pas installé. Installez-le avec 'pip install igraph'.")
    louvain = g.community_multilevel()
    membership = louvain.membership
    return {g.vs[i]['name'] if 'name' in g.vs[i].attributes() else i: c for i, c in enumerate(membership)}

