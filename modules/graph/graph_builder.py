from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
import zipfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import pandas as pd


# Imports compatibles "package" et "script" (utile avant intégration Flask)
try:
    from .cosinus import build_cosine_similarity_graph
    from .correlation import build_correlation_graph
    from .knn import build_knn_graph
    from .biparty import build_bipartite_graph
    from .dag import build_dag_from_dependencies
    from .ner import extract_ner_entities, build_ner_graph
    from .community import detect_communities_nx, detect_communities_igraph
    from .igraph_utils import build_igraph_from_edgelist, export_igraph_to_cytoscape_data
    from .text_cooccurrence import build_cooccurrence_graph
    from .doc_similarity import build_document_similarity_graph
    from .doc_citations import build_document_citation_graph
except Exception:  # pragma: no cover
    from cosinus import build_cosine_similarity_graph
    from correlation import build_correlation_graph
    from knn import build_knn_graph
    from biparty import build_bipartite_graph
    from dag import build_dag_from_dependencies
    from ner import extract_ner_entities, build_ner_graph
    from community import detect_communities_nx, detect_communities_igraph
    from igraph_utils import build_igraph_from_edgelist, export_igraph_to_cytoscape_data
    from text_cooccurrence import build_cooccurrence_graph
    from doc_similarity import build_document_similarity_graph
    from doc_citations import build_document_citation_graph


SUPPORTED_GRAPH_TYPES = [
    "cosinus",
    "correlation",
    "knn",
    "bipartite",
    "dag",
    "edgelist",
    "igraph",
    "community",
    "ner",
    "cooccurrence",
    "doc_similarity",
    "doc_citations",
    "doc_clusters",
]


@dataclass(frozen=True)
class InputItem:
    """Entrée normalisée (prête à être routée vers un builder)."""

    path: str
    kind: str  # "csv" | "txt" | "json" | "unknown"
    data: Any


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _read_zip(path: str) -> List[InputItem]:
    out: List[InputItem] = []
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            lower = name.lower()
            if lower.endswith("/"):
                continue
            # ignorer les artefacts macOS
            base = os.path.basename(name)
            if lower.startswith("__macosx/") or base.startswith("._"):
                continue
            with zf.open(name) as f:
                raw = f.read()
            pseudo_path = f"{path}:{name}"
            if lower.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(raw))
                out.append(InputItem(path=pseudo_path, kind="csv", data=df))
            elif lower.endswith(".tsv"):
                df = pd.read_csv(io.BytesIO(raw), sep="\t")
                out.append(InputItem(path=pseudo_path, kind="csv", data=df))
            elif lower.endswith(".json"):
                out.append(InputItem(path=pseudo_path, kind="json", data=json.loads(raw.decode("utf-8"))))
            elif lower.endswith(".pdf"):
                text = _extract_pdf_text(raw)
                if text:
                    out.append(InputItem(path=pseudo_path, kind="txt", data=text))
                else:
                    # Fallback: certains fichiers finissent en .pdf mais contiennent du texte/HTML.
                    try:
                        fallback = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        fallback = raw.decode("latin-1")
                    if fallback.strip():
                        out.append(InputItem(path=pseudo_path, kind="txt", data=fallback))
            else:
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw.decode("latin-1")
                out.append(InputItem(path=pseudo_path, kind="txt", data=text))
    return out


def _extract_pdf_text(raw: bytes) -> str:
    """Extraction simple de texte PDF (optionnelle).

    Si `pypdf` n'est pas installé, retourne une chaîne vide.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return ""

    # Certains ZIP contiennent des fichiers .pdf qui ne sont pas des PDF (HTML, etc.).
    # On évite de déclencher des warnings/bruits inutiles.
    if not raw.startswith(b"%PDF"):
        return ""

    try:
        reader = PdfReader(io.BytesIO(raw))
        parts: List[str] = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
        return "\n\n".join(parts)
    except Exception:
        return ""


def load_inputs(file_paths: Union[str, Sequence[str]]) -> List[InputItem]:
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    items: List[InputItem] = []
    for p in file_paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".zip":
            items.extend(_read_zip(p))
        elif ext in {".csv", ".tsv"}:
            items.append(InputItem(path=p, kind="csv", data=_read_table(p)))
        elif ext == ".txt":
            items.append(InputItem(path=p, kind="txt", data=_read_text(p)))
        elif ext == ".json":
            items.append(InputItem(path=p, kind="json", data=_read_json(p)))
        else:
            items.append(InputItem(path=p, kind="unknown", data=_read_text(p)))
    return items


def _df_to_numeric_matrix(df: pd.DataFrame, *, drop_non_numeric: bool = True) -> List[List[float]]:
    if drop_non_numeric:
        df = df.select_dtypes(include=["number"])
    if df.shape[1] < 2:
        raise ValueError("CSV: besoin d'au moins 2 colonnes numériques")
    return df.to_numpy(dtype=float).tolist()


def _df_to_edge_list(
    df: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: Optional[str] = "weight",
) -> List[Tuple[Any, Any, float]]:
    if source_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"edgelist: colonnes requises: '{source_col}', '{target_col}'")

    edges: List[Tuple[Any, Any, float]] = []
    for _, row in df.iterrows():
        u = row[source_col]
        v = row[target_col]
        if pd.isna(u) or pd.isna(v):
            continue
        w = 1.0
        if weight_col and weight_col in df.columns and not pd.isna(row[weight_col]):
            w = float(row[weight_col])
        edges.append((u, v, w))
    return edges


def _merge_graphs(graphs: Sequence[nx.Graph], *, strategy: str = "compose") -> nx.Graph:
    if not graphs:
        return nx.Graph()
    if strategy not in {"compose", "disjoint_union"}:
        raise ValueError("merge_strategy must be 'compose' or 'disjoint_union'")

    out = graphs[0].copy()
    for g in graphs[1:]:
        if strategy == "compose":
            out = nx.compose(out, g)
        else:
            out = nx.disjoint_union(out, g)
    return out


Builder = Callable[[List[InputItem], Dict[str, Any]], Any]


def _build_cosinus(inputs: List[InputItem], opts: Dict[str, Any]) -> nx.Graph:
    matrices: List[List[List[float]]] = []
    for item in inputs:
        if item.kind == "csv":
            matrices.append(_df_to_numeric_matrix(item.data))
    if not matrices:
        raise ValueError("cosinus: fournir au moins un CSV avec colonnes numériques")
    X = [row for m in matrices for row in m] if len(matrices) > 1 else matrices[0]
    return build_cosine_similarity_graph(X, **opts)


def _build_correlation(inputs: List[InputItem], opts: Dict[str, Any]) -> nx.Graph:
    matrices: List[List[List[float]]] = []
    for item in inputs:
        if item.kind == "csv":
            matrices.append(_df_to_numeric_matrix(item.data))
    if not matrices:
        raise ValueError("correlation: fournir au moins un CSV")
    X = [row for m in matrices for row in m] if len(matrices) > 1 else matrices[0]
    return build_correlation_graph(X, **opts)


def _build_knn(inputs: List[InputItem], opts: Dict[str, Any]) -> nx.Graph:
    matrices: List[List[List[float]]] = []
    for item in inputs:
        if item.kind == "csv":
            matrices.append(_df_to_numeric_matrix(item.data))
    if not matrices:
        raise ValueError("knn: fournir au moins un CSV")
    X = [row for m in matrices for row in m] if len(matrices) > 1 else matrices[0]
    return build_knn_graph(X, **opts)


def _build_edgelist(inputs: List[InputItem], opts: Dict[str, Any]) -> nx.Graph:
    graphs: List[nx.Graph] = []
    for item in inputs:
        if item.kind != "csv":
            continue
        edges = _df_to_edge_list(
            item.data,
            source_col=str(opts.get("source_col", "source")),
            target_col=str(opts.get("target_col", "target")),
            weight_col=opts.get("weight_col", "weight"),
        )
        directed = bool(opts.get("directed", False))
        G: nx.Graph = nx.DiGraph() if directed else nx.Graph()
        weight_attr = str(opts.get("weight_attr", "weight"))
        for u, v, w in edges:
            G.add_edge(u, v, **{weight_attr: float(w)})
        graphs.append(G)

    if not graphs:
        raise ValueError("edgelist: fournir au moins un CSV avec colonnes source/target")
    return _merge_graphs(graphs, strategy=str(opts.get("merge_strategy", "compose")))


def _build_igraph(inputs: List[InputItem], opts: Dict[str, Any]) -> Any:
    edgelist: List[Sequence[Any]] = []

    for item in inputs:
        if item.kind == "csv":
            edges = _df_to_edge_list(
                item.data,
                source_col=str(opts.get("source_col", "source")),
                target_col=str(opts.get("target_col", "target")),
                weight_col=opts.get("weight_col", "weight"),
            )
            edgelist.extend(edges)
        elif item.kind == "json" and isinstance(item.data, list):
            edgelist.extend(item.data)

    if not edgelist:
        raise ValueError("igraph: fournir un edgelist via CSV ou JSON")
    return build_igraph_from_edgelist(edgelist, directed=bool(opts.get("directed", False)))


def _build_bipartite(_: List[InputItem], opts: Dict[str, Any]) -> nx.Graph:
    return build_bipartite_graph(**opts)


def _build_dag(_: List[InputItem], opts: Dict[str, Any]) -> nx.DiGraph:
    return build_dag_from_dependencies(**opts)


def _build_ner(inputs: List[InputItem], opts: Dict[str, Any]) -> nx.Graph:
    texts: List[str] = []
    for item in inputs:
        if item.kind in {"txt", "unknown"}:
            texts.append(str(item.data))
    if not texts and "text" in opts:
        texts = [str(opts["text"])]
    if not texts:
        raise ValueError("ner: fournir au moins un .txt ou option text=...")

    text = "\n\n".join(texts)
    labels = opts.get("labels")
    entities = extract_ner_entities(text, labels=labels) if labels else extract_ner_entities(text)
    return build_ner_graph(entities, distance_metric=str(opts.get("distance_metric", "overlap")))


def _build_community(inputs: List[InputItem], opts: Dict[str, Any]) -> Dict[Any, int]:
    graph = opts.get("graph")
    if graph is None:
        graph = _build_edgelist(inputs, opts)

    if hasattr(graph, "nodes") and hasattr(graph, "edges"):
        return detect_communities_nx(graph)
    if "igraph" in str(type(graph)).lower():
        return detect_communities_igraph(graph)
    raise ValueError("community: fournir un graphe NetworkX ou igraph")


def _build_cooccurrence(inputs: List[InputItem], opts: Dict[str, Any]) -> nx.Graph:
    texts: List[str] = []
    for item in inputs:
        if item.kind in {"txt", "unknown"}:
            texts.append(str(item.data))
    if not texts and "text" in opts:
        texts = [str(opts["text"])]
    if not texts:
        raise ValueError("cooccurrence: fournir au moins un .txt ou option text=...")
    return build_cooccurrence_graph(texts, **opts)


def _documents_from_inputs(inputs: List[InputItem], opts: Dict[str, Any]) -> Dict[str, str]:
    """Construit un corpus {doc_id: text} à partir des InputItems.

    - Pour un ZIP: `InputItem.path` ressemble à "archive.zip:folder/file.txt"
    - On accepte aussi du texte direct via option `text=` (doc_id = "inline:text")
    """
    docs: Dict[str, str] = {}
    for item in inputs:
        if item.kind in {"txt", "unknown"}:
            docs[str(item.path)] = str(item.data)
    if not docs and "text" in opts:
        docs["inline:text"] = str(opts["text"])
    return docs


def _build_doc_similarity(inputs: List[InputItem], opts: Dict[str, Any]) -> nx.Graph:
    docs = _documents_from_inputs(inputs, opts)
    if len(docs) < 1:
        raise ValueError("doc_similarity: fournir un ZIP/ensemble de .txt/.md/.html ou option text=...")
    return build_document_similarity_graph(docs, **opts)


def _build_doc_citations(inputs: List[InputItem], opts: Dict[str, Any]) -> nx.DiGraph:
    docs = _documents_from_inputs(inputs, opts)
    if len(docs) < 1:
        raise ValueError("doc_citations: fournir un ZIP/ensemble de documents texte")
    return build_document_citation_graph(docs, **opts)


def _build_doc_clusters(inputs: List[InputItem], opts: Dict[str, Any]) -> Dict[str, int]:
    # Construit d'abord le graphe de similarité, puis clusterise (sans dépendance externe)
    G = _build_doc_similarity(inputs, opts)
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_edges() == 0:
        return {str(n): i for i, n in enumerate(G.nodes())}

    from networkx.algorithms.community import greedy_modularity_communities

    communities = list(greedy_modularity_communities(G))
    partition: Dict[str, int] = {}
    for i, comm in enumerate(communities):
        for node in comm:
            partition[str(node)] = i
    return partition


_REGISTRY: Dict[str, Builder] = {
    "cosinus": _build_cosinus,
    "correlation": _build_correlation,
    "knn": _build_knn,
    "bipartite": _build_bipartite,
    "dag": _build_dag,
    "edgelist": _build_edgelist,
    "igraph": _build_igraph,
    "community": _build_community,
    "ner": _build_ner,
    "cooccurrence": _build_cooccurrence,
    "doc_similarity": _build_doc_similarity,
    "doc_citations": _build_doc_citations,
    "doc_clusters": _build_doc_clusters,
}


def build_graph_from_inputs(inputs: List[InputItem], graph_type: str, **options: Any) -> Any:
    if graph_type not in _REGISTRY:
        raise ValueError(f"Type de graphe non supporté: {graph_type}. Supported={sorted(_REGISTRY.keys())}")
    return _REGISTRY[graph_type](inputs, options)


def build_graph_from_files(file_paths: Union[str, Sequence[str]], graph_type: str, **options: Any) -> Any:
    return build_graph_from_inputs(load_inputs(file_paths), graph_type, **options)


def export_graph_data(graph: Any, *, format: str = "node_link") -> Dict[str, Any]:
    fmt = format.lower()

    if fmt in {"node_link", "json", "networkx"}:
        if hasattr(graph, "nodes") and hasattr(graph, "edges"):
            data = nx.node_link_data(graph)
            # Compat NetworkX: certaines versions utilisent `links`, d'autres `edges`.
            if "edges" in data and "links" not in data:
                data["links"] = data["edges"]
            if "links" in data and "edges" not in data:
                data["edges"] = data["links"]
            return data
        if "igraph" in str(type(graph)).lower():
            return export_igraph_to_cytoscape_data(graph)
        if isinstance(graph, dict):
            return graph
        raise ValueError("export node_link: type inconnu")

    if fmt == "cytoscape":
        if hasattr(graph, "nodes") and hasattr(graph, "edges"):
            nodes = [{"data": {"id": str(n), **(graph.nodes[n] or {})}} for n in graph.nodes]
            edges = []
            for u, v, data in graph.edges(data=True):
                payload = {"source": str(u), "target": str(v)}
                if data:
                    payload.update(data)
                edges.append({"data": payload})
            return {"elements": {"nodes": nodes, "edges": edges}}
        if "igraph" in str(type(graph)).lower():
            return export_igraph_to_cytoscape_data(graph)
        raise ValueError("export cytoscape: type inconnu")

    if fmt == "communities":
        if isinstance(graph, dict):
            return {"partition": graph}
        raise ValueError("export communities: attendre dict node->community")

    raise ValueError(f"Format d'export non supporté: {format}")


def export_graph_text(graph: Any, *, format: str = "json") -> str:
    fmt = format.lower()
    if fmt in {"json", "node_link", "cytoscape", "communities"}:
        return json.dumps(export_graph_data(graph, format=fmt), ensure_ascii=False)

    if fmt in {"graphml", "gexf"}:
        if not (hasattr(graph, "nodes") and hasattr(graph, "edges")):
            raise ValueError(f"{fmt}: uniquement pour NetworkX")
        buf = io.StringIO()
        if fmt == "graphml":
            nx.write_graphml(graph, buf)
        else:
            nx.write_gexf(graph, buf)
        return buf.getvalue()

    raise ValueError(f"Format d'export non supporté: {format}")


def compute_graph_metrics(graph: nx.Graph) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics["num_nodes"] = int(graph.number_of_nodes())
    metrics["num_edges"] = int(graph.number_of_edges())
    metrics["degrees"] = {str(n): int(d) for n, d in graph.degree()}

    if graph.number_of_nodes() == 0:
        metrics["diameter"] = None
        metrics["clustering"] = None
        return metrics

    try:
        connected = nx.is_weakly_connected(graph) if graph.is_directed() else nx.is_connected(graph)
    except Exception:
        connected = False

    metrics["diameter"] = nx.diameter(graph) if connected and graph.number_of_nodes() > 1 else None
    metrics["clustering"] = nx.clustering(graph) if not graph.is_directed() else None
    return metrics


def filter_graph_by_degree(graph: nx.Graph, *, min_degree: int = 1) -> nx.Graph:
    nodes_to_keep = [n for n, d in graph.degree() if d >= min_degree]
    return graph.subgraph(nodes_to_keep).copy()


def main(file_path: Union[str, Sequence[str]], graph_type: str, export_format: str = "json", **kwargs: Any) -> str:
    """Compat: conserve l'ancien contrat (fichier + type + export)."""
    graph = build_graph_from_files(file_path, graph_type, **kwargs)
    return export_graph_text(graph, format=export_format)
