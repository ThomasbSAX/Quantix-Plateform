"""Outils de génération de graphes.

Ce package regroupe plusieurs méthodes (corrélation, kNN, cosinus, biparti, DAG, NER, etc.)
et un point d’entrée unifié via `graph_builder`.
"""

from .graph_builder import (
    SUPPORTED_GRAPH_TYPES,
    build_graph_from_files,
    build_graph_from_inputs,
    export_graph_data,
    export_graph_text,
)
