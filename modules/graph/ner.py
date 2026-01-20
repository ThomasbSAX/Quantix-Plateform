import json
from typing import List, Dict

try:
    from gliner import GLiNER
except ImportError:
    GLiNER = None

# Liste des entités NER courantes
NER_LABELS = ["PERSON", "ORG", "LOCATION", "DATE", "MISC"]

def extract_ner_entities(text: str, labels: List[str] = NER_LABELS) -> List[Dict]:
    """
    Utilise GLiNER 2 pour extraire les entités nommées d'un texte.
    Retourne une liste de dicts {text, label, start, end}
    """
    if GLiNER is None:
        raise ImportError("GLiNER n'est pas installé. Installez-le avec 'pip install gliner'.")
    model = GLiNER.from_pretrained('urchade/gliner_base')
    entities = model.predict_entities(text, labels=labels)
    return entities

def build_ner_graph(entities: List[Dict], distance_metric: str = "overlap"):
    """
    Construit un graphe NER basé sur les entités extraites.
    Les nœuds sont les entités, les arêtes sont pondérées par la distance (overlap, proximité, etc.)
    """
    import networkx as nx
    G = nx.Graph()
    for i, ent1 in enumerate(entities):
        G.add_node(i, label=ent1['label'], text=ent1['text'])
    # Simple: arête si overlap ou proximité
    for i, ent1 in enumerate(entities):
        for j, ent2 in enumerate(entities):
            if i >= j:
                continue
            # Overlap
            if ent1['label'] == ent2['label']:
                continue
            dist = abs(ent1['start'] - ent2['start'])
            G.add_edge(i, j, weight=1/(1+dist))
    return G
