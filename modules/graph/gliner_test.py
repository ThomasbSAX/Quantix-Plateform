from gliner import GLiNER

# Charger le modèle GlinER 2 (multilingue)
model = GLiNER.from_pretrained('urchade/gliner_multi')

# Lire le texte depuis test.txt
with open('test.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# Prédire les entités avec des labels spécifiques
labels = ["person", "company", "location"]
results = model.predict_entities(text, labels=labels)

# Afficher les résultats par type d'entité
from collections import defaultdict
entities_by_label = defaultdict(list)
for entity in results:
    entities_by_label[entity['label']].append(entity['text'])
import json
print(json.dumps({"entities": dict(entities_by_label)}, ensure_ascii=False, indent=2))
