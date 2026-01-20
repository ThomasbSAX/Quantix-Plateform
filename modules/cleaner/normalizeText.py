import re
import unicodedata
from typing import Optional


def normalize_text_full(s: Optional[str]) -> Optional[str]:
    """
    Normalisation textuelle complète (robuste, data/NLP).

    Opérations :
    - suppression des emojis et symboles pictographiques Unicode
    - normalisation Unicode (NFKD)
    - suppression des accents (sans toucher aux lettres)
    - passage en minuscules
    - suppression des caractères de contrôle
    - normalisation des espaces (espaces multiples, tabs, retours ligne)
    """
    if s is None:
        return None
    s = str(s)

    # suppression emojis / pictogrammes (plages Unicode larges)
    s = re.sub(
        r"[\U0001F300-\U0001FAFF"
        r"\U00002700-\U000027BF"
        r"\U0001F100-\U0001F1FF"
        r"\U000024C2-\U0001F251]+",
        " ",
        s
    )

    # normalisation Unicode + suppression accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # minuscules
    s = s.lower()

    # suppression caractères de contrôle invisibles
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

    # normalisation espaces
    s = re.sub(r"\s+", " ", s).strip()

    return s
