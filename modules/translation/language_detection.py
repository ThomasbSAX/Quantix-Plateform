"""
Module de détection de langue pour les documents.
"""

from collections import Counter


def _detect(text: str):
    """Détection lazy (langdetect est optionnel)."""
    try:
        from langdetect import detect  # type: ignore
    except Exception as e:
        raise ImportError(
            "Le paquet 'langdetect' est requis pour la détection de langue. "
            "Installez-le avec `pip install langdetect`."
        ) from e

    return detect(text)


def detect_document_language(text, sample_size=1000):
    """
    Détecte la langue dominante d'un document en analysant un échantillon.
    Retourne le code de langue (ex: 'fr', 'en', 'es').
    """
    if not text or len(text.strip()) < 10:
        return None
    
    try:
        # Prend un échantillon du texte (début, milieu, fin)
        text_length = len(text)
        samples = []
        
        # Échantillon du début
        samples.append(text[:min(sample_size, text_length)])
        
        # Échantillon du milieu
        if text_length > sample_size * 2:
            mid = text_length // 2
            samples.append(text[mid:mid + sample_size])
        
        # Échantillon de la fin
        if text_length > sample_size:
            samples.append(text[-sample_size:])
        
        # Détecte la langue pour chaque échantillon
        detected_languages = []
        for sample in samples:
            try:
                lang = _detect(sample)
                detected_languages.append(lang)
            except:
                pass
        
        if detected_languages:
            # Retourne la langue la plus fréquente
            lang_counter = Counter(detected_languages)
            return lang_counter.most_common(1)[0][0]
        
        return None
    except Exception as e:
        print(f"Erreur détection langue: {e}")
        return None


def should_translate_segment(text, document_lang, min_words=3):
    """
    Détermine si un segment de texte doit être traduit.
    Ignore les citations courtes dans une langue différente.
    """
    if not text or len(text.strip()) < 5:
        return False
    
    # Compte le nombre de mots
    word_count = len(text.split())
    
    # Les segments très courts (moins de 3 mots) dans une autre langue sont probablement des citations
    if word_count < min_words:
        try:
            segment_lang = _detect(text)
            # Si la langue est différente de la langue du document, c'est une citation
            if segment_lang != document_lang:
                return False
        except:
            pass
    
    return True
