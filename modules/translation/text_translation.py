"""
Module de traduction de texte avec gestion des chunks.
"""

import re
from collections import OrderedDict
from typing import Any, Optional
from .language_detection import should_translate_segment


# Cache LRU en mémoire (par process) pour éviter de retraduire les mêmes segments.
# Très utile pour en-têtes/pieds de page, cellules répétées, etc.
_TRANSLATION_CACHE_MAX = 2000
_translation_cache: "OrderedDict[tuple[Any, ...], str]" = OrderedDict()


def _cache_get(key: tuple[Any, ...]) -> Optional[str]:
    try:
        val = _translation_cache.get(key)
        if val is None:
            return None
        # rafraîchir LRU
        _translation_cache.move_to_end(key)
        return val
    except Exception:
        return None


def _cache_put(key: tuple[Any, ...], value: str) -> None:
    try:
        _translation_cache[key] = value
        _translation_cache.move_to_end(key)
        # éviction LRU
        while len(_translation_cache) > _TRANSLATION_CACHE_MAX:
            _translation_cache.popitem(last=False)
    except Exception:
        # le cache ne doit jamais casser la traduction
        return


def _has_letters(text: str) -> bool:
    # Supporte toutes écritures (latin, cyrillique, arabe, etc.) via isalpha()
    return any(ch.isalpha() for ch in text)


def translate_text(text, translator, max_chunk_size=4500, document_lang=None, preserve_foreign_quotes=True):
    """
    Traduit un texte en gérant la limite de caractères.
    Découpe intelligemment pour éviter de casser des mots ou des structures.
    Peut préserver les citations dans une langue étrangère.
    """
    if not text or not text.strip():
        return text

    # Segment sans lettres => en général inutile d'appeler un traducteur réseau
    # (chiffres, symboles, ponctuation, formules, etc.).
    # On conserve tel quel pour accélérer.
    try:
        if not _has_letters(text):
            return text
    except Exception:
        pass
    
    # Si protection des citations activée, vérifier si c'est une citation
    if preserve_foreign_quotes and document_lang:
        if not should_translate_segment(text, document_lang):
            return text

    # Cache exact sur le segment complet (utile si translate_text est appelé
    # sur des éléments identiques dans différents endroits du document).
    base_key = (id(translator), document_lang, bool(preserve_foreign_quotes), int(max_chunk_size))
    full_key = base_key + (text,)
    cached_full = _cache_get(full_key)
    if cached_full is not None:
        return cached_full
    
    try:
        def _translate_one(chunk: str) -> str:
            k = base_key + (chunk,)
            cached = _cache_get(k)
            if cached is not None:
                return cached
            out = translator.translate(chunk)
            # Sécurise type
            out = out if isinstance(out, str) else str(out)
            _cache_put(k, out)
            return out

        if len(text) <= max_chunk_size:
            out = _translate_one(text)
            _cache_put(full_key, out)
            return out
        else:
            # Découpage intelligent par phrases pour préserver la structure
            # Cherche les points suivis d'espaces ou de fin de ligne
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                # Si ajouter cette phrase dépasse la limite
                if len(current_chunk) + len(sentence) > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = sentence
                    else:
                        # La phrase est trop longue, on la découpe proprement
                        words = sentence.split()
                        for word in words:
                            if len(current_chunk) + len(word) + 1 > max_chunk_size:
                                chunks.append(current_chunk)
                                current_chunk = word
                            else:
                                current_chunk += (" " if current_chunk else "") + word
                else:
                    current_chunk += (" " if current_chunk else "") + sentence
            
            if current_chunk:
                chunks.append(current_chunk)
            
            # Traduit chaque chunk et les joint avec des espaces appropriés
            translated_chunks = [_translate_one(chunk) for chunk in chunks]
            out = " ".join(translated_chunks)
            _cache_put(full_key, out)
            return out
    except Exception as e:
        print(f"Erreur de traduction: {e}")
        return text
