"""Point d'entrée principal pour la traduction multiformat.

Objectifs:
- Exposer (ré-exporter) toutes les fonctions des modules de traduction via ce fichier.
- Rester "import-safe" pour Flask: aucune dépendance lourde requise au moment de l'import.
  Les imports sont faits à l'appel (lazy import).
"""

from __future__ import annotations

from pathlib import Path
import importlib
from typing import Any, Callable, Optional


def _lazy_attr(module: str, attr: str) -> Any:
    """Récupère un symbole via import lazy.

    module: nom relatif (ex: ".pdf_translator")
    attr: nom d'attribut dans le module
    """
    try:
        mod = importlib.import_module(module, package=__package__)
    except Exception as e:
        raise ImportError(
            f"Impossible d'importer {module} (package={__package__}). "
            f"Dépendance manquante ou erreur d'import: {e}"
        ) from e

    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise ImportError(f"Symbole introuvable: {module}:{attr}") from e


# --- Ré-exports (wrappers lazy) ---

# format_handlers.py
def translate_txt(input_path, output_path, translator):
    return _lazy_attr(".format_handlers", "translate_txt")(input_path, output_path, translator)


def translate_csv(input_path, output_path, translator):
    return _lazy_attr(".format_handlers", "translate_csv")(input_path, output_path, translator)


def translate_json(input_path, output_path, translator):
    return _lazy_attr(".format_handlers", "translate_json")(input_path, output_path, translator)


# table_translator.py
def translate_table_column(
    input_path,
    output_path,
    *,
    column,
    target_lang,
    source_lang: str = "auto",
    output_column=None,
    replace: bool = False,
    preserve_math: bool = True,
    cache_unique: bool = True,
    preserve_empty: bool = True,
    encoding: str = "utf-8",
    delimiter=None,
    quotechar=None,
):
    return _lazy_attr(".table_translator", "translate_table_column")(
        input_path,
        output_path,
        column=column,
        target_lang=target_lang,
        source_lang=source_lang,
        output_column=output_column,
        replace=replace,
        preserve_math=preserve_math,
        cache_unique=cache_unique,
        preserve_empty=preserve_empty,
        encoding=encoding,
        delimiter=delimiter,
        quotechar=quotechar,
    )

def translate_table_columns(
    input_path,
    output_path,
    *,
    columns,
    target_lang,
    source_lang: str = "auto",
    replace: bool = False,
    output_suffix=None,
    preserve_math: bool = True,
    cache_unique: bool = True,
    preserve_empty: bool = True,
    encoding: str = "utf-8",
    delimiter=None,
    quotechar=None,
):
    return _lazy_attr(".table_translator", "translate_table_columns")(
        input_path,
        output_path,
        columns=columns,
        target_lang=target_lang,
        source_lang=source_lang,
        replace=replace,
        output_suffix=output_suffix,
        preserve_math=preserve_math,
        cache_unique=cache_unique,
        preserve_empty=preserve_empty,
        encoding=encoding,
        delimiter=delimiter,
        quotechar=quotechar,
    )


# html_translator.py
def translate_html(input_path, output_path, translator):
    return _lazy_attr(".html_translator", "translate_html")(input_path, output_path, translator)


# markdown_translator.py
def translate_md(input_path, output_path, translator):
    return _lazy_attr(".markdown_translator", "translate_md")(input_path, output_path, translator)


# pdf_translator.py
def translate_pdf(
    input_path,
    output_path,
    translator,
    pages_to_translate=None,
    preserve_foreign_quotes: bool = True,
    progress_callback: Optional[Callable[..., Any]] = None,
):
    return _lazy_attr(".pdf_translator", "translate_pdf")(
        input_path,
        output_path,
        translator,
        pages_to_translate=pages_to_translate,
        preserve_foreign_quotes=preserve_foreign_quotes,
        progress_callback=progress_callback,
    )


# pdf_translator_docling.py
def translate_pdf_with_docling(input_path, output_path, translator, pages_to_translate=None):
    return _lazy_attr(".pdf_translator_docling", "translate_pdf_with_docling")(
        input_path, output_path, translator, pages_to_translate=pages_to_translate
    )


def translate_pdf_smart(
    input_path,
    output_path,
    translator,
    pages_to_translate=None,
    preserve_foreign_quotes: bool = True,
    use_docling: bool = True,
):
    return _lazy_attr(".pdf_translator_docling", "translate_pdf_smart")(
        input_path,
        output_path,
        translator,
        pages_to_translate=pages_to_translate,
        preserve_foreign_quotes=preserve_foreign_quotes,
        use_docling=use_docling,
    )


# docx_translator.py
def _is_protected_run(run):
    return _lazy_attr(".docx_translator", "_is_protected_run")(run)


def _group_runs_by_style(paragraph):
    return _lazy_attr(".docx_translator", "_group_runs_by_style")(paragraph)


def _translate_paragraph(paragraph, translator):
    return _lazy_attr(".docx_translator", "_translate_paragraph")(paragraph, translator)


def translate_docx(input_path, output_path, translator):
    return _lazy_attr(".docx_translator", "translate_docx")(input_path, output_path, translator)


# image_translator.py
def translate_image(input_path, output_path, translator):
    return _lazy_attr(".image_translator", "translate_image")(input_path, output_path, translator)


# image_translator_docling.py
def translate_image_with_docling(input_path, output_path, translator):
    return _lazy_attr(".image_translator_docling", "translate_image_with_docling")(
        input_path, output_path, translator
    )


def translate_image_smart(input_path, output_path, translator, use_docling: bool = True):
    return _lazy_attr(".image_translator_docling", "translate_image_smart")(
        input_path, output_path, translator, use_docling=use_docling
    )


# language_detection.py
def detect_document_language(text, sample_size: int = 1000):
    return _lazy_attr(".language_detection", "detect_document_language")(text, sample_size=sample_size)


def should_translate_segment(text, document_lang, min_words: int = 3):
    return _lazy_attr(".language_detection", "should_translate_segment")(text, document_lang, min_words=min_words)


# math_detection.py
def is_mathematical_content(text):
    return _lazy_attr(".math_detection", "is_mathematical_content")(text)


def is_likely_math_notation(text):
    return _lazy_attr(".math_detection", "is_likely_math_notation")(text)


# text_translation.py
def translate_text(
    text,
    translator,
    max_chunk_size: int = 4500,
    document_lang=None,
    preserve_foreign_quotes: bool = True,
):
    return _lazy_attr(".text_translation", "translate_text")(
        text,
        translator,
        max_chunk_size=max_chunk_size,
        document_lang=document_lang,
        preserve_foreign_quotes=preserve_foreign_quotes,
    )


def docling_available() -> bool:
    """Indique si les traducteurs Docling peuvent être importés."""
    try:
        importlib.import_module(".pdf_translator_docling", package=__package__)
        importlib.import_module(".image_translator_docling", package=__package__)
        return True
    except Exception:
        return False


def translate(
    input_path,
    output_path,
    target_lang,
    pages_to_translate=None,
    preserve_foreign_quotes: bool = True,
    use_docling: bool = True,
    progress_callback: Optional[Callable[..., Any]] = None,
):
    """
    Traduit un fichier en préservant exactement son format d'origine.
    
    Args:
        input_path: Chemin du fichier source
        output_path: Chemin du fichier traduit
        target_lang: Langue cible (code ISO: 'fr', 'en', 'es', etc.)
        pages_to_translate: Pour PDF - liste des pages à traduire (1-indexé: [1,2,3]) ou None pour tout
        preserve_foreign_quotes: Si True, ne traduit pas les courtes citations en langue étrangère
        use_docling: (Ignoré) La voie Docling/Granite est actuellement désactivée.
        progress_callback: Fonction appelée avec (page_actuelle, total_pages) pour suivre la progression
    
    Exemples:
        # Traduire tout un PDF avec Docling (détection intelligente des formules)
        translate("doc.pdf", "doc_fr.pdf", "fr", use_docling=True)
        
        # Traduire uniquement les pages 1, 3, 5 d'un PDF (méthode traditionnelle)
        translate("doc.pdf", "doc_fr.pdf", "fr", pages_to_translate=[1, 3, 5], use_docling=False)
        
        # Traduire une image avec Docling
        translate("image.jpg", "image_fr.jpg", "fr", use_docling=True)
    """
    ext = Path(input_path).suffix.lower()
    try:
        from deep_translator import GoogleTranslator
    except Exception as e:
        raise ImportError(
            "deep-translator est requis pour créer un GoogleTranslator. "
            "Installez-le avec `pip install deep-translator`."
        ) from e

    translator = GoogleTranslator(source="auto", target=target_lang)
    
    # Convertir les numéros de pages de 1-indexé à 0-indexé pour PDF
    pages_0_indexed = None
    if pages_to_translate is not None and ext == ".pdf":
        pages_0_indexed = [p - 1 for p in pages_to_translate]
    
    # NOTE: Docling/Granite est désactivé (fonctionnalité en développement).
    # On force la traduction traditionnelle, quel que soit le paramètre `use_docling`.
    use_docling = False

    # Méthode traditionnelle
    print("Mode: Traduction traditionnelle")
    
    handlers = {
        ".txt": translate_txt,
        ".csv": translate_csv,
        ".json": translate_json,
        ".html": translate_html,
        ".htm": translate_html,
        ".md": translate_md,
        ".markdown": translate_md,
        ".pdf": lambda inp, outp, trans: translate_pdf(
            inp,
            outp,
            trans,
            pages_to_translate=pages_0_indexed,
            preserve_foreign_quotes=preserve_foreign_quotes,
            progress_callback=progress_callback,
        ),
        ".docx": translate_docx,
        ".jpg": translate_image,
        ".jpeg": translate_image,
        ".png": translate_image,
        ".bmp": translate_image,
        ".tiff": translate_image,
        ".tif": translate_image,
    }
    
    if ext in handlers:
        print(f"Début de la traduction vers {target_lang.upper()}...")
        handlers[ext](input_path, output_path, translator)
        print(f"Traduction terminée: {output_path}")
    else:
        raise ValueError(f"Format non pris en charge: {ext}\nFormats supportés: {', '.join(handlers.keys())}")


__all__ = [
    # main entrypoint
    "translate",
    "docling_available",
    # format handlers
    "translate_txt",
    "translate_csv",
    "translate_json",
    # html/markdown
    "translate_html",
    "translate_md",
    # pdf
    "translate_pdf",
    "translate_pdf_with_docling",
    "translate_pdf_smart",
    # docx
    "translate_docx",
    "_is_protected_run",
    "_group_runs_by_style",
    "_translate_paragraph",
    # images
    "translate_image",
    "translate_image_with_docling",
    "translate_image_smart",
    # language/math/text
    "detect_document_language",
    "should_translate_segment",
    "is_mathematical_content",
    "is_likely_math_notation",
    "translate_text",
        "translate_table_columns",
]

