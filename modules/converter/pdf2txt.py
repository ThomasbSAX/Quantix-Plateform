from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import logging
import re
import json

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption

Logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Normalisation légère du texte (sans casser maths / unicode)
# ------------------------------------------------------------

def _normalize_text(text: str) -> str:
    # supprimer coupures de mots en fin de ligne (ex: "inter-\nnational")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # lignes multiples → max 2 sauts
    text = re.sub(r"\n{3,}", "\n\n", text)

    # espaces inutiles
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# ------------------------------------------------------------
# PDF → texte brut via Granite
# ------------------------------------------------------------

def pdf_to_text_granite(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    do_ocr: bool = True,
    normalize: bool = True,
    save_metadata: bool = True,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Convertit un PDF en texte brut via Granite (Docling).

    Sorties :
    - <name>.txt      : texte extrait
    - <name>.meta.json (optionnel) : métadonnées utiles pipeline

    Returns:
        dict {
            success: bool,
            text_path: str | None,
            chars: int,
            pages: int | None,
            used_ocr: bool,
            error: str | None
        }
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    output_dir = Path(output_dir or pdf_path.parent / "result" / "texte")
    output_dir.mkdir(parents=True, exist_ok=True)

    text_path = output_dir / f"{pdf_path.stem}.txt"
    meta_path = output_dir / f"{pdf_path.stem}.meta.json"

    # --- Pipeline Granite
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr

    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }

    try:
        converter = DocumentConverter(format_options=format_options)
        result = converter.convert(str(pdf_path))
    except Exception as exc:
        return {
            "success": False,
            "text_path": None,
            "chars": 0,
            "pages": None,
            "used_ocr": do_ocr,
            "error": f"Granite conversion failed: {exc}",
        }

    # --- Extraction texte
    try:
        text = result.document.export_to_text()
    except Exception as exc:
        return {
            "success": False,
            "text_path": None,
            "chars": 0,
            "pages": None,
            "used_ocr": do_ocr,
            "error": f"Text export failed: {exc}",
        }

    if not text.strip():
        return {
            "success": False,
            "text_path": None,
            "chars": 0,
            "pages": None,
            "used_ocr": do_ocr,
            "error": "Texte vide (PDF image sans OCR ?)",
        }

    if normalize:
        text = _normalize_text(text)

    # --- Écriture fichier texte
    text_path.write_text(text, encoding=encoding)

    # --- Métadonnées utiles
    pages = None
    try:
        pages = len(result.document.pages)
    except Exception:
        pass

    meta = {
        "source_pdf": str(pdf_path),
        "output_text": str(text_path),
        "characters": len(text),
        "pages": pages,
        "used_ocr": do_ocr,
        "engine": "granite-docling",
    }

    if save_metadata:
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding=encoding,
        )

    return {
        "success": True,
        "text_path": str(text_path),
        "chars": len(text),
        "pages": pages,
        "used_ocr": do_ocr,
        "error": None,
    }


__all__ = ["pdf_to_text_granite"]


def convert(pdf_path: str | Path, txt_path: str | Path, **kwargs) -> Path:
    """Wrapper compatibilité: PDF -> TXT."""

    txt_path = Path(txt_path)
    res = pdf_to_text_granite(pdf_path, output_dir=txt_path.parent, **kwargs)
    if not isinstance(res, dict) or not res.get("success"):
        raise RuntimeError(res.get("error") if isinstance(res, dict) else "Conversion PDF → TXT échouée")

    produced = Path(res["text_path"])  # type: ignore[index]
    if produced.resolve() != txt_path.resolve():
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        produced.replace(txt_path)
    return txt_path


__all__.append("convert")
