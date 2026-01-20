from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging
import time

Logger = logging.getLogger(__name__)


def pdf_to_json_granite(
    pdf_path: str | Path,
    output_dir: Optional[str | Path] = None,
    *,
    preserve_formulas: bool = False,
    normalize: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Convertit un PDF en JSON structuré via Granite (IBM Docling).

    Responsabilité UNIQUE :
        PDF → JSON sémantique

    Aucun traitement image, OCR, LaTeX ou enrichissement ici.
    Cette fonction est le socle du pipeline.

    Args:
        pdf_path: chemin vers le PDF
        output_dir: dossier de sortie (défaut: result/json/)
        preserve_formulas: conserve les tokens <formula_not_decoded> si True
        normalize: nettoie et normalise légèrement la sortie
        verbose: logs détaillés

    Returns:
        dict contenant:
            - success: bool
            - json_path: str
            - pages: int
            - formulas_detected: int
            - figures_detected: int
            - raw_json: dict (chargé en mémoire)
    """

    t0 = time.time()
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    # --- sortie ---
    output_base = Path(output_dir or "result/json")
    output_base.mkdir(parents=True, exist_ok=True)
    json_path = output_base / f"{pdf_path.stem}.json"

    if verbose:
        Logger.info("PDF → JSON Granite")
        Logger.info("PDF: %s", pdf_path)
        Logger.info("OUT: %s", json_path)

    # --- Granite ---
    try:
        from .graniteIBM import granite
    except Exception as exc:
        raise RuntimeError("Granite / Docling non disponible") from exc

    try:
        produced = granite(
            input_path=str(pdf_path),
            output_format="json",
            preserve_formulas=preserve_formulas,
        )
        produced = Path(produced)
    except Exception as exc:
        raise RuntimeError(f"Granite a échoué: {exc}") from exc

    if not produced.exists():
        raise RuntimeError("Granite n'a pas produit de JSON")

    # --- chargement ---
    with open(produced, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # --- statistiques ---
    pages = data.get("meta", {}).get("page_count", None)

    formulas = 0
    figures = 0

    for block in data.get("main-text", []):
        if block.get("name") == "formula":
            formulas += 1
        elif block.get("name") in ("figure", "image"):
            figures += 1

    # --- normalisation minimale (non destructive) ---
    if normalize:
        for block in data.get("main-text", []):
            if block.get("name") == "formula":
                if "text" not in block or not block["text"]:
                    block["text"] = "<formula_not_decoded>"
            if "page" in block:
                block["page"] = int(block["page"])

    # --- méta Quantix ---
    data["_quantix"] = {
        "source_pdf": str(pdf_path),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": "granite-docling",
        "pages": pages,
        "formulas_detected": formulas,
        "figures_detected": figures,
        "elapsed_s": round(time.time() - t0, 2),
    }

    # --- écriture finale ---
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if verbose:
        Logger.info("OK – pages=%s formulas=%d figures=%d",
                    pages, formulas, figures)

    return {
        "success": True,
        "json_path": str(json_path),
        "pages": pages,
        "formulas_detected": formulas,
        "figures_detected": figures,
        "raw_json": data,
    }


__all__ = ["pdf_to_json_granite"]


def convert(pdf_path: str | Path, tex_path: str | Path, **kwargs) -> Path:
    """Conversion PDF -> LaTeX (best-effort) via PDF->Markdown->LaTeX.

    Cette implémentation évite de dépendre d'un export LaTeX direct.
    """

    from .pdf2markdown import pdf_to_markdown_granite
    from .markdown2latex import markdown_to_latex

    tex_path = Path(tex_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    md_path = pdf_to_markdown_granite(pdf_path, output_dir=tex_path.parent)
    body = markdown_to_latex(md_path)
    tex_path.write_text(body, encoding="utf-8")
    return tex_path


__all__.append("convert")
