"""
Conversion DOCX -> Markdown robuste.

Stratégie :
- docling (IBM Granite) si dispo
- fallback python-docx avec :
  * styles inline sûrs (bold/italic/strike/underline)
  * listes (puces / numérotées)
  * tableaux Markdown valides
  * images extraites et insérées à la bonne position
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Any, Union, Optional
import logging
import re

Logger = logging.getLogger(__name__)


# =========================
# Dépendances
# =========================

def _ensure_docx() -> None:
    try:
        import docx  # type: ignore
    except Exception as exc:
        raise ImportError(
            "python-docx requis. Installez via `pip install python-docx`"
        ) from exc


# =========================
# Markdown helpers
# =========================

def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _format_run(run) -> str:
    text = run.text or ""
    if not text:
        return ""

    pre, post = "", ""

    if run.bold:
        pre += "**"
        post = "**" + post
    if run.italic:
        pre += "*"
        post = "*" + post
    if getattr(run.font, "strike", False):
        pre += "~~"
        post = "~~" + post
    if getattr(run, "underline", False):
        pre += "<u>"
        post = "</u>" + post

    return f"{pre}{text}{post}"


def _paragraph_text(p) -> str:
    return "".join(_format_run(r) for r in p.runs).strip()


# =========================
# Images
# =========================

def _extract_images(doc, images_dir: Path) -> List[Path]:
    images_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []

    for rel in doc.part.rels.values():
        try:
            part = rel.target_part
            if part.content_type.startswith("image/"):
                ext = part.content_type.split("/")[-1]
                name = f"image_{len(out)+1}.{ext}"
                path = images_dir / name
                path.write_bytes(part.blob)
                out.append(path)
        except Exception:
            continue
    return out


def _paragraph_has_image(p) -> bool:
    for r in p.runs:
        try:
            if r._r.xpath(".//w:drawing"):
                return True
        except Exception:
            pass
    return False


# =========================
# Bloc iteration (ordre DOCX)
# =========================

def _iter_blocks(doc) -> Iterable[Any]:
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl
    from docx.text.paragraph import Paragraph
    from docx.table import Table

    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


# =========================
# Conversion principale
# =========================

def convert(
    docx_path: Union[str, Path],
    md_path: Union[str, Path],
    *,
    encoding: str = "utf-8",
    use_docling: bool = True,
    extract_images: bool = True,
    convert_tables: bool = True,
) -> Path:
    docx_path = Path(docx_path)
    md_path = Path(md_path)

    if not docx_path.exists():
        raise FileNotFoundError(docx_path)

    # -------------------------
    # docling / Granite
    # -------------------------
    if use_docling:
        try:
            from docling.document_converter import DocumentConverter  # type: ignore
            from docling.datamodel.base_models import InputFormat  # type: ignore

            conv = DocumentConverter()
            res = conv.convert(str(docx_path), input_format=InputFormat.DOCX)
            md = res.document.export_to_markdown()
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(md, encoding=encoding)
            return md_path
        except Exception as exc:
            Logger.warning("docling échoué, fallback python-docx: %s", exc)

    # -------------------------
    # python-docx fallback
    # -------------------------
    _ensure_docx()
    from docx import Document  # type: ignore
    from docx.text.paragraph import Paragraph
    from docx.table import Table

    doc = Document(str(docx_path))

    images_dir = md_path.parent / f"{md_path.stem}_images"
    images = _extract_images(doc, images_dir) if extract_images else []
    img_iter = iter(images)

    out: List[str] = []
    url_re = re.compile(r"(https?://\S+)")

    for block in _iter_blocks(doc):

        # -------- Paragraph --------
        if isinstance(block, Paragraph):
            style = (block.style.name or "").lower()
            text = _paragraph_text(block)

            # Image anchor
            if extract_images and _paragraph_has_image(block):
                try:
                    img = next(img_iter)
                    out.append(f"![]({images_dir.name}/{img.name})\n")
                    continue
                except StopIteration:
                    pass

            # Headings
            if style.startswith("heading"):
                try:
                    lvl = int(style.split()[-1])
                except Exception:
                    lvl = 2
                lvl = min(max(lvl, 1), 6)
                out.append("#" * lvl + " " + text + "\n")
                continue

            # Lists
            if "list" in style or "bullet" in style:
                out.append(f"- {text}\n")
                continue
            if "number" in style:
                out.append(f"1. {text}\n")
                continue

            # Plain text
            if text:
                text = url_re.sub(r"[\1](\1)", text)
                out.append(text + "\n\n")

        # -------- Table --------
        elif convert_tables and isinstance(block, Table):
            rows: List[List[str]] = []
            for r in block.rows:
                row = []
                for c in r.cells:
                    cell_text = " ".join(
                        _md_escape(_paragraph_text(p))
                        for p in c.paragraphs
                        if _paragraph_text(p)
                    )
                    row.append(cell_text)
                rows.append(row)

            if rows:
                header = rows[0]
                out.append("| " + " | ".join(header) + " |\n")
                out.append("| " + " | ".join(["---"] * len(header)) + " |\n")
                for r in rows[1:]:
                    out.append("| " + " | ".join(r) + " |\n")
                out.append("\n")

    md = "".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding=encoding)
    return md_path


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("docx2markdown")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--no-docling", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    convert(
        args.input,
        args.output,
        use_docling=not args.no_docling,
        extract_images=not args.no_images,
        convert_tables=not args.no_tables,
    )
