from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption


def pdf_to_markdown_granite(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    preserve_formulas: bool = False,
    extract_images: bool = False,
) -> Path:
    """
    Convertit un PDF en Markdown en utilisant Granite IBM (Docling).

    Note: conversion best-effort. Pour le moment, la préservation des formules et
    l'extraction/liaison des images ne sont pas garanties (fonctionnalité en développement).
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    output_dir = Path(output_dir or pdf_path.parent / "result" / "markdown")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_md = output_dir / f"{pdf_path.stem}.md"

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True

    if preserve_formulas:
        pipeline_options.do_formula_enrichment = True

    pipeline_options.generate_picture_images = extract_images
    pipeline_options.generate_page_images = False

    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }

    converter = DocumentConverter(format_options=format_options)
    result = converter.convert(str(pdf_path))

    markdown = result.document.export_to_markdown()

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(markdown)

    return output_md


def convert(pdf_path: str | Path, md_path: str | Path, **kwargs) -> Path:
    """Wrapper compatibilité: PDF -> Markdown."""

    md_path = Path(md_path)
    produced = pdf_to_markdown_granite(pdf_path, output_dir=md_path.parent, **kwargs)
    produced = Path(produced)
    if produced.resolve() != md_path.resolve():
        md_path.parent.mkdir(parents=True, exist_ok=True)
        produced.replace(md_path)
    return md_path


__all__ = ["pdf_to_markdown_granite", "convert"]
