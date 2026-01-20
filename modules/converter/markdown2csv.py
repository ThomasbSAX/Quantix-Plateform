from __future__ import annotations

from pathlib import Path
import csv
import re
from typing import List, Iterable, Tuple


# ==========================================================
# Utils
# ==========================================================

def _is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\s*[-:| ]+\s*", line))


def _split_markdown_row(line: str) -> List[str]:
    line = line.strip().strip("|")
    return [c.strip() for c in line.split("|")]


def _split_csv_like(line: str) -> List[str] | None:
    for sep in [",", ";", "\t"]:
        if sep in line:
            return [c.strip() for c in line.split(sep)]
    return None


def _normalize_rows(rows: List[List[str]]) -> List[List[str]]:
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    return [r + [""] * (width - len(r)) for r in rows]


# ==========================================================
# Detectors
# ==========================================================

def detect_markdown_tables(lines: List[str]) -> List[List[List[str]]]:
    tables = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                row = _split_markdown_row(lines[i])
                if row and not all(_is_separator(c) for c in row):
                    block.append(row)
                i += 1
            if len(block) >= 2:
                tables.append(_normalize_rows(block))
            continue
        i += 1
    return tables


def detect_list_tables(lines: List[str]) -> List[List[List[str]]]:
    tables = []
    buffer = []
    for line in lines:
        m = re.match(r"\s*[-*+]\s+(.*)", line)
        if m:
            cells = re.split(r"\s{2,}|\t", m.group(1))
            buffer.append([c.strip() for c in cells])
        else:
            if len(buffer) >= 2:
                tables.append(_normalize_rows(buffer))
            buffer = []
    if len(buffer) >= 2:
        tables.append(_normalize_rows(buffer))
    return tables


def detect_aligned_text_tables(lines: List[str]) -> List[List[List[str]]]:
    tables = []
    buffer = []
    for line in lines:
        # Évite de capturer des listes "- a  b" déjà gérées par detect_list_tables
        if re.match(r"\s*[-*+]\s+", line):
            if len(buffer) >= 2:
                tables.append(_normalize_rows(buffer))
            buffer = []
            continue

        if line.strip() and re.search(r"\s{2,}", line):
            buffer.append(re.split(r"\s{2,}", line.strip()))
        else:
            if len(buffer) >= 2:
                tables.append(_normalize_rows(buffer))
            buffer = []
    if len(buffer) >= 2:
        tables.append(_normalize_rows(buffer))
    return tables


def detect_csv_like_tables(lines: List[str]) -> List[List[List[str]]]:
    tables = []
    buffer = []
    for line in lines:
        row = _split_csv_like(line)
        if row:
            buffer.append(row)
        else:
            if len(buffer) >= 2:
                tables.append(_normalize_rows(buffer))
            buffer = []
    if len(buffer) >= 2:
        tables.append(_normalize_rows(buffer))
    return tables


def detect_inline_csv_like_tables(text: str) -> List[List[List[str]]]:
    """Détecte des tables CSV-like "en ligne".

    Cas typique: OCR d'image où toutes les lignes sont concaténées dans un seul
    paragraphe, ex: "S001,8.0,... S002,1.3,...".
    """

    if not text:
        return []

    # On travaille sur une version aplatie pour gérer les retours à la ligne d'OCR.
    flat = re.sub(r"\s+", " ", text).strip()
    if not flat or "," not in flat:
        return []

    # Heuristique: début de "ligne" = token (alpha/$/digit) + 3-4 chiffres, puis virgule, puis un chiffre.
    # Exemple attendu: S001,8.0  /  5002,1.3  /  $008, 2.0
    row_start = re.compile(r"(?:(?<=\s)|^)([A-Za-z\$\d]\d{3})\s*,\s*\d")
    starts = [m.start(1) for m in row_start.finditer(flat)]
    if len(starts) < 2:
        return []

    # Découpe en segments [start_i, start_{i+1})
    segments: List[str] = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(flat)
        seg = flat[s:e].strip()
        if seg:
            segments.append(seg)

    rows: List[List[str]] = []
    for seg in segments:
        row = _split_csv_like(seg)
        if row and len(row) >= 2:
            rows.append(row)

    if len(rows) < 2:
        return []

    return [_normalize_rows(rows)]


# ==========================================================
# Master extractor
# ==========================================================

def extract_tables_from_markdown(md_path: str | Path) -> List[List[List[str]]]:
    text = Path(md_path).read_text(encoding="utf-8")
    return extract_tables_from_text(text)


def extract_tables_from_text(text: str) -> List[List[List[str]]]:
    """Extrait des tableaux depuis du texte (Markdown ou texte semi-structuré).

    Objectif: factoriser la détection de tables pour les pipelines:
    - PDF -> Markdown (Granite/Docling) -> tables
    - Image -> Markdown (Docling) -> tables
    - Markdown -> CSV
    """

    lines = (text or "").splitlines()

    # Ordre important: on privilégie les tables explicites (pipe), puis listes,
    # puis colonnes alignées, puis CSV-like.
    raw: List[List[List[str]]] = []
    raw += detect_markdown_tables(lines)
    raw += detect_list_tables(lines)
    raw += detect_aligned_text_tables(lines)
    raw += detect_csv_like_tables(lines)
    raw += detect_inline_csv_like_tables(text or "")

    # Déduplication simple (évite des doubles détections entre heuristiques)
    seen: set[tuple[tuple[str, ...], ...]] = set()
    tables: List[List[List[str]]] = []
    for t in raw:
        key = tuple(tuple(str(c) for c in row) for row in t)
        if key in seen:
            continue
        seen.add(key)
        tables.append(t)

    return tables


# ==========================================================
# CSV Export
# ==========================================================

def export_tables_to_csv(
    tables: List[List[List[str]]],
    output_dir: str | Path,
    prefix: str = "table",
    delimiter: str = ",",
) -> List[Path]:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, table in enumerate(tables, start=1):
        path = output_dir / f"{prefix}_{i:03d}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerows(table)
        paths.append(path)

    return paths


__all__ = [
    "extract_tables_from_markdown",
    "extract_tables_from_text",
    "export_tables_to_csv",
]


def convert(md_path: str | Path, out_path: str | Path, **kwargs):
    """Wrapper compatibilité: Markdown -> CSV.

    Par défaut, on produit un fichier CSV unique (table 0), car c'est ce que
    l'UI attend le plus souvent.

    kwargs supportés:
    - delimiter: str (défaut ",")
    - table_index: int (défaut 0)
    - combine_tables: bool (défaut False) -> concatène toutes les tables dans un seul CSV
    - multi_tables: bool (défaut False) -> 1 CSV par table (retourne liste de fichiers)
    """

    md_path = Path(md_path)
    out_path = Path(out_path)

    delimiter = str(kwargs.get("delimiter", ","))
    table_index = int(kwargs.get("table_index", 0))
    combine_tables = bool(kwargs.get("combine_tables", False))
    multi_tables = bool(kwargs.get("multi_tables", False))

    tables = extract_tables_from_markdown(md_path)
    if not tables:
        raise RuntimeError("Aucun tableau détecté dans le Markdown")

    if table_index < 0:
        raise ValueError("table_index doit être >= 0")

    # Mode 1: multi-fichiers (1 CSV par table)
    if multi_tables and not combine_tables:
        output_dir = out_path.parent / f"{out_path.stem}_tables"
        return export_tables_to_csv(tables, output_dir=output_dir, prefix="table", delimiter=delimiter)

    # Mode 2: concat toutes les tables dans un seul CSV
    if combine_tables and len(tables) > 1:
        combined: List[List[str]] = []
        for idx, t in enumerate(tables):
            if idx:
                combined.append([])
            combined.extend(t)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerows(combined)
        return out_path

    # Mode 3 (défaut): une seule table (par index)
    if table_index >= len(tables):
        raise IndexError(
            f"Table {table_index} inexistante (tables détectées: {len(tables)})"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(tables[table_index])
    return out_path


__all__.append("convert")
