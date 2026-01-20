"""Routeur universel de conversion de formats.

But:
- Fournir un point d'entrée unique côté Flask.
- Garder les dépendances lourdes optionnelles (imports seulement à l'usage).
"""

from __future__ import annotations

import importlib
import inspect
import time
from pathlib import Path
from typing import Any, Optional

# Table de correspondance (source, cible) -> (module, fonction)
# À enrichir dynamiquement si besoin
CONVERSION_MAP = {
    ("csv", "pdf"): ("csv2pdf", "convert"),
    ("csv", "xlsx"): ("csv2xlsx", "convert"),
    ("csv", "json"): ("csv2json", "convert"),
    ("csv", "txt"): ("csv2txt", "convert"),
    ("csv", "tsv"): ("csv2tsv", "convert"),
    ("csv", "docx"): ("csv2docx", "convert"),
    ("json", "csv"): ("json2csv", "convert"),
    ("json", "xlsx"): ("json2xlsx", "convert"),
    ("xlsx", "csv"): ("xlsx2csv", "convert"),
    ("xlsx", "json"): ("xlsx2json", "convert"),
    ("tsv", "csv"): ("tsv2csv", "convert"),
    ("txt", "docx"): ("txt2docx", "txt_to_docx"),
    ("txt", "pdf"): ("txt2pdf", "txt_to_pdf"),
    ("txt", "csv"): ("txt2csv", "txt_to_csv"),
    ("docx", "pdf"): ("docx2pdf", "convert"),
    ("docx", "markdown"): ("docx2markdown", "convert"),
    ("pdf", "csv"): ("pdf2csv", "convert"),
    ("pdf", "docx"): ("pdf2docx", "convert"),
    ("pdf", "image"): ("pdf2image", "convert"),
    ("pdf", "json"): ("pdf2json", "convert"),
    ("pdf", "latex"): ("pdf2latex", "convert"),
    ("pdf", "markdown"): ("pdf2markdown", "convert"),
    ("pdf", "txt"): ("pdf2txt", "convert"),
    ("pptx", "pdf"): ("pptx2pdf", "convert"),
    ("pptx", "txt"): ("pptx2txt", "convert"),
    ("pptx", "markdown"): ("pptx2markdown", "convert"),
    ("pptx", "image"): ("pptx2image", "pptx_to_images"),
    ("markdown", "txt"): ("markdown2txt", "markdown2txt"),
    ("markdown", "csv"): ("markdown2csv", "convert"),
    ("markdown", "latex"): ("markdown2latex", "convert"),
    ("markdown", "pptx"): ("markdown2pptx", "convert"),
    ("png", "csv"): ("image2csv", "convert"),
    ("jpg", "csv"): ("image2csv", "convert"),
    ("png", "markdown"): ("image2markdown", "convert"),
    ("jpg", "markdown"): ("image2markdown", "convert"),

    # Conversions image -> image (Pillow)
    ("png", "jpg"): ("image2image", "image_to_image"),
    ("jpg", "png"): ("image2image", "image_to_image"),
    # ... à compléter selon besoins ...
}


def _normalize_format(fmt: Optional[str]) -> Optional[str]:
    if fmt is None:
        return None
    s = str(fmt).strip().lower().lstrip(".")
    # alias UI
    if s == "excel":
        return "xlsx"
    if s == "md":
        return "markdown"
    if s == "xls":
        return "xlsx"
    if s == "jpeg":
        return "jpg"
    return s


def _default_output_path(src_path: Path, dst_format: str, *, output_dir: Optional[Path] = None) -> Path:
    out_dir = output_dir or src_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    nonce = str(time.time_ns())[-6:]
    return out_dir / f"{src_path.stem}_converted_{ts}_{nonce}.{dst_format}"


def _build_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for (a, b) in CONVERSION_MAP.keys():
        graph.setdefault(str(a), set()).add(str(b))
    return graph


def _find_path(src_fmt: str, dst_fmt: str, *, max_steps: int = 3) -> Optional[list[str]]:
    """Retourne une liste de formats [src, ..., dst] si un chemin existe."""
    if src_fmt == dst_fmt:
        return [src_fmt]

    graph = _build_graph()
    queue: list[tuple[str, list[str]]] = [(src_fmt, [src_fmt])]
    seen: set[str] = {src_fmt}

    while queue:
        node, path = queue.pop(0)
        if len(path) - 1 >= max_steps:
            continue
        for nxt in sorted(graph.get(node, set())):
            if nxt == dst_fmt:
                return path + [nxt]
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, path + [nxt]))
    return None


def _call_converter(func, src_path: Path, out_path: Path, **kwargs: Any):
    """Appelle un convertisseur malgré des signatures hétérogènes."""
    # 1) appel standard (src, out)
    try:
        return func(str(src_path), str(out_path), **kwargs)
    except TypeError:
        pass

    # 2) tentative par signature (certains attendent output_dir)
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
    except Exception:
        params = []

    if params:
        # Si on voit output_dir/out_dir, on passe le dossier au lieu d'un chemin fichier.
        out_dir = out_path
        if out_path.suffix:
            out_dir = out_path.parent / out_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        if "output_dir" in params:
            return func(str(src_path), str(out_dir), output_dir=str(out_dir), **kwargs)
        if "out_dir" in params:
            return func(str(src_path), str(out_dir), out_dir=str(out_dir), **kwargs)
        if "output_folder" in params:
            return func(str(src_path), str(out_dir), output_folder=str(out_dir), **kwargs)

        # Certains wrappers utilisent out_path comme keyword
        if "out_path" in params:
            return func(str(src_path), out_path=str(out_path), **kwargs)
        if "output_path" in params:
            return func(str(src_path), output_path=str(out_path), **kwargs)

    # 3) dernier recours: re-tenter sans kwargs
    return func(str(src_path), str(out_path))


def _normalize_result(result: Any, *, out_path: Path) -> Any:
    """Normalise le retour des convertisseurs.

    - bool/None: on retourne out_path si le fichier existe
    - str/Path: Path
    - dict/list: inchangé (multi-sorties)
    """
    if result is None:
        if out_path.exists():
            return out_path
        return None

    if isinstance(result, bool):
        if result and out_path.exists():
            return out_path
        if result:
            return out_path
        raise RuntimeError("La conversion a échoué (retour False)")

    if isinstance(result, (str, Path)):
        return Path(result)

    # Certains convertisseurs renvoient toujours une liste (même pour 1 fichier)
    if isinstance(result, (list, tuple)):
        if len(result) == 1 and isinstance(result[0], (str, Path)):
            return Path(result[0])
        return result

    return result


def convert_any_to_any(
    src_path: str | Path,
    *,
    src_format: Optional[str] = None,
    dst_format: str,
    output_dir: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    **kwargs: Any,
):
    """Convertit un fichier d'un format vers un autre.

    Retour:
    - généralement: `Path` du fichier de sortie
    - parfois: `dict` ou `list[Path]` selon le convertisseur (multi-sorties)
    """

    src_path = Path(src_path)
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    src_fmt = _normalize_format(src_format) or _normalize_format(src_path.suffix)  # type: ignore[arg-type]
    if src_fmt and src_fmt.startswith("."):
        src_fmt = src_fmt[1:]

    dst_fmt = _normalize_format(dst_format)
    if not dst_fmt:
        raise ValueError("Format cible manquant")

    # No-op si même format
    if src_fmt == dst_fmt:
        return src_path

    out_dir = Path(output_dir) if output_dir else None

    def _run_edge(a: str, b: str, in_path: Path, out_path_local: Path):
        module_name, func_name = CONVERSION_MAP[(a, b)]
        module_path = f"modules.converter.{module_name}"
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
        result = _call_converter(func, in_path, out_path_local, **kwargs)
        return _normalize_result(result, out_path=out_path_local)

    key = (str(src_fmt), str(dst_fmt))
    # Conversion directe
    if key in CONVERSION_MAP:
        out_path = Path(output_path) if output_path else _default_output_path(src_path, str(dst_fmt), output_dir=out_dir)
        return _run_edge(str(src_fmt), str(dst_fmt), src_path, out_path)

    # Fallback: conversion en plusieurs étapes (ex: docx->markdown->csv)
    path = _find_path(str(src_fmt), str(dst_fmt), max_steps=3)
    if not path or len(path) < 2:
        supported = ", ".join(sorted({f"{a}->{b}" for (a, b) in CONVERSION_MAP.keys()}))
        raise ValueError(f"Conversion {src_fmt} -> {dst_fmt} non supportée. Supportées: {supported}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    tmp_dir = (out_dir or src_path.parent) / "_tmp_conversions" / f"{src_path.stem}_{ts}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    current = src_path
    intermediates: list[Path] = []
    last_result: Any = None

    for idx in range(len(path) - 1):
        a = path[idx]
        b = path[idx + 1]
        is_last = idx == (len(path) - 2)

        if is_last:
            out_path_step = Path(output_path) if output_path else _default_output_path(src_path, str(dst_fmt), output_dir=out_dir)
        else:
            out_path_step = _default_output_path(current, str(b), output_dir=tmp_dir)
            intermediates.append(out_path_step)

        last_result = _run_edge(a, b, current, out_path_step)

        # Si on reçoit une multi-sortie au milieu, on ne peut pas enchaîner proprement.
        if not is_last and isinstance(last_result, (dict, list, tuple)):
            raise RuntimeError(f"Conversion multi-sorties sur étape {a}->{b}, enchaînement impossible")

        if isinstance(last_result, Path):
            current = last_result
        else:
            # si le convertisseur renvoie autre chose, on stoppe
            break

    # Cleanup best-effort des intermédiaires
    for p in intermediates:
        try:
            if p.exists() and p.is_file():
                p.unlink()
        except Exception:
            pass
    try:
        # on supprime le dossier si vide
        if tmp_dir.exists() and tmp_dir.is_dir() and not any(tmp_dir.rglob("*")):
            tmp_dir.rmdir()
    except Exception:
        pass

    return last_result


__all__ = ["CONVERSION_MAP", "convert_any_to_any"]
