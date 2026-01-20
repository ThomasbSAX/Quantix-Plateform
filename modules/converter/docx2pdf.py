"""
Conversion DOCX -> PDF robuste avec fallback automatique.

Backends supportés :
- docx2pdf (Python)
- pandoc (CLI)
- libreoffice / soffice (headless)

API :
- convert(input, output=None, backend='auto', overwrite=True, timeout=None)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union, List

Logger = logging.getLogger(__name__)


# =========================
# Exceptions
# =========================

class DOCX2PDFError(RuntimeError):
    pass


# =========================
# Utils
# =========================

def _which(*names: str) -> Optional[str]:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


# =========================
# Backends
# =========================

def _run_docx2pdf(
    input_path: Path,
    output_path: Path,
) -> Path:
    try:
        import docx2pdf  # type: ignore
    except Exception as exc:
        raise DOCX2PDFError("docx2pdf non installé") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    docx2pdf.convert(str(input_path), str(output_path))

    if not output_path.exists():
        raise DOCX2PDFError("docx2pdf n’a pas produit le PDF attendu")

    return output_path


def _run_pandoc(
    input_path: Path,
    output_path: Path,
    *,
    timeout: Optional[int],
) -> Path:
    exe = _which("pandoc")
    if not exe:
        raise DOCX2PDFError("pandoc introuvable")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, str(input_path), "-o", str(output_path)]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        raise DOCX2PDFError(f"pandoc a échoué:\n{err}") from exc

    if not output_path.exists():
        raise DOCX2PDFError("pandoc n’a pas produit le PDF attendu")

    return output_path


def _run_libreoffice(
    input_path: Path,
    output_path: Path,
    *,
    timeout: Optional[int],
) -> Path:
    exe = _which("soffice", "libreoffice")
    if not exe:
        raise DOCX2PDFError("LibreOffice (soffice/libreoffice) introuvable")

    with tempfile.TemporaryDirectory(prefix="docx2pdf_lo_") as tmp:
        tmpdir = Path(tmp)
        cmd = [
            exe,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(tmpdir),
            str(input_path),
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.decode(errors="ignore") if exc.stderr else ""
            raise DOCX2PDFError(f"LibreOffice a échoué:\n{err}") from exc

        produced = tmpdir / (input_path.stem + ".pdf")
        if not produced.exists():
            raise DOCX2PDFError("LibreOffice n’a pas produit le PDF attendu")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        produced.replace(output_path)

    return output_path


# =========================
# API publique
# =========================

def convert(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    backend: str = "auto",
    overwrite: bool = True,
    timeout: Optional[int] = None,
) -> Path:
    """
    Convertit un DOCX en PDF.

    backend :
    - 'auto'        : docx2pdf → pandoc → libreoffice
    - 'docx2pdf'
    - 'pandoc'
    - 'libreoffice'
    """

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    out = Path(output_path) if output_path else input_path.with_suffix(".pdf")

    if out.exists() and not overwrite:
        raise FileExistsError(out)

    backend = backend.lower()
    errors: List[str] = []

    def _try(fn, name: str):
        try:
            return fn()
        except Exception as exc:
            Logger.debug("%s failed: %s", name, exc)
            errors.append(f"{name}: {exc}")
            return None

    # ordre strict et logique
    if backend == "auto":
        for name, fn in [
            ("docx2pdf", lambda: _run_docx2pdf(input_path, out)),
            ("pandoc", lambda: _run_pandoc(input_path, out, timeout=timeout)),
            ("libreoffice", lambda: _run_libreoffice(input_path, out, timeout=timeout)),
        ]:
            res = _try(fn, name)
            if res:
                return res

    elif backend == "docx2pdf":
        return _run_docx2pdf(input_path, out)

    elif backend == "pandoc":
        return _run_pandoc(input_path, out, timeout=timeout)

    elif backend == "libreoffice":
        return _run_libreoffice(input_path, out, timeout=timeout)

    else:
        raise ValueError(f"backend invalide: {backend}")

    raise DOCX2PDFError(
        "Aucune méthode de conversion fonctionnelle.\n"
        + "\n".join(errors)
    )


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("docx2pdf")
    ap.add_argument("input", help="chemin .docx")
    ap.add_argument("output", nargs="?", help="chemin .pdf (optionnel)")
    ap.add_argument("--backend", choices=["auto", "docx2pdf", "pandoc", "libreoffice"], default="auto")
    ap.add_argument("--no-overwrite", action="store_true")
    ap.add_argument("--timeout", type=int)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    pdf = convert(
        args.input,
        args.output,
        backend=args.backend,
        overwrite=not args.no_overwrite,
        timeout=args.timeout,
    )
    print(pdf)
