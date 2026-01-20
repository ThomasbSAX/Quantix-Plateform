from pathlib import Path
import subprocess
import shutil
import logging
import traceback
from typing import Optional

Logger = logging.getLogger(__name__)


def _try_pdf2docx_pkg(pdf_path: Path, out_path: Path) -> bool:
    try:
        from pdf2docx import Converter

        cv = Converter(str(pdf_path))
        cv.convert(str(out_path))
        cv.close()
        return out_path.exists()
    except Exception:
        Logger.debug("pdf2docx package failed: %s", traceback.format_exc())
        return False


def _try_soffice(pdf_path: Path, out_path: Path, timeout: Optional[int] = None) -> bool:
    soffice = shutil.which("soffice")
    if not soffice:
        return False
    outdir = out_path.parent
    cmd = [soffice, "--headless", "--convert-to", "docx", "--outdir", str(outdir), str(pdf_path)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        return out_path.exists()
    except Exception:
        Logger.debug("soffice conversion failed: %s", traceback.format_exc())
        return False


def _try_pandoc(pdf_path: Path, out_path: Path, timeout: Optional[int] = None) -> bool:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        return False
    cmd = [pandoc, str(pdf_path), "-o", str(out_path)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        return out_path.exists()
    except Exception:
        Logger.debug("pandoc conversion failed: %s", traceback.format_exc())
        return False


def _try_ocr_to_docx(pdf_path: Path, out_path: Path, timeout: Optional[int] = None) -> bool:
    # best-effort OCR fallback using pdf2image + pytesseract + python-docx
    try:
        from pdf2image import convert_from_path
        import pytesseract
        from docx import Document
    except Exception:
        Logger.debug("OCR fallback requirements missing: %s", traceback.format_exc())
        return False

    try:
        images = convert_from_path(str(pdf_path))
        doc = Document()
        for img in images:
            text = pytesseract.image_to_string(img)
            for para in text.splitlines():
                doc.add_paragraph(para)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(out_path))
        return out_path.exists()
    except Exception:
        Logger.debug("OCR to DOCX failed: %s", traceback.format_exc())
        return False


def pdf_to_docx(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    backend: Optional[str] = None,
    timeout: Optional[int] = 300,
    on_error: str = "raise",
    error_log: Optional[str] = None,
) -> Optional[Path]:
    """Convertit un PDF en DOCX en essayant plusieurs backends.

    - `backend`: force un backend parmi 'pdf2docx','soffice','pandoc','ocr'.
    - `on_error`: 'raise' (par défaut), 'warn' (log + return None), 'ignore' (silent None).

    Retourne le `Path` du fichier DOCX en cas de succès, sinon None si `on_error` != 'raise'.
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        msg = f"Fichier PDF introuvable: {pdf_path}"
        if on_error == "raise":
            raise FileNotFoundError(pdf_path)
        Logger.warning(msg)
        if error_log:
            Path(error_log).write_text(msg)
        return None

    output_dir = Path(output_dir or pdf_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_docx = output_dir / (pdf_path.stem + ".docx")

    errors: list[str] = []

    backends = [backend] if backend else ["pdf2docx", "soffice", "pandoc", "ocr"]
    for be in backends:
        if be is None:
            continue
        be = be.lower()
        try:
            ok = False
            if be == "pdf2docx":
                ok = _try_pdf2docx_pkg(pdf_path, output_docx)
            elif be == "soffice":
                ok = _try_soffice(pdf_path, output_docx, timeout=timeout)
            elif be == "pandoc":
                ok = _try_pandoc(pdf_path, output_docx, timeout=timeout)
            elif be == "ocr":
                ok = _try_ocr_to_docx(pdf_path, output_docx, timeout=timeout)
            else:
                # unknown backend string
                continue

            if ok:
                return output_docx
            else:
                errors.append(f"Backend {be} failed for {pdf_path}")
        except Exception as exc:
            tb = traceback.format_exc()
            errors.append(f"Backend {be} exception: {exc}\n{tb}")

    msg = f"Tous les backends ont échoué pour convertir {pdf_path}: {errors}"
    if on_error == "raise":
        raise RuntimeError(msg)
    Logger.warning(msg)
    if error_log:
        try:
            Path(error_log).write_text("\n".join(errors))
        except Exception:
            Logger.warning("Impossible d'écrire le fichier d'erreurs: %s", error_log)
    return None


def convert(pdf_path: str | Path, docx_path: str | Path, **kwargs):
    """Wrapper compatibilité: PDF -> DOCX.

    Écrit un fichier DOCX à l'emplacement demandé si possible.
    """

    pdf_path = Path(pdf_path)
    docx_path = Path(docx_path)
    produced = pdf_to_docx(pdf_path, output_dir=docx_path.parent, **kwargs)
    if produced is None:
        raise RuntimeError("Conversion PDF → DOCX échouée")
    produced = Path(produced)
    if produced.resolve() != docx_path.resolve():
        docx_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            produced.replace(docx_path)
        except Exception:
            # fallback copy
            docx_path.write_bytes(produced.read_bytes())
    return docx_path


__all__ = ["pdf_to_docx", "convert"]
