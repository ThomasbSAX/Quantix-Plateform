from pathlib import Path
import subprocess
import shutil
import logging

Logger = logging.getLogger(__name__)


def pptx_to_pdf(
    pptx_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    timeout: int | None = 120,
) -> Path:
    """
    Convertit un PPTX en PDF avec LibreOffice.

    CONTRAT :
    - 1 slide PowerPoint = 1 page PDF
    - respect du layout, des polices et des images
    - aucune rasterisation inutile

    Dépendance :
    - LibreOffice / soffice doit être disponible dans le PATH
    """

    pptx_path = Path(pptx_path).resolve()
    if not pptx_path.exists():
        raise FileNotFoundError(pptx_path)

    soffice = shutil.which("soffice")
    if soffice is None:
        raise RuntimeError("LibreOffice (soffice) introuvable dans le PATH")

    output_dir = Path(output_dir or pptx_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        soffice,
        "--headless",
        "--nologo",
        "--nofirststartwizard",
        "--convert-to", "pdf",
        "--outdir", str(output_dir),
        str(pptx_path),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timeout conversion PPTX → PDF") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Erreur LibreOffice lors de la conversion PPTX → PDF") from exc

    out_pdf = output_dir / f"{pptx_path.stem}.pdf"
    if not out_pdf.exists():
        raise RuntimeError("PDF non généré après conversion PPTX")

    return out_pdf


__all__ = ["pptx_to_pdf"]


def convert(pptx_path: str | Path, pdf_path: str | Path, **kwargs) -> Path:
    """Wrapper compatibilité: PPTX -> PDF."""

    pdf_path = Path(pdf_path)
    produced = pptx_to_pdf(pptx_path, output_dir=pdf_path.parent, **kwargs)
    produced = Path(produced)
    if produced.resolve() != pdf_path.resolve():
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        produced.replace(pdf_path)
    return pdf_path


__all__.append("convert")
