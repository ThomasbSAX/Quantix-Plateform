from pathlib import Path
from typing import List, Optional
import logging
import traceback
import shutil
import subprocess

Logger = logging.getLogger(__name__)


# =========================
# Backend PyMuPDF (fitz)
# =========================

def _pdf_to_images_fitz(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int,
    image_format: str,
    first_page: Optional[int],
    last_page: Optional[int],
    grayscale: bool,
) -> List[Path]:
    import fitz  # PyMuPDF

    output_dir.mkdir(parents=True, exist_ok=True)

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    images: List[Path] = []

    doc = fitz.open(pdf_path)
    try:
        total = doc.page_count
        start = max(0, (first_page - 1) if first_page else 0)
        end = min(total, last_page if last_page else total)

        for i in range(start, end):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            out = output_dir / f"page_{i+1:04d}.{image_format}"
            if grayscale:
                from PIL import Image
                import io

                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
                img.save(out)
            else:
                pix.save(out)

            images.append(out)

    finally:
        doc.close()

    return images


# =========================
# Backend Poppler (pdftoppm)
# =========================

def _pdf_to_images_poppler(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int,
    image_format: str,
    first_page: Optional[int],
    last_page: Optional[int],
    timeout: Optional[int],
) -> List[Path]:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise FileNotFoundError("pdftoppm introuvable")

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "page"

    args = [
        pdftoppm,
        "-r", str(dpi),
        f"-{image_format}",
        str(pdf_path),
        str(prefix),
    ]

    if first_page:
        args += ["-f", str(first_page)]
    if last_page:
        args += ["-l", str(last_page)]

    subprocess.run(
        args,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
    )

    images = sorted(output_dir.glob(f"page-*.{image_format}"))

    # normalisation des noms
    normalized: List[Path] = []
    for idx, img in enumerate(images, start=1):
        out = output_dir / f"page_{idx:04d}.{image_format}"
        img.rename(out)
        normalized.append(out)

    return normalized


# =========================
# API principale
# =========================

def pdf_to_images(
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 300,
    image_format: str = "png",
    backend: Optional[str] = None,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
    grayscale: bool = False,
    timeout: Optional[int] = None,
    on_error: str = "raise",
    error_log: Optional[str] = None,
) -> Optional[List[Path]]:
    """
    Convertit un PDF en images avec garantie 1 page = 1 image.
    """

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        msg = f"PDF introuvable: {pdf_path}"
        if on_error == "raise":
            raise FileNotFoundError(msg)
        Logger.warning(msg)
        return None

    errors: list[str] = []

    order = [backend] if backend else ["fitz", "poppler"]

    for be in order:
        try:
            if be == "fitz":
                imgs = _pdf_to_images_fitz(
                    pdf_path,
                    output_dir,
                    dpi=dpi,
                    image_format=image_format,
                    first_page=first_page,
                    last_page=last_page,
                    grayscale=grayscale,
                )
                if imgs:
                    return imgs

            elif be == "poppler":
                imgs = _pdf_to_images_poppler(
                    pdf_path,
                    output_dir,
                    dpi=dpi,
                    image_format=image_format,
                    first_page=first_page,
                    last_page=last_page,
                    timeout=timeout,
                )
                if imgs:
                    return imgs

            else:
                errors.append(f"Backend inconnu: {be}")

        except Exception as exc:
            errors.append(f"{be} → {exc}")
            Logger.debug(traceback.format_exc())

    msg = f"Échec conversion PDF → images: {errors}"
    if on_error == "raise":
        raise RuntimeError(msg)

    Logger.warning(msg)
    if error_log:
        Path(error_log).write_text("\n".join(errors))

    return None


def convert(pdf_path: str | Path, out_path: str | Path, **kwargs):
    """Wrapper compatibilité: PDF -> images.

    `out_path` sert de préfixe; les images sont écrites dans un dossier sibling.
    Retourne la liste des images produites.
    """

    out_path = Path(out_path)
    output_dir = out_path.parent / f"{out_path.stem}_images"
    return pdf_to_images(pdf_path, output_dir=output_dir, **kwargs)


__all__ = ["pdf_to_images", "convert"]
