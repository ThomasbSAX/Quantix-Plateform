from __future__ import annotations

from pathlib import Path
import logging
import traceback
from typing import Optional

try:
    from PIL import Image, UnidentifiedImageError, ExifTags
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    UnidentifiedImageError = Exception  # type: ignore
    ExifTags = None  # type: ignore

Logger = logging.getLogger(__name__)


class ImageToPDFError(RuntimeError):
    pass


# =========================
# EXIF orientation
# =========================

def _apply_exif_orientation(img: "Image.Image") -> "Image.Image":
    """
    Applique l’orientation EXIF sans modifier la résolution.
    """
    if ExifTags is None:
        return img

    try:
        exif = img._getexif()
        if not exif:
            return img

        orientation_key = next(
            (k for k, v in ExifTags.TAGS.items() if v == "Orientation"),
            None,
        )
        if orientation_key is None:
            return img

        orientation = exif.get(orientation_key)
        if orientation == 3:
            return img.transpose(Image.ROTATE_180)
        if orientation == 6:
            return img.transpose(Image.ROTATE_270)
        if orientation == 8:
            return img.transpose(Image.ROTATE_90)

    except Exception:
        pass

    return img


# =========================
# Image -> PDF (1 image)
# =========================

def image_to_pdf(
    image: str | Path,
    output_pdf: str | Path,
    *,
    dpi: Optional[int] = None,        # DPI metadata only (no resampling)
    quality: Optional[int] = None,    # PDF encoder (Pillow)
    optimize: bool = True,
    on_error: str = "raise",          # "raise" | "warn" | "ignore"
    error_log: Optional[str | Path] = None,
) -> bool:
    """
    Convertit UNE image en PDF, sans aucune perte de résolution.

    - AUCUN resize
    - pixels conservés 1:1 (4K strictement préservée)
    - orientation EXIF corrigée
    - écriture atomique
    """

    image_path = Path(image)
    output_path = Path(output_pdf)

    if Image is None:
        msg = "Pillow non installé. Installez `Pillow`."
        if on_error == "raise":
            raise ImageToPDFError(msg)
        Logger.warning(msg)
        return False

    if not image_path.exists():
        msg = f"Image introuvable: {image_path}"
        if on_error == "raise":
            raise FileNotFoundError(image_path)
        Logger.warning(msg)
        if error_log:
            Path(error_log).write_text(msg)
        return False

    try:
        img = Image.open(image_path)
    except UnidentifiedImageError as exc:
        msg = f"Fichier non-image ou corrompu: {image_path}"
        if on_error == "raise":
            raise ImageToPDFError(msg) from exc
        Logger.warning(msg)
        return False
    except Exception as exc:
        msg = f"Erreur à l’ouverture de l’image {image_path}: {exc}"
        if on_error == "raise":
            raise ImageToPDFError(msg) from exc
        Logger.exception(msg)
        return False

    try:
        # Orientation EXIF
        img = _apply_exif_orientation(img)

        # PDF Pillow requiert RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_kwargs = {"save_all": False}
        if optimize:
            save_kwargs["optimize"] = True
        if quality is not None:
            save_kwargs["quality"] = int(quality)
        if dpi is not None:
            save_kwargs["dpi"] = (int(dpi), int(dpi))

        # écriture atomique
        tmp = output_path.with_suffix(".pdf.tmp")
        img.save(tmp, format="PDF", **save_kwargs)
        tmp.replace(output_path)

    except Exception as exc:
        msg = f"Erreur lors de la génération du PDF {output_path}: {exc}"
        Logger.exception(msg)
        if error_log:
            Path(error_log).write_text(msg + "\n" + traceback.format_exc())
        if on_error == "raise":
            raise ImageToPDFError(msg) from exc
        return False
    finally:
        try:
            img.close()
        except Exception:
            pass

    return True


__all__ = ["image_to_pdf", "ImageToPDFError"]
