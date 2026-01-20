from __future__ import annotations

from pathlib import Path
import logging
import traceback
from typing import Optional

try:
    from PIL import Image, UnidentifiedImageError, ImageSequence, ExifTags
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    UnidentifiedImageError = Exception  # type: ignore
    ImageSequence = None  # type: ignore
    ExifTags = None  # type: ignore

Logger = logging.getLogger(__name__)


class ImageConversionError(RuntimeError):
    pass


# =========================
# Utils
# =========================

def _handle_error(
    msg: str,
    *,
    exc: Optional[Exception],
    on_error: str,
    error_log: Optional[str | Path],
) -> bool:
    if exc:
        Logger.exception(msg)
    else:
        Logger.warning(msg)

    if error_log:
        try:
            Path(error_log).write_text(
                msg + ("\n\n" + traceback.format_exc() if exc else ""),
                encoding="utf-8",
            )
        except Exception:
            Logger.warning("Impossible d’écrire error_log: %s", error_log)

    if on_error == "raise":
        raise ImageConversionError(msg) from exc
    return False


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    """
    Applique l’orientation EXIF (évite images tournées après conversion).
    """
    if ExifTags is None:
        return img
    try:
        exif = img.getexif()
        if not exif:
            return img
        orientation_key = next(
            k for k, v in ExifTags.TAGS.items() if v == "Orientation"
        )
        orientation = exif.get(orientation_key)
        if orientation == 3:
            return img.rotate(180, expand=True)
        if orientation == 6:
            return img.rotate(270, expand=True)
        if orientation == 8:
            return img.rotate(90, expand=True)
    except Exception:
        pass
    return img


def _atomic_save(img: Image.Image, out: Path, **kwargs) -> None:
    """
    Écriture atomique pour éviter fichiers partiels en cas d’erreur.
    """
    # Important: Pillow déduit le format à partir de l'extension.
    # Un suffixe ".jpg.tmp" casserait cette déduction (extension finale ".tmp").
    tmp = out.with_suffix(".tmp" + out.suffix)
    img.save(tmp, **kwargs)
    tmp.replace(out)


# =========================
# API principale
# =========================

def image_to_image(
    input_image: str | Path,
    output_image: str | Path,
    *,
    mode: Optional[str] = None,           # "RGB", "RGBA", "L", etc.
    quality: Optional[int] = None,        # JPEG/WEBP: 0–100
    optimize: bool = True,
    preserve_exif: bool = True,
    allow_animated: bool = True,
    on_error: str = "raise",               # "raise" | "warn" | "ignore"
    error_log: Optional[str | Path] = None,
) -> bool:
    """
    Convertit une image de manière robuste.

    - applique orientation EXIF
    - gère GIF/APNG animés
    - écriture atomique
    - erreurs contrôlées
    """

    input_path = Path(input_image)
    output_path = Path(output_image)

    # Defaults sûrs par format de sortie
    # JPEG ne supporte pas l'alpha: on force RGB si non précisé.
    if mode is None and output_path.suffix.lower() in (".jpg", ".jpeg"):
        mode = "RGB"

    if Image is None:
        return _handle_error(
            "Pillow non installé. Installez `Pillow`.",
            exc=None,
            on_error=on_error,
            error_log=error_log,
        )

    if not input_path.exists():
        return _handle_error(
            f"Fichier d’entrée introuvable: {input_path}",
            exc=None,
            on_error=on_error,
            error_log=error_log,
        )

    try:
        img = Image.open(input_path)
    except UnidentifiedImageError as exc:
        return _handle_error(
            f"Fichier non-image ou corrompu: {input_path}",
            exc=exc,
            on_error=on_error,
            error_log=error_log,
        )
    except Exception as exc:
        return _handle_error(
            f"Erreur à l’ouverture de l’image: {input_path}",
            exc=exc,
            on_error=on_error,
            error_log=error_log,
        )

    # Orientation EXIF
    try:
        img = _apply_exif_orientation(img)
    except Exception:
        pass

    # Préparer options de sauvegarde
    save_kwargs: dict = {"optimize": optimize}
    if quality is not None:
        save_kwargs["quality"] = int(quality)

    # EXIF
    exif = None
    if preserve_exif:
        try:
            exif = img.info.get("exif")
            if exif:
                save_kwargs["exif"] = exif
        except Exception:
            pass

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Animations (GIF/APNG)
        if (
            allow_animated
            and getattr(img, "is_animated", False)
            and ImageSequence is not None
        ):
            frames = [f.copy() for f in ImageSequence.Iterator(img)]
            if mode is not None:
                frames = [f.convert(mode) for f in frames]

            first = frames[0]
            _atomic_save(
                first,
                output_path,
                save_all=True,
                append_images=frames[1:],
                **save_kwargs,
            )
        else:
            if mode is not None:
                img = img.convert(mode)
            _atomic_save(img, output_path, **save_kwargs)

    except MemoryError as exc:
        return _handle_error(
            f"Image trop volumineuse pour la mémoire: {input_path}",
            exc=exc,
            on_error=on_error,
            error_log=error_log,
        )
    except OSError as exc:
        return _handle_error(
            f"Erreur E/S lors de la sauvegarde: {output_path}",
            exc=exc,
            on_error=on_error,
            error_log=error_log,
        )
    except Exception as exc:
        return _handle_error(
            f"Erreur inattendue {input_path} -> {output_path}",
            exc=exc,
            on_error=on_error,
            error_log=error_log,
        )

    return True


__all__ = ["image_to_image", "ImageConversionError"]
