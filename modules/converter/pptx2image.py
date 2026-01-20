from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


def pptx_to_images(
    pptx_path: str | Path,
    output_dir: str | Path,
    *,
    image_format: str = "png",
    dpi: Optional[int] = None,
    prefix: str = "slide",
    timeout: int = 120,
) -> List[Path]:
    """
    Convertit un PPTX en images avec garantie:
    → 1 slide = 1 image

    Utilise LibreOffice (soffice) en mode headless.

    Returns:
        Liste ordonnée des images générées.
    """

    pptx_path = Path(pptx_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pptx_path.exists():
        raise FileNotFoundError(pptx_path)

    soffice = shutil.which("soffice")
    if not soffice:
        raise RuntimeError("LibreOffice (soffice) introuvable sur le système")

    # Nettoyer anciens fichiers du même préfixe
    for f in output_dir.glob(f"{pptx_path.stem}*.{image_format}"):
        try:
            f.unlink()
        except Exception:
            pass

    cmd = [
        soffice,
        "--headless",
        "--invisible",
        "--convert-to", image_format,
        "--outdir", str(output_dir),
        str(pptx_path),
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
    )

    # LibreOffice génère typiquement :
    # <name>.png, <name>_1.png, <name>_2.png ...
    generated = sorted(output_dir.glob(f"{pptx_path.stem}*.{image_format}"))

    if not generated:
        raise RuntimeError("Aucune image générée par LibreOffice")

    # Renommage propre : slide_0001.png, slide_0002.png, ...
    final_images: List[Path] = []

    for idx, img in enumerate(generated, start=1):
        new_name = f"{prefix}_{idx:04d}.{image_format}"
        new_path = output_dir / new_name
        img.replace(new_path)
        final_images.append(new_path)

    return final_images


__all__ = ["pptx_to_images"]
