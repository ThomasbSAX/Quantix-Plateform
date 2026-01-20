"""Package de conversion de formats.

Objectif:
- Fournir une API stable et intégrable côté Flask.
- Éviter les imports fragiles/externes (un `converter.py` parent n'existe pas dans ce repo).

API principale:
- `convert_any_to_any` (routeur)
- `FormatConverter` (wrapper orienté objet)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .convert_router import CONVERSION_MAP, convert_any_to_any


@dataclass
class FormatConverter:
    """Wrapper simple autour du routeur de conversion."""

    src_path: str | Path

    def convert(
        self,
        target_format: str,
        *,
        source_format: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
        **kwargs: Any,
    ):
        return convert_any_to_any(
            self.src_path,
            src_format=source_format,
            dst_format=target_format,
            output_dir=output_dir,
            **kwargs,
        )


__all__ = [
    "FormatConverter",
    "convert_any_to_any",
    "CONVERSION_MAP",
]
