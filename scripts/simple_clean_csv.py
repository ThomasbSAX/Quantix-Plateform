import argparse
import re
from pathlib import Path

import pandas as pd


def _normalize_cell(value: object) -> object:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    # leave non-strings as-is
    if not isinstance(value, str):
        return value

    s = value
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    if s == "":
        return None
    return s


def clean_csv(
    input_path: Path,
    output_path: Path,
    *,
    drop_empty_rows_threshold: float = 1.0,
    drop_empty_cols_threshold: float = 1.0,
) -> dict:
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)

    before_shape = df.shape

    # Normalize all cells (applymap is deprecated in recent pandas)
    df = df.apply(lambda col: col.map(_normalize_cell))

    # Drop columns based on % empties
    if df.shape[0] > 0 and df.shape[1] > 0 and drop_empty_cols_threshold < 1.0:
        empty_frac_by_col = df.isna().mean(axis=0)
        keep_cols = empty_frac_by_col < drop_empty_cols_threshold
        df = df.loc[:, keep_cols]

    # Drop rows based on % empties
    if df.shape[0] > 0 and df.shape[1] > 0 and drop_empty_rows_threshold < 1.0:
        empty_frac_by_row = df.isna().mean(axis=1)
        keep_rows = empty_frac_by_row < drop_empty_rows_threshold
        df = df.loc[keep_rows, :]

    # Always drop fully empty columns/rows
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    after_shape = df.shape

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return {
        "before_shape": before_shape,
        "after_shape": after_shape,
        "dropped_rows": int(before_shape[0] - after_shape[0]),
        "dropped_cols": int(before_shape[1] - after_shape[1]),
        "drop_empty_rows_threshold": float(drop_empty_rows_threshold),
        "drop_empty_cols_threshold": float(drop_empty_cols_threshold),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Simple CSV cleaner (lowercase, trim, remove empty rows/cols, normalize spaces).")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument(
        "--drop-empty-rows-threshold",
        type=float,
        default=1.0,
        help=(
            "Supprime une ligne si la proportion de cellules vides est >= à ce seuil (0.0 à 1.0). "
            "Ex: 0.7 => supprime si >= 70% vide. Par défaut: 1.0 (supprime seulement les lignes entièrement vides)."
        ),
    )
    p.add_argument(
        "--drop-empty-cols-threshold",
        type=float,
        default=1.0,
        help=(
            "Supprime une colonne si la proportion de cellules vides est >= à ce seuil (0.0 à 1.0). "
            "Ex: 0.7 => supprime si >= 70% vide. Par défaut: 1.0 (supprime seulement les colonnes entièrement vides)."
        ),
    )
    args = p.parse_args()

    if not (0.0 <= args.drop_empty_rows_threshold <= 1.0):
        raise SystemExit("--drop-empty-rows-threshold doit être entre 0.0 et 1.0")
    if not (0.0 <= args.drop_empty_cols_threshold <= 1.0):
        raise SystemExit("--drop-empty-cols-threshold doit être entre 0.0 et 1.0")

    stats = clean_csv(
        args.input,
        args.output,
        drop_empty_rows_threshold=args.drop_empty_rows_threshold,
        drop_empty_cols_threshold=args.drop_empty_cols_threshold,
    )
    print(stats)


if __name__ == "__main__":
    main()
