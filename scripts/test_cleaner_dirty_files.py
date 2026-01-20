from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from modules.cleaner.Cleaner import Cleaner


@dataclass(frozen=True)
class Case:
    name: str
    cleaner_kwargs: Dict[str, Any]


def _make_dirty_dataframe(n: int = 120) -> pd.DataFrame:
    random.seed(7)
    np.random.seed(7)

    names = [
        "  Alice  ",
        "BOB",
        "Chloé",
        "d’Artagnan",
        "ÉLODIE",
        " jean  pierre ",
        "Mickaël",
        "  Noël",
        "Zoë",
        "O'CONNOR",
        "محمد",
        "李雷",
    ]

    countries = [" FR", "fr ", "USA", " uk", "DE", "es", "  ", None]

    # Dates hétérogènes
    base = datetime(2024, 1, 1)
    dates = []
    for _ in range(n):
        d = base + timedelta(days=int(np.random.randint(0, 365)))
        fmt = np.random.choice(["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "text"])
        if fmt == "text":
            dates.append(f"  {d.strftime('%d %b %Y')} ")
        else:
            dates.append(d.strftime(fmt))

    # Numériques: mélange FR/EN, milliers, sci, strings
    values = []
    for i in range(n):
        x = float(np.random.normal(1000, 120))
        # outliers
        if i in {5, 37, 88}:
            x *= 25
        if i in {12, 79}:
            x *= -8

        style = np.random.choice(["fr", "en", "space_thousand", "dot_thousand", "sci", "plain", "messy"])
        if style == "fr":
            # 1 234,56
            values.append(f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", " "))
        elif style == "en":
            # 1,234.56
            values.append(f"{x:,.2f}")
        elif style == "space_thousand":
            values.append(f"{x:,.0f}".replace(",", " "))
        elif style == "dot_thousand":
            values.append(f"{x:,.0f}".replace(",", "."))
        elif style == "sci":
            exp = int(np.random.choice([2, 3, 4, 6]))
            basev = x / (10**exp)
            values.append(f"{basev:.3f}e{exp}")
        elif style == "plain":
            values.append(str(round(x, 3)))
        else:
            values.append(f"  {str(round(x, 2)).replace('.', ',')}  ")

    # Mesures avec unités
    units = []
    for i in range(n):
        if i % 5 == 0:
            units.append(f"{np.random.randint(1, 20)} km")
        elif i % 5 == 1:
            units.append(f"{np.random.randint(200, 900)}m")
        elif i % 5 == 2:
            units.append(f"{np.random.uniform(1, 5):.1f} mi")
        elif i % 5 == 3:
            units.append(f"{np.random.randint(10, 90)} km/h")
        else:
            # Doit être ignoré par l'extraction d'unités (monnaie)
            units.append(f"prix {np.random.randint(5, 200)}€")

    comments = [
        "  Très  satisfait!!! ",
        "<b>Très</b> satisfait!!!",
        "N/A",
        "NULL",
        "  bon\tproduit\nmais livraison lente ",
        "J’aime bien — mais…  ",
        "C'est  super  !!!",
    ]

    rows = []
    for i in range(n):
        rows.append(
            {
                " id ": f" {i + 1} ",
                "Nom ": random.choice(names),
                "Pays": random.choice(countries),
                "date_embauche ": dates[i],
                "salaire (€) ": values[i],
                "distance/vitesse ": units[i],
                "Commentaire": random.choice(comments),
                "email ": random.choice(["test@example.com", "bad@@mail", " jean.pierre@exemple.fr ", None]),
            }
        )

    # Ajout de quelques lignes très cassées
    rows.append(
        {
            " id ": " 999 ",
            "Nom ": None,
            "Pays": "  ",
            "date_embauche ": "??",
            "salaire (€) ": "",
            "distance/vitesse ": "",
            "Commentaire": "   ",
            "email ": "",
        }
    )
    rows.append(
        {
            " id ": "1000",
            "Nom ": "   ",
            "Pays": None,
            "date_embauche ": None,
            "salaire (€) ": "NaN",
            "distance/vitesse ": "1000 hPa",
            "Commentaire": "NULL",
            "email ": "a@b.com",
        }
    )

    return pd.DataFrame(rows)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarize_case(name: str, stats: Dict[str, Any], report: Dict[str, Any]) -> None:
    rb, cb = stats.get("rows_before"), stats.get("cols_before")
    ra, ca = stats.get("rows_after"), stats.get("cols_after")

    dtype_changes = stats.get("dtype_changes") or {}

    unit_conversions = report.get("unit_conversions")
    sensitive_masking = report.get("sensitive_masking")

    print(f"\n=== {name} ===")
    print(f"shape: {rb}x{cb} -> {ra}x{ca}")
    print(f"ops_count: {len(stats.get('operations') or [])}")
    print(f"dtype_changes: {len(dtype_changes)}")
    if isinstance(unit_conversions, list):
        print(f"unit_conversions: {len(unit_conversions)}")
    if isinstance(sensitive_masking, list):
        print(f"sensitive_masking: {len(sensitive_masking)}")


def run_for_file(input_path: Path, cases: list[Case], out_root: Path) -> None:
    print(f"\n## Testing on: {input_path}")

    for case in cases:
        cleaner = Cleaner(**case.cleaner_kwargs)
        df_clean = cleaner.clean(file_path=input_path, auto_detect_types=True)
        stats = cleaner.get_stats(detailed=True)
        report = cleaner.get_transformation_report()

        _summarize_case(case.name, stats, report)

        case_dir = out_root / input_path.stem / case.name
        case_dir.mkdir(parents=True, exist_ok=True)

        # Save cleaned output for quick inspection
        cleaned_csv = case_dir / "cleaned_preview.csv"
        df_clean.to_csv(cleaned_csv, index=False)

        # Save JSON report
        report_path = case_dir / "report.json"
        _write_json(report_path, report)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    out_root = root / "reports" / "dirty_cleaner_tests"
    data_dir.mkdir(exist_ok=True)

    df_dirty = _make_dirty_dataframe(n=120)

    csv_path = data_dir / "generated_dirty_cleaner_test.csv"
    xlsx_path = data_dir / "generated_dirty_cleaner_test.xlsx"

    # CSV volontairement avec ; (et colonnes sales)
    df_dirty.to_csv(csv_path, index=False, sep=";")
    df_dirty.to_excel(xlsx_path, index=False)

    print(f"Wrote: {csv_path} ({df_dirty.shape[0]}x{df_dirty.shape[1]})")
    print(f"Wrote: {xlsx_path} ({df_dirty.shape[0]}x{df_dirty.shape[1]})")

    cases = [
        Case(name="defaults_safe", cleaner_kwargs={}),
        Case(
            name="with_units",
            cleaner_kwargs={"convert_units": True, "unit_parse_threshold": 0.25, "unit_mode": "add"},
        ),
        Case(
            name="with_units_and_masking",
            cleaner_kwargs={
                "convert_units": True,
                "unit_parse_threshold": 0.25,
                "unit_mode": "add",
                "mask_sensitive_data": True,
            },
        ),
        Case(
            name="aggressive_outliers_drop",
            cleaner_kwargs={"remove_outliers": True, "drop_missing": True, "missing_threshold": 0.4},
        ),
    ]

    run_for_file(csv_path, cases, out_root)
    run_for_file(xlsx_path, cases, out_root)

    print(f"\nArtifacts written under: {out_root}")


if __name__ == "__main__":
    main()
