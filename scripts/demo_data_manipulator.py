from __future__ import annotations

from pathlib import Path

import pandas as pd

from modules.data_manipulation import DataManipulator, FeatureSpec, MergeSpec


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    # Exemple: dataset A et B (tu peux remplacer par tes vrais chemins)
    a_path = root / "data" / "generated_dirty_cleaner_test.csv"
    b_path = root / "data" / "test_employees.csv"  # existe déjà dans data/

    dm = DataManipulator()

    df_a = dm.load(a_path)
    df_b = dm.load(b_path)

    # Exemple de merge: à adapter selon tes clés
    # Ici on tente un merge "safe": si les colonnes sont absentes, on skippe.
    if "id" in df_a.columns and "employee_id" in df_b.columns:
        merged = dm.merge(df_a, df_b, MergeSpec(how="left", left_on="id", right_on="employee_id", validate=None, indicator=True))
    else:
        merged = df_a

    # Features automatiques (flags email/tel/ip/uuid/etc.) sur toutes colonnes texte
    dm.add_contact_and_id_features(merged, prefix="has_")

    # Feature units sur une colonne précise si elle existe
    if "distance_vitesse" in merged.columns:
        dm.add_unit_columns(merged, source="distance_vitesse", target_unit="m")

    # Exemple de features déclaratives
    feats = []
    if "Commentaire" in merged.columns:
        feats.append(
            FeatureSpec(
                name="comment_has_html",
                kind="regex_flag",
                source="Commentaire",
                pattern=r"<[^>]+>",
            )
        )
    if feats:
        dm.apply_features(merged, feats)

    out_dir = root / "exports" / "data_manipulator_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(out_dir / "merged_with_features.csv", index=False)
    dm.write_report(out_dir / "report.json")

    print(f"Wrote: {out_dir/'merged_with_features.csv'}")
    print(f"Wrote: {out_dir/'report.json'}")


if __name__ == "__main__":
    main()
