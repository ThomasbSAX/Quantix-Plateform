from __future__ import annotations

import argparse
from pathlib import Path

from modules.data_manipulation.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a data manipulation pipeline (JSON/YAML)")
    parser.add_argument("config", type=str, help="Path to pipeline config (.json/.yml/.yaml)")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Workspace root (used to resolve relative paths). Default: inferred from config location.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    root = Path(args.root).resolve() if args.root else None

    df, report = run_pipeline(config_path, workspace_root=root)

    print(f"OK: rows={len(df)} cols={len(df.columns)}")
    if report.get("operations"):
        print(f"ops={len(report['operations'])}")


if __name__ == "__main__":
    main()
