from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .data_manipulator import DataManipulator, FeatureSpec, MergeSpec


PathLike = Union[str, Path]


class PipelineError(RuntimeError):
    pass


def _read_config(path: Path) -> Dict[str, Any]:
    suf = path.suffix.lower()
    if suf in {".json"}:
        return json.loads(path.read_text(encoding="utf-8"))
    if suf in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise PipelineError("YAML config requires 'pyyaml' (pip install pyyaml)") from e
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise PipelineError(f"Unsupported config type: {suf}")


def _maybe(path: Optional[str], root: Path) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def run_pipeline(config_path: PathLike, *, workspace_root: Optional[PathLike] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Exécute une pipeline de manipulation décrite dans un fichier JSON/YAML.

    Retourne (df_final, report).
    """
    config_path = Path(config_path)
    root = Path(workspace_root).resolve() if workspace_root else config_path.resolve().parents[1]

    cfg = _read_config(config_path)
    dm = DataManipulator()

    datasets = cfg.get("datasets") or {}
    left_cfg = datasets.get("left") or {}
    right_cfg = datasets.get("right")

    left_path = _maybe(left_cfg.get("path"), root)
    if left_path is None:
        raise PipelineError("Missing datasets.left.path")

    left_kwargs = left_cfg.get("load_kwargs") or {}
    df_left = dm.load(left_path, **left_kwargs)

    df_right = None
    if right_cfg and right_cfg.get("path"):
        right_path = _maybe(right_cfg.get("path"), root)
        right_kwargs = (right_cfg.get("load_kwargs") or {}) if isinstance(right_cfg, dict) else {}
        df_right = dm.load(right_path, **right_kwargs)

    df = df_left

    merge_cfg = cfg.get("merge")
    if merge_cfg and df_right is not None:
        spec = MergeSpec(
            how=merge_cfg.get("how", "left"),
            on=merge_cfg.get("on"),
            left_on=merge_cfg.get("left_on"),
            right_on=merge_cfg.get("right_on"),
            suffixes=tuple(merge_cfg.get("suffixes", ["_x", "_y"])),
            validate=merge_cfg.get("validate"),
            indicator=bool(merge_cfg.get("indicator", False)),
        )
        try:
            df = dm.merge(df, df_right, spec)
        except KeyError as e:
            raise PipelineError(
                "Merge failed (missing key). "
                f"left columns={list(df.columns)} right columns={list(df_right.columns)}"
            ) from e

    operations: List[Dict[str, Any]] = cfg.get("operations") or []

    for step in operations:
        op = step.get("op")
        args = step.get("args") or {}

        if op == "rename_columns":
            mapping = step.get("mapping") or args.get("mapping")
            if not isinstance(mapping, dict):
                raise PipelineError("rename_columns requires mapping")
            df = dm.rename_columns(df, mapping)

        elif op == "drop_columns":
            cols = step.get("columns") or args.get("columns")
            if not isinstance(cols, list):
                raise PipelineError("drop_columns requires columns: [..]")
            df = dm.drop_columns(df, cols, errors=args.get("errors", "ignore"))

        elif op == "replace_in_column":
            column = args.get("column") or step.get("column")
            pattern = args.get("pattern") or step.get("pattern")
            repl = args.get("repl") or step.get("repl")
            if not column or pattern is None or repl is None:
                raise PipelineError("replace_in_column requires column, pattern, repl")
            df = dm.replace_in_column(
                df,
                column=str(column),
                pattern=str(pattern),
                repl=str(repl),
                regex=bool(args.get("regex", True)),
            )

        elif op == "merge":
            if df_right is None:
                raise PipelineError("merge op requires datasets.right")
            m = args or step.get("merge") or {}
            spec = MergeSpec(
                how=m.get("how", "left"),
                on=m.get("on"),
                left_on=m.get("left_on"),
                right_on=m.get("right_on"),
                suffixes=tuple(m.get("suffixes", ["_x", "_y"])),
                validate=m.get("validate"),
                indicator=bool(m.get("indicator", False)),
            )
            try:
                df = dm.merge(df, df_right, spec)
            except KeyError as e:
                raise PipelineError(
                    "Merge failed (missing key). "
                    f"left columns={list(df.columns)} right columns={list(df_right.columns)}"
                ) from e

        elif op == "clean":
            if not isinstance(args, dict):
                raise PipelineError("clean requires args object")
            df = dm.clean(df, **args)

        elif op == "add_contact_and_id_features":
            df = dm.add_contact_and_id_features(
                df,
                source_columns=args.get("source_columns"),
                prefix=args.get("prefix", "has_"),
            )

        elif op == "add_unit_columns":
            df = dm.add_unit_columns(
                df,
                source=args["source"],
                target_unit=args.get("target_unit"),
                exclude_currency=bool(args.get("exclude_currency", True)),
            )

        elif op == "add_regex_flags":
            source = args["source"]
            patterns = args.get("patterns")
            if not isinstance(patterns, dict):
                raise PipelineError("add_regex_flags requires patterns dict")
            df = dm.add_regex_flags(
                df,
                source=source,
                patterns=patterns,
                prefix=args.get("prefix", "has_"),
                to_int=bool(args.get("to_int", True)),
            )

        elif op == "apply_features":
            feats = step.get("features") or args.get("features")
            if not isinstance(feats, list):
                raise PipelineError("apply_features requires features: [..]")
            feature_specs = [FeatureSpec(**f) for f in feats]
            df = dm.apply_features(df, feature_specs)

        elif op == "save":
            out_path = step.get("path") or args.get("path")
            if not out_path:
                raise PipelineError("save requires path")
            dm.save(df, _maybe(out_path, root) or Path(out_path), **(args.get("save_kwargs") or {}))

        else:
            raise PipelineError(f"Unknown op: {op}")

    out_cfg = cfg.get("output") or {}
    report_path = out_cfg.get("report_path")
    if report_path:
        dm.write_report(_maybe(report_path, root) or Path(report_path))

    report = dm.get_report()
    report.setdefault("config", {})
    report["config"].update({"config_path": str(config_path)})

    # Ajoute un mini-resume final
    report.setdefault("summary", {})
    report["summary"].update({"rows": int(len(df)), "cols": int(len(df.columns))})

    return df, report
