"""Generate a gallery of plots from a CSV.

Usage (from this folder):
  .venv/bin/python gallery_from_csv.py test_num.csv

It writes HTML files into ./gallery_out

This is meant as a dev tool to quickly validate plot functions.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
# `modules/` lives in "Code site web", so we need that on sys.path.
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from modules.plots import get_plot_catalog, make_figure  # type: ignore  # noqa: E402
except Exception:
    # Fallback if running with this folder as the only workspace root.
    from .engine import get_plot_catalog, make_figure  # type: ignore


@dataclass
class GalleryItem:
    filename: str
    title: str
    df: pd.DataFrame
    spec: Dict[str, Any]


def _safe_slug(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _ensure_positive(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return bool(len(s)) and bool((s > 0).all())


def _add_bins(df: pd.DataFrame) -> pd.DataFrame:
    dff = df.copy()

    if "attendance_percent" in dff.columns:
        try:
            dff["attendance_bin"] = pd.qcut(dff["attendance_percent"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        except Exception:
            dff["attendance_bin"] = pd.cut(dff["attendance_percent"], 4, labels=["B1", "B2", "B3", "B4"])

    if "sleep_hours" in dff.columns:
        try:
            dff["sleep_bin"] = pd.qcut(dff["sleep_hours"], 3, labels=["low", "mid", "high"])
        except Exception:
            dff["sleep_bin"] = pd.cut(dff["sleep_hours"], 3, labels=["low", "mid", "high"])

    return dff


def _agg_mean(df: pd.DataFrame, group_col: str, y_col: str) -> pd.DataFrame:
    tmp = df[[group_col, y_col]].copy()
    tmp[group_col] = tmp[group_col].astype("string").fillna("(NA)")
    out = tmp.groupby(group_col, dropna=False)[y_col].mean().reset_index()
    out = out.sort_values(y_col, ascending=False)
    return out


def build_gallery(df: pd.DataFrame) -> List[GalleryItem]:
    df = _add_bins(df)

    items: List[GalleryItem] = []

    def add(title: str, spec: Dict[str, Any], *, df_override: Optional[pd.DataFrame] = None) -> None:
        filename = f"{len(items)+1:02d}_{_safe_slug(title)[:60]}.html"
        items.append(GalleryItem(filename=filename, title=title, df=df_override if df_override is not None else df, spec=spec))

    # Core numeric columns from this dataset
    num_cols = [
        c
        for c in ["hours_studied", "sleep_hours", "attendance_percent", "previous_scores", "exam_score"]
        if c in df.columns
    ]

    # Choose a target
    y = "exam_score" if "exam_score" in df.columns else (num_cols[-1] if num_cols else None)
    x1 = "hours_studied" if "hours_studied" in df.columns else (num_cols[0] if num_cols else None)
    x2 = "sleep_hours" if "sleep_hours" in df.columns else (num_cols[1] if len(num_cols) > 1 else x1)

    if y is None or x1 is None:
        raise RuntimeError("CSV ne contient pas assez de colonnes numériques attendues")

    # 1-4 Distributions
    add(
        "Histogramme score + stats + fit normal",
        {
            "plot_type": "histogram",
            "x": y,
            "nbins": 35,
            "compare_distributions": ["normal"],
            "stats_lines": ["mean", "median", "q25", "q75"],
            "color": "#1f77b4",
            "title": f"Histogramme {y}",
        },
    )

    if "previous_scores" in df.columns:
        add(
            "Histogramme previous_scores + fits",
            {
                "plot_type": "histogram",
                "x": "previous_scores",
                "nbins": 30,
                "compare_distributions": ["normal", "lognormal", "exponential", "uniform"],
                "stats_lines": ["mean", "median", "q10", "q90"],
                "color": "#2ca02c",
                "title": "previous_scores (avec fits)",
            },
        )

    if "attendance_percent" in df.columns:
        add(
            "Histogramme attendance (density)",
            {
                "plot_type": "histogram",
                "x": "attendance_percent",
                "nbins": 40,
                "normalize": "probability density",
                "stats_lines": ["mean", "median"],
                "color": "#9467bd",
                "title": "attendance_percent (density)",
            },
        )

    add(
        "KDE score + fit normal",
        {
            "plot_type": "kde",
            "x": y,
            "compare_distributions": ["normal"],
            "stats_lines": ["mean", "median"],
            "color": "#ff7f0e",
            "title": f"KDE {y}",
        },
    )

    # 5-8 Box/violin + ECDF + QQ
    if "attendance_bin" in df.columns:
        add(
            "Boxplot score par quartile d'attendance",
            {
                "plot_type": "box",
                "x": "attendance_bin",
                "y": y,
                "color": "#1f77b4",
                "title": "Boxplot score vs attendance_bin",
            },
        )
        add(
            "Violin score par quartile d'attendance",
            {
                "plot_type": "violin",
                "x": "attendance_bin",
                "y": y,
                "color": "#1f77b4",
                "title": "Violin score vs attendance_bin",
            },
        )

    add(
        "ECDF score + stats",
        {
            "plot_type": "ecdf",
            "x": y,
            "stats_lines": ["mean", "median", "q25", "q75"],
            "color": "#17becf",
            "title": f"ECDF {y}",
        },
    )

    add(
        "QQ-plot normal (score)",
        {
            "plot_type": "qqplot_normal",
            "x": y,
            "fit_line": True,
            "color": "#d62728",
            "title": f"QQ-plot normal: {y}",
        },
    )

    # 9-14 Scatter + regressions + tangents + critical points
    add(
        "Scatter x1 vs score (couleur par attendance_bin)",
        {
            "plot_type": "scatter",
            "x": x1,
            "y": y,
            "color_by": "attendance_bin" if "attendance_bin" in df.columns else None,
            "opacity": 0.85,
            "title": f"{x1} vs {y}",
        },
    )

    add(
        "Scatter x1 vs score + regression linéaire",
        {
            "plot_type": "scatter",
            "x": x1,
            "y": y,
            "regression": "linear",
            "tangent_at": [float(df[x1].median())] if x1 in df.columns else [],
            "show_fixed_points": True,
            "color": "#1f77b4",
            "title": "Regression linéaire + tangente + points fixes",
        },
    )

    # Poly 2
    add(
        "Scatter x2 vs score + poly2 + stationnaires",
        {
            "plot_type": "scatter",
            "x": x2,
            "y": y,
            "regression": "poly2",
            "tangent_at": [float(np.nanmedian(pd.to_numeric(df[x2], errors="coerce")))],
            "show_stationary_points": True,
            "color": "#2ca02c",
            "title": "Poly2 + tangente + stationnaires",
        },
    )

    # Poly 3
    add(
        "Scatter x1 vs score + poly3 + tangentes + inflexion",
        {
            "plot_type": "scatter",
            "x": x1,
            "y": y,
            "regression": "poly3",
            "tangent_at": [float(df[x1].quantile(0.2)), float(df[x1].quantile(0.8))],
            "show_stationary_points": True,
            "show_inflection_points": True,
            "show_fixed_points": True,
            "color": "#ff7f0e",
            "title": "Poly3 + tangentes + points critiques",
        },
    )

    # Extra scatter: attendance vs score with poly2
    if "attendance_percent" in df.columns:
        add(
            "Scatter attendance vs score + poly2",
            {
                "plot_type": "scatter",
                "x": "attendance_percent",
                "y": y,
                "regression": "poly2",
                "show_stationary_points": True,
                "color": "#9467bd",
                "title": "attendance_percent vs score (poly2)",
            },
        )

    # 2D densities
    add(
        "Densité 2D (heatmap) x1 vs score",
        {"plot_type": "density2d_heatmap", "x": x1, "y": y, "nbinsx": 35, "nbinsy": 35, "title": "density heatmap"},
    )
    add(
        "Densité 2D (contour) x1 vs score",
        {"plot_type": "density2d_contour", "x": x1, "y": y, "title": "density contour"},
    )

    # 15-18 Line, correlations, categorical summaries
    # Line: sort by x1
    df_sorted = df.sort_values(x1).reset_index(drop=True)
    add(
        "Courbe score vs x1 (triée)",
        {"plot_type": "line", "x": x1, "y": y, "markers": True, "regression": "poly2", "title": "line + poly2"},
        df_override=df_sorted,
    )

    add(
        "Heatmap corrélations",
        {"plot_type": "corr_heatmap", "method": "pearson", "title": "corr"},
    )

    if "student_id" in df.columns:
        add(
            "Count plot student_id (top 30)",
            {"plot_type": "count", "x": "student_id", "top_n": 30, "color": "#7f7f7f", "title": "count student_id"},
        )

    if "attendance_bin" in df.columns:
        agg = _agg_mean(df, "attendance_bin", y)
        add(
            "Bar mean score par attendance_bin",
            {"plot_type": "bar", "x": "attendance_bin", "y": y, "color": "#1f77b4", "title": "mean score by attendance_bin"},
            df_override=agg,
        )

    if "sleep_bin" in df.columns:
        agg2 = _agg_mean(df, "sleep_bin", y)
        add(
            "Bar mean score par sleep_bin",
            {"plot_type": "bar", "x": "sleep_bin", "y": y, "color": "#2ca02c", "title": "mean score by sleep_bin"},
            df_override=agg2,
        )

    # Multivariate (radar / parallel coords / scatter matrix)
    if "attendance_bin" in df.columns:
        add(
            "Radar (spider) par attendance_bin",
            {
                "plot_type": "radar",
                "group_by": "attendance_bin",
                "metrics": [c for c in ["hours_studied", "sleep_hours", "previous_scores", "exam_score"] if c in df.columns],
                "agg": "mean",
                "top_n": 4,
                "normalize": True,
                "fill": True,
                "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                "title": "Radar par attendance_bin",
            },
        )

    dims = [c for c in ["hours_studied", "sleep_hours", "attendance_percent", "previous_scores", "exam_score"] if c in df.columns]
    if len(dims) >= 4:
        add(
            "Coordonnées parallèles (dims principales)",
            {"plot_type": "parallel_coordinates", "dimensions": dims[:5], "color_by": "attendance_percent", "title": "Parallel coordinates"},
        )
    if len(dims) >= 4:
        add(
            "Scatter matrix (dims principales)",
            {"plot_type": "scatter_matrix", "dimensions": dims[:5], "color_by": "attendance_bin" if "attendance_bin" in df.columns else None, "diagonal_visible": True, "title": "Scatter matrix"},
        )

    # 19-20: Log-axis examples (only if safe)
    if _ensure_positive(df[x1]):
        add(
            "ECDF log-x sur x1",
            {"plot_type": "ecdf", "x": x1, "log_x": True, "stats_lines": ["median", "q10", "q90"], "title": f"ECDF log-x {x1}"},
        )

    if _ensure_positive(df[y]):
        add(
            "Histogramme log-x sur score",
            {"plot_type": "histogram", "x": y, "log_x": True, "nbins": 35, "stats_lines": ["mean", "median"], "title": f"Histogram log-x {y}"},
        )

    # Ensure we end up around 20. If more, trim; if less, add variants.
    if len(items) < 20 and "previous_scores" in df.columns:
        add(
            "KDE previous_scores + fits",
            {
                "plot_type": "kde",
                "x": "previous_scores",
                "compare_distributions": ["normal", "lognormal"],
                "stats_lines": ["mean", "median"],
                "title": "KDE previous_scores",
            },
        )

    return items


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python gallery_from_csv.py path/to/file.csv")
        return 2

    csv_path = Path(sys.argv[1]).expanduser().resolve()
    if not csv_path.exists():
        print(f"CSV introuvable: {csv_path}")
        return 2

    out_dir = HERE / "gallery_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Optional: choose how many plots.
    n = 24
    if len(sys.argv) >= 3:
        try:
            n = int(sys.argv[2])
        except Exception:
            n = 24

    items = build_gallery(df)[:n]
    print(f"Generating {len(items)} plots -> {out_dir}")

    index_rows: List[str] = []
    failures: List[Tuple[str, str]] = []

    for item in items:
        try:
            fig = make_figure(item.df, item.spec)
            html = fig.to_html(include_plotlyjs="cdn", full_html=True)
            (out_dir / item.filename).write_text(html, encoding="utf-8")
            index_rows.append(f"<li><a href='{item.filename}'>{item.title}</a></li>")
            print(f"OK  {item.filename}  ({item.spec.get('plot_type')})")
        except Exception as e:
            failures.append((item.title, str(e)))
            print(f"ERR {item.filename}: {e}")

    index_html = """<!doctype html>
<html><head><meta charset='utf-8'><title>Plot Gallery</title></head>
<body>
<h1>Plot Gallery</h1>
<ul>
{rows}
</ul>
</body></html>
""".format(rows="\n".join(index_rows))

    (out_dir / "index.html").write_text(index_html, encoding="utf-8")

    meta = {
        "csv": str(csv_path),
        "n_plots": len(items),
        "catalog_version": get_plot_catalog().get("version"),
        "failures": [{"title": t, "error": err} for t, err in failures],
        "specs": [{"title": it.title, "file": it.filename, "spec": it.spec} for it in items],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if failures:
        print(f"Done with {len(failures)} failures. See meta.json")
        return 1

    print("Done. Open gallery_out/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
