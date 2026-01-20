from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .engine import register_plot
from .encoding import encode_unique_as_int
from .utils import param_bool, param_int, param_list_str, require_columns


def _base_color(params: Mapping[str, Any]) -> str:
    c = params.get("color")
    return str(c) if c else "#1f77b4"


def _agg_name(name: str) -> str:
    n = str(name or "mean").strip().lower()
    if n in {"mean", "avg"}:
        return "mean"
    if n in {"median", "med"}:
        return "median"
    if n in {"min"}:
        return "min"
    if n in {"max"}:
        return "max"
    return "mean"


@register_plot("radar")
def radar(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    """Spider/radar chart.

    Params:
      - group_by: column name (categorical) to draw one trace per group
      - metrics: list of numeric columns
      - agg: mean|median|min|max (default mean)
      - top_n: limit number of groups (by size)
      - normalize: bool -> min-max normalize each metric across groups
      - fill: bool -> fill polygon
    """
    group_by = params.get("group_by")
    if not isinstance(group_by, str) or not group_by:
        raise ValueError("'group_by' is required")

    metrics = param_list_str(params, "metrics")
    if not metrics:
        raise ValueError("'metrics' is required (list of numeric columns)")

    require_columns(df, [group_by] + metrics)

    agg = _agg_name(str(params.get("agg", "mean")))
    top_n = param_int(params, "top_n", 8) or 8
    normalize = param_bool(params, "normalize", True)
    fill = param_bool(params, "fill", True)

    colors = param_list_str(params, "colors")
    single_color = _base_color(params)

    dff = df[[group_by] + metrics].copy()
    dff[group_by] = dff[group_by].astype("string").fillna("(NA)")

    # pick top groups by count
    group_sizes = dff[group_by].value_counts(dropna=False)
    groups = group_sizes.head(top_n).index.tolist()
    dff = dff[dff[group_by].isin(groups)]

    # aggregate
    agg_df = dff.groupby(group_by, dropna=False)[metrics].agg(agg).reset_index()

    # min-max normalize per metric
    values = agg_df[metrics].to_numpy(dtype=float)
    if normalize:
        mins = np.nanmin(values, axis=0)
        maxs = np.nanmax(values, axis=0)
        denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
        values = (values - mins) / denom
        agg_df[metrics] = values

    categories = metrics

    fig = go.Figure()

    for idx, row in agg_df.iterrows():
        name = str(row[group_by])
        r = [float(row[m]) if pd.notna(row[m]) else np.nan for m in metrics]
        # close polygon
        r = r + [r[0]]
        theta = categories + [categories[0]]

        line_color = None
        if colors:
            line_color = colors[idx % len(colors)]
        elif single_color:
            line_color = single_color

        fig.add_trace(
            go.Scatterpolar(
                r=r,
                theta=theta,
                mode="lines+markers",
                name=name,
                fill="toself" if fill else None,
                opacity=float(params.get("opacity", 0.75) or 0.75),
                line=dict(color=line_color) if line_color else None,
                marker=dict(color=line_color) if line_color else None,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1] if normalize else None),
        ),
        showlegend=True,
    )

    return fig


@register_plot("parallel_coordinates")
def parallel_coordinates(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    dims = param_list_str(params, "dimensions")
    if not dims:
        raise ValueError("'dimensions' is required")
    require_columns(df, dims)

    color_by = params.get("color_by")
    if isinstance(color_by, str) and color_by:
        require_columns(df, [color_by])

    color_col = None
    dff = df
    if isinstance(color_by, str) and color_by:
        series = df[color_by]
        if pd.api.types.is_numeric_dtype(series):
            color_col = color_by
        else:
            # Plotly parallel_coordinates requires numeric colors.
            encoded = encode_unique_as_int(series, strategy="frequency").values
            dff = df.copy()
            dff["__color_encoded"] = encoded
            color_col = "__color_encoded"

    fig = px.parallel_coordinates(
        dff,
        dimensions=dims,
        color=color_col,
        color_continuous_scale="Viridis",
    )
    return fig


@register_plot("scatter_matrix")
def scatter_matrix(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    dims = param_list_str(params, "dimensions")
    if not dims:
        raise ValueError("'dimensions' is required")
    require_columns(df, dims)

    color_by = params.get("color_by")
    if isinstance(color_by, str) and color_by:
        require_columns(df, [color_by])

    fig = px.scatter_matrix(
        df,
        dimensions=dims,
        color=color_by if isinstance(color_by, str) and color_by else None,
        opacity=float(params.get("opacity", 0.75) or 0.75),
    )

    fig.update_traces(diagonal_visible=param_bool(params, "diagonal_visible", True))
    return fig
