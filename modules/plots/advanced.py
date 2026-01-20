from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .engine import register_plot
from .utils import (
    coerce_datetime,
    coerce_numeric,
    param_bool,
    param_int,
    param_list_str,
    require_columns,
)


def _base_color(params: Mapping[str, Any]) -> str:
    c = params.get("color")
    return str(c) if c else "#1f77b4"


def _agg_fn(name: str) -> Callable[[pd.Series], float]:
    n = str(name or "sum").strip().lower()
    if n in {"sum", "total"}:
        return lambda s: float(np.nansum(s.to_numpy(dtype=float)))
    if n in {"mean", "avg"}:
        return lambda s: float(np.nanmean(s.to_numpy(dtype=float)))
    if n in {"median", "med"}:
        return lambda s: float(np.nanmedian(s.to_numpy(dtype=float)))
    if n == "min":
        return lambda s: float(np.nanmin(s.to_numpy(dtype=float)))
    if n == "max":
        return lambda s: float(np.nanmax(s.to_numpy(dtype=float)))
    return lambda s: float(np.nansum(s.to_numpy(dtype=float)))


@register_plot("stacked_area")
def stacked_area(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    """Stacked area chart (time/category).

    Params:
      - x: datetime/categorical column
      - y: numeric column
      - group_by: categorical column (one area per group)
      - agg: sum|mean|median|min|max (default sum)
      - top_n: limit groups (by row count)
    """
    xcol = params.get("x")
    ycol = params.get("y")
    group_by = params.get("group_by")

    if not isinstance(xcol, str) or not xcol:
        raise ValueError("'x' is required")
    if not isinstance(ycol, str) or not ycol:
        raise ValueError("'y' is required")
    if not isinstance(group_by, str) or not group_by:
        raise ValueError("'group_by' is required")

    require_columns(df, [xcol, ycol, group_by])

    agg = str(params.get("agg", "sum"))
    top_n = param_int(params, "top_n", 12) or 12

    dff = df[[xcol, ycol, group_by]].copy()
    dff[ycol] = coerce_numeric(dff[ycol])
    dff[xcol] = coerce_datetime(dff[xcol])
    dff[group_by] = dff[group_by].astype("string").fillna("(NA)")

    dff = dff.dropna(subset=[xcol, ycol])
    if dff.empty:
        raise ValueError("No data after filtering (x/y)")

    # Pick top groups by frequency
    group_order = dff[group_by].value_counts(dropna=False).index.tolist()
    if top_n and top_n > 0:
        group_order = group_order[:top_n]
        dff = dff[dff[group_by].isin(group_order)]

    f = _agg_fn(agg)
    grouped = (
        dff.groupby([xcol, group_by], dropna=False)[ycol]
        .apply(f)
        .reset_index(name=ycol)
        .sort_values([xcol, group_by])
    )

    fig = px.area(
        grouped,
        x=xcol,
        y=ycol,
        color=group_by,
        category_orders={group_by: [str(x) for x in group_order]},
    )
    fig.update_layout(xaxis_title=xcol, yaxis_title=ycol)
    return fig


@register_plot("cumulative_sum")
def cumulative_sum(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    """Cumulative sum over x (optionally per group).

    Params:
      - x: datetime/categorical column
      - y: numeric column
      - group_by: optional categorical column
    """
    xcol = params.get("x")
    ycol = params.get("y")
    group_by = params.get("group_by")

    if not isinstance(xcol, str) or not xcol:
        raise ValueError("'x' is required")
    if not isinstance(ycol, str) or not ycol:
        raise ValueError("'y' is required")

    required = [xcol, ycol] + ([group_by] if isinstance(group_by, str) and group_by else [])
    require_columns(df, required)

    dff = df[required].copy()
    dff[ycol] = coerce_numeric(dff[ycol])
    dff[xcol] = coerce_datetime(dff[xcol])
    dff = dff.dropna(subset=[xcol, ycol])
    if dff.empty:
        raise ValueError("No data after filtering (x/y)")

    if isinstance(group_by, str) and group_by:
        dff[group_by] = dff[group_by].astype("string").fillna("(NA)")
        dff = dff.sort_values([group_by, xcol])
        dff["__cumsum"] = dff.groupby(group_by, dropna=False)[ycol].cumsum()
        fig = px.line(dff, x=xcol, y="__cumsum", color=group_by)
        fig.update_layout(xaxis_title=xcol, yaxis_title=f"Cumul({ycol})")
        return fig

    dff = dff.sort_values([xcol])
    dff["__cumsum"] = dff[ycol].cumsum()
    fig = px.line(dff, x=xcol, y="__cumsum")
    fig.update_traces(line=dict(color=_base_color(params), width=3))
    fig.update_layout(xaxis_title=xcol, yaxis_title=f"Cumul({ycol})")
    return fig


@register_plot("heatmap_pivot")
def heatmap_pivot(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    """Heatmap from a pivot table.

    Params:
      - x: column for heatmap x axis (categories/datetime)
      - y: column for heatmap y axis (categories)
      - value: numeric column
      - agg: mean|sum|median|min|max (default mean)
      - sort_x: bool (default True)
      - sort_y: bool (default True)
    """
    xcol = params.get("x")
    ycol = params.get("y")
    vcol = params.get("value")

    if not isinstance(xcol, str) or not xcol:
        raise ValueError("'x' is required")
    if not isinstance(ycol, str) or not ycol:
        raise ValueError("'y' is required")
    if not isinstance(vcol, str) or not vcol:
        raise ValueError("'value' is required")

    require_columns(df, [xcol, ycol, vcol])

    agg = str(params.get("agg", "mean"))
    sort_x = param_bool(params, "sort_x", True)
    sort_y = param_bool(params, "sort_y", True)

    dff = df[[xcol, ycol, vcol]].copy()
    dff[vcol] = coerce_numeric(dff[vcol])
    dff[xcol] = dff[xcol].astype("string").fillna("(NA)")
    dff[ycol] = dff[ycol].astype("string").fillna("(NA)")
    dff = dff.dropna(subset=[vcol])

    if dff.empty:
        raise ValueError("No numeric values for heatmap")

    agg_norm = agg.strip().lower()
    if agg_norm in {"sum", "total"}:
        aggfunc = "sum"
    elif agg_norm in {"median", "med"}:
        aggfunc = "median"
    elif agg_norm == "min":
        aggfunc = "min"
    elif agg_norm == "max":
        aggfunc = "max"
    else:
        aggfunc = "mean"

    piv = pd.pivot_table(
        dff,
        index=ycol,
        columns=xcol,
        values=vcol,
        aggfunc=aggfunc,
        dropna=False,
    )

    if sort_x:
        piv = piv.reindex(sorted(piv.columns.astype(str)), axis=1)
    if sort_y:
        piv = piv.reindex(sorted(piv.index.astype(str)), axis=0)

    fig = px.imshow(
        piv,
        aspect="auto",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(xaxis_title=xcol, yaxis_title=ycol)
    return fig


@register_plot("funnel")
def funnel(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    """Funnel chart (pipeline / conversion).

    Params:
      - stage: categorical column (funnel steps)
      - value: numeric column
      - group_by: optional categorical column (one funnel per group)
    """
    stage = params.get("stage")
    value = params.get("value")
    group_by = params.get("group_by")

    if not isinstance(stage, str) or not stage:
        raise ValueError("'stage' is required")
    if not isinstance(value, str) or not value:
        raise ValueError("'value' is required")

    required = [stage, value] + ([group_by] if isinstance(group_by, str) and group_by else [])
    require_columns(df, required)

    dff = df[required].copy()
    dff[value] = coerce_numeric(dff[value])
    dff[stage] = dff[stage].astype("string").fillna("(NA)")

    if isinstance(group_by, str) and group_by:
        dff[group_by] = dff[group_by].astype("string").fillna("(NA)")

    dff = dff.dropna(subset=[value])
    if dff.empty:
        raise ValueError("No numeric values for funnel")

    # Aggregate by stage (+ optional group)
    agg = _agg_fn(str(params.get("agg", "sum")))

    group_cols = [stage] + ([group_by] if isinstance(group_by, str) and group_by else [])
    grouped = dff.groupby(group_cols, dropna=False)[value].apply(agg).reset_index(name=value)

    # Keep stage order as it appears in source data
    stage_order = pd.unique(dff[stage]).tolist()

    fig = px.funnel(
        grouped,
        y=stage,
        x=value,
        color=group_by if isinstance(group_by, str) and group_by else None,
        category_orders={stage: [str(s) for s in stage_order]},
    )

    fig.update_layout(xaxis_title=value, yaxis_title=stage)
    return fig


@register_plot("waterfall")
def waterfall(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    """Waterfall chart.

    Params:
      - x: label column
      - y: numeric delta column
      - add_total: bool (default True) -> append total bar
    """
    xcol = params.get("x")
    ycol = params.get("y")

    if not isinstance(xcol, str) or not xcol:
        raise ValueError("'x' is required")
    if not isinstance(ycol, str) or not ycol:
        raise ValueError("'y' is required")

    require_columns(df, [xcol, ycol])

    add_total = param_bool(params, "add_total", True)

    dff = df[[xcol, ycol]].copy()
    dff[xcol] = dff[xcol].astype("string").fillna("(NA)")
    dff[ycol] = coerce_numeric(dff[ycol])
    dff = dff.dropna(subset=[ycol])
    if dff.empty:
        raise ValueError("No numeric values for waterfall")

    labels = dff[xcol].astype(str).tolist()
    values = dff[ycol].to_numpy(dtype=float).tolist()

    measures = ["relative"] * len(values)

    if add_total:
        labels = labels + ["Total"]
        values = values + [float(np.nansum(np.asarray(values, dtype=float)))]
        measures = measures + ["total"]

    fig = go.Figure(
        go.Waterfall(
            name="",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            increasing=dict(marker=dict(color="#2ca02c")),
            decreasing=dict(marker=dict(color="#d62728")),
            totals=dict(marker=dict(color=_base_color(params))),
        )
    )

    fig.update_layout(xaxis_title=xcol, yaxis_title=ycol)
    return fig
