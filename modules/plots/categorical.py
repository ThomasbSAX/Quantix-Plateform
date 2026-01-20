from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .engine import register_plot
from .utils import coerce_numeric, param_bool, param_int, require_columns, top_categories, validate_positive_for_log


def _base_color(params: Mapping[str, Any]) -> str:
    c = params.get("color")
    return str(c) if c else "#1f77b4"


@register_plot("bar")
def bar(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    xcol = params.get("x")
    ycol = params.get("y")
    if not isinstance(xcol, str) or not isinstance(ycol, str) or not xcol or not ycol:
        raise ValueError("'x' and 'y' are required")
    require_columns(df, [xcol, ycol])

    orientation = str(params.get("orientation", "v")).lower()

    fig = px.bar(df, x=xcol if orientation == "v" else ycol, y=ycol if orientation == "v" else xcol)
    fig.update_traces(marker_color=_base_color(params))
    if param_bool(params, "log_y", False):
        validate_positive_for_log(coerce_numeric(df[ycol]).dropna().values, axis_label=ycol)
        fig.update_yaxes(type="log")
    return fig


@register_plot("count")
def count(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    xcol = params.get("x")
    if not isinstance(xcol, str) or not xcol:
        raise ValueError("'x' is required")
    require_columns(df, [xcol])

    top_n = param_int(params, "top_n", 30) or 30
    vc = top_categories(df[xcol], top_n=top_n)

    fig = px.bar(x=vc.index.astype(str), y=vc.values)
    fig.update_traces(marker_color=_base_color(params))
    fig.update_layout(xaxis_title=xcol, yaxis_title="Count")
    return fig
