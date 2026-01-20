from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .engine import register_plot


@register_plot("corr_heatmap")
def corr_heatmap(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    method = str(params.get("method", "pearson")).lower()
    numeric = df.select_dtypes(include=["number"]).copy()
    if numeric.shape[1] < 2:
        raise ValueError("Need at least two numeric columns for correlation")

    corr = numeric.corr(method=method)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(xaxis_title="", yaxis_title="")
    return fig
