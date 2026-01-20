from __future__ import annotations

from typing import Any, Mapping, Optional

import plotly.graph_objects as go


def _pick(params: Mapping[str, Any], key: str, default: Any = None) -> Any:
    v = params.get(key, default)
    return default if v is None else v


def apply_style(fig: go.Figure, params: Mapping[str, Any]) -> go.Figure:
    template = _pick(params, "template", "plotly_white")
    height = _pick(params, "height", 520)
    width = _pick(params, "width", None)
    title = _pick(params, "title", None)

    fig.update_layout(
        template=template,
        height=height,
        width=width,
        title=title,
        margin=dict(l=55, r=25, t=65 if title else 35, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if _pick(params, "transparent_bg", False):
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    return fig
