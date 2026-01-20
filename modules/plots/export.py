from __future__ import annotations

from typing import Any, Mapping, Optional

import plotly.graph_objects as go


def fig_to_json(fig: go.Figure) -> dict:
    return fig.to_plotly_json()


def fig_to_html(
    fig: go.Figure,
    include_plotlyjs: str | bool = "cdn",
    full_html: bool = False,
    div_id: Optional[str] = None,
) -> str:
    return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=full_html, div_id=div_id)
