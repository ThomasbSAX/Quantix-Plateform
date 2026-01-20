from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

import pandas as pd
import plotly.graph_objects as go

from .catalog import PLOT_CATALOG
from .style import apply_style


PlotBuilder = Callable[[pd.DataFrame, Mapping[str, Any]], go.Figure]


@dataclass(frozen=True)
class PlotError(Exception):
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


_REGISTRY: Dict[str, PlotBuilder] = {}


def register_plot(plot_type: str) -> Callable[[PlotBuilder], PlotBuilder]:
    def decorator(fn: PlotBuilder) -> PlotBuilder:
        if plot_type in _REGISTRY:
            raise RuntimeError(f"Plot type already registered: {plot_type}")
        _REGISTRY[plot_type] = fn
        return fn

    return decorator


def get_plot_catalog() -> dict:
    """Returns a frontend-friendly catalog of supported plot types + params."""
    return PLOT_CATALOG


def make_figure(df: pd.DataFrame, spec: Mapping[str, Any]) -> go.Figure:
    """Build a Plotly figure from a dataframe and a JSON-like spec.

    Expected spec format (minimal):
        {"plot_type": "histogram", "x": "col"}

    The rest of the keys are plot-type specific.
    """
    if not isinstance(spec, Mapping):
        raise PlotError("spec must be a mapping/dict")

    plot_type = spec.get("plot_type")
    if not plot_type or not isinstance(plot_type, str):
        raise PlotError("spec.plot_type is required")

    builder = _REGISTRY.get(plot_type)
    if builder is None:
        supported = ", ".join(sorted(_REGISTRY.keys()))
        raise PlotError(f"Unsupported plot_type '{plot_type}'. Supported: {supported}")

    params: MutableMapping[str, Any] = dict(spec)
    params.pop("plot_type", None)

    fig = builder(df, params)
    fig = apply_style(fig, params)
    return fig


def make_figure_json(df: pd.DataFrame, spec: Mapping[str, Any]) -> dict:
    """Convenience: return a JSON-serializable Plotly figure dict."""
    fig = make_figure(df, spec)
    return fig.to_plotly_json()


# Import builders to register them.
from . import numeric as _numeric  # noqa: E402,F401
from . import categorical as _categorical  # noqa: E402,F401
from . import correlation as _correlation  # noqa: E402,F401
from . import text as _text  # noqa: E402,F401
from . import multivariate as _multivariate  # noqa: E402,F401
from . import advanced as _advanced  # noqa: E402,F401
