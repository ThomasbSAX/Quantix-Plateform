"""Utilities to generate professional, web-friendly plots for a Flask-backed HTML UI.

This package is intentionally UI-agnostic:
- Your Flask routes should load/prepare a pandas.DataFrame.
- The frontend sends a JSON "spec" (plot_type + parameters).
- `make_figure(df, spec)` returns a Plotly Figure (serializable to JSON / HTML).
"""

from .engine import make_figure, make_figure_json, get_plot_catalog
from .profile import profile_dataframe
from .encoding import encode_dataframe_columns, encode_unique_as_int

__all__ = [
    "make_figure",
    "make_figure_json",
    "get_plot_catalog",
    "profile_dataframe",
    "encode_unique_as_int",
    "encode_dataframe_columns",
]
