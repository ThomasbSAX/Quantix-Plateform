from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .distributions import fit_distribution, pdf
from .engine import register_plot
from .utils import (
    coerce_datetime,
    coerce_numeric,
    finite_1d,
    param_bool,
    param_float,
    param_int,
    param_list_float,
    param_list_str,
    require_columns,
    validate_positive_for_log,
)


def _base_color(params: Mapping[str, Any]) -> str:
    c = params.get("color")
    return str(c) if c else "#1f77b4"


def _inv_norm_cdf(p: np.ndarray) -> np.ndarray:
    """Approx inverse CDF for standard normal.

    Pure-numpy approximation (Acklam-style), good enough for QQ plots.
    Input p in (0,1).
    """
    p = np.asarray(p, dtype=float)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)

    # Coefficients for rational approximation.
    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )

    plow = 0.02425
    phigh = 1 - plow

    x = np.empty_like(p)

    # Lower region
    mask = p < plow
    if np.any(mask):
        q = np.sqrt(-2 * np.log(p[mask]))
        x[mask] = (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    # Central region
    mask = (p >= plow) & (p <= phigh)
    if np.any(mask):
        q = p[mask] - 0.5
        r = q * q
        x[mask] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )

    # Upper region
    mask = p > phigh
    if np.any(mask):
        q = np.sqrt(-2 * np.log(1 - p[mask]))
        x[mask] = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    return x


@register_plot("histogram")
def histogram(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    xcol = params.get("x")
    if not isinstance(xcol, str) or not xcol:
        raise ValueError("'x' is required")
    require_columns(df, [xcol])

    x = coerce_numeric(df[xcol])
    x = x.dropna()
    nbins = param_int(params, "nbins", 30) or 30
    normalize = params.get("normalize")  # plotly: '', 'probability', 'percent', 'density', 'probability density'

    fig = px.histogram(
        x=x,
        nbins=nbins,
        histnorm=normalize,
        opacity=float(params.get("opacity", 0.85) or 0.85),
    )
    fig.update_traces(marker_color=_base_color(params))
    fig.update_layout(bargap=0.02, xaxis_title=xcol)

    if param_bool(params, "log_x", False):
        validate_positive_for_log(x.values, axis_label=xcol)
        fig.update_xaxes(type="log")

    if param_bool(params, "rug", False):
        # Rug as a thin scatter near y=0.
        xrug = finite_1d(x.values)
        if xrug.size:
            fig.add_trace(
                go.Scatter(
                    x=xrug,
                    y=np.zeros_like(xrug),
                    mode="markers",
                    name="rug",
                    marker=dict(size=6, color=_base_color(params), opacity=0.35),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            _add_stats_lines_1d(fig, x.values, params)
    _overlay_distributions(fig, x.values, normalize, params)
    return fig


@register_plot("ecdf")
def ecdf(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    xcol = params.get("x")
    if not isinstance(xcol, str) or not xcol:
        raise ValueError("'x' is required")
    require_columns(df, [xcol])

    x = finite_1d(coerce_numeric(df[xcol]).dropna().values)
    if x.size < 2:
        raise ValueError("Not enough numeric data for ECDF")

    if param_bool(params, "log_x", False):
        validate_positive_for_log(x, axis_label=xcol)

    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="ECDF",
            line=dict(color=_base_color(params), width=3),
            hovertemplate=f"{xcol}=%{{x:.4g}}<br>F=%{{y:.4g}}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title=xcol, yaxis_title="F(x)", yaxis=dict(range=[0, 1]))

    if param_bool(params, "log_x", False):
        fig.update_xaxes(type="log")

    _add_stats_lines_1d(fig, x, params, y_max=1.0)
    return fig


@register_plot("kde")
def kde(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    xcol = params.get("x")
    if not isinstance(xcol, str) or not xcol:
        raise ValueError("'x' is required")
    require_columns(df, [xcol])

    x = finite_1d(coerce_numeric(df[xcol]).dropna().values)
    if x.size < 5:
        raise ValueError("Not enough numeric data to compute density")

    if param_bool(params, "log_x", False):
        validate_positive_for_log(x, axis_label=xcol)

    bandwidth = param_float(params, "bandwidth", None)

    grid = np.linspace(np.min(x), np.max(x), 400)
    density = _gaussian_kde_1d(x, grid, bandwidth=bandwidth)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=density,
            mode="lines",
            name="KDE",
            line=dict(color=_base_color(params), width=3),
        )
    )

    _overlay_distributions(fig, x, normalize="density", params=params)
    _add_stats_lines_1d(fig, x, params, y_max=float(np.max(density)) if density.size else None)
    fig.update_layout(xaxis_title=xcol, yaxis_title="Densité")
    if param_bool(params, "log_x", False):
        fig.update_xaxes(type="log")
    return fig


@register_plot("qqplot_normal")
def qqplot_normal(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    """QQ plot vs standard normal.

    Plots theoretical normal quantiles (x) vs sample quantiles (y).
    """
    xcol = params.get("x")
    if not isinstance(xcol, str) or not xcol:
        raise ValueError("'x' is required")
    require_columns(df, [xcol])

    x = finite_1d(coerce_numeric(df[xcol]).dropna().values)
    if x.size < 8:
        raise ValueError("Not enough data for QQ plot")

    sample = np.sort(x)
    n = sample.size
    probs = (np.arange(1, n + 1, dtype=float) - 0.5) / float(n)
    theo = _inv_norm_cdf(probs)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=theo,
            y=sample,
            mode="markers",
            name="points",
            marker=dict(size=7, color=_base_color(params), opacity=float(params.get("opacity", 0.85) or 0.85)),
            hovertemplate="z=%{x:.4g}<br>q=%{y:.4g}<extra></extra>",
        )
    )

    # Reference line: fit y = a + b x
    if param_bool(params, "fit_line", True):
        b, a = np.polyfit(theo, sample, deg=1)
        xs = np.linspace(float(np.min(theo)), float(np.max(theo)), 100)
        ys = a + b * xs
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="ligne ajustée",
                line=dict(color="#111111", width=2),
                hoverinfo="skip",
            )
        )

    fig.update_layout(xaxis_title="Quantiles N(0,1)", yaxis_title=f"Quantiles de {xcol}")
    return fig


@register_plot("density2d_heatmap")
def density2d_heatmap(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    xcol = params.get("x")
    ycol = params.get("y")
    if not isinstance(xcol, str) or not isinstance(ycol, str) or not xcol or not ycol:
        raise ValueError("'x' and 'y' are required")
    require_columns(df, [xcol, ycol])

    nbinsx = param_int(params, "nbinsx", 40) or 40
    nbinsy = param_int(params, "nbinsy", 40) or 40

    fig = px.density_heatmap(df, x=xcol, y=ycol, nbinsx=nbinsx, nbinsy=nbinsy, color_continuous_scale="Viridis")

    if param_bool(params, "log_x", False):
        validate_positive_for_log(coerce_numeric(df[xcol]).dropna().values, axis_label=xcol)
        fig.update_xaxes(type="log")
    if param_bool(params, "log_y", False):
        validate_positive_for_log(coerce_numeric(df[ycol]).dropna().values, axis_label=ycol)
        fig.update_yaxes(type="log")

    return fig


@register_plot("density2d_contour")
def density2d_contour(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    xcol = params.get("x")
    ycol = params.get("y")
    if not isinstance(xcol, str) or not isinstance(ycol, str) or not xcol or not ycol:
        raise ValueError("'x' and 'y' are required")
    require_columns(df, [xcol, ycol])

    fig = px.density_contour(df, x=xcol, y=ycol)
    fig.update_traces(contours_coloring="heatmap")

    if param_bool(params, "log_x", False):
        validate_positive_for_log(coerce_numeric(df[xcol]).dropna().values, axis_label=xcol)
        fig.update_xaxes(type="log")
    if param_bool(params, "log_y", False):
        validate_positive_for_log(coerce_numeric(df[ycol]).dropna().values, axis_label=ycol)
        fig.update_yaxes(type="log")

    return fig


@register_plot("box")
def box(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    ycol = params.get("y")
    xcol = params.get("x")
    if not isinstance(ycol, str) or not ycol:
        raise ValueError("'y' is required")
    require_columns(df, [ycol] + ([xcol] if isinstance(xcol, str) and xcol else []))

    points = params.get("points", "outliers")  # 'all', 'outliers', False

    fig = px.box(
        df,
        x=xcol if isinstance(xcol, str) and xcol else None,
        y=ycol,
        points=points,
    )
    fig.update_traces(marker_color=_base_color(params))
    if param_bool(params, "log_y", False):
        validate_positive_for_log(coerce_numeric(df[ycol]).dropna().values, axis_label=ycol)
        fig.update_yaxes(type="log")
    return fig


@register_plot("violin")
def violin(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    ycol = params.get("y")
    xcol = params.get("x")
    if not isinstance(ycol, str) or not ycol:
        raise ValueError("'y' is required")
    require_columns(df, [ycol] + ([xcol] if isinstance(xcol, str) and xcol else []))

    points = params.get("points", "outliers")

    fig = px.violin(
        df,
        x=xcol if isinstance(xcol, str) and xcol else None,
        y=ycol,
        points=points,
        box=True,
    )
    fig.update_traces(meanline_visible=True)
    fig.update_traces(marker_color=_base_color(params))
    if param_bool(params, "log_y", False):
        validate_positive_for_log(coerce_numeric(df[ycol]).dropna().values, axis_label=ycol)
        fig.update_yaxes(type="log")
    return fig


@register_plot("scatter")
def scatter(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    xcol = params.get("x")
    ycol = params.get("y")
    if not isinstance(xcol, str) or not isinstance(ycol, str) or not xcol or not ycol:
        raise ValueError("'x' and 'y' are required")
    require_columns(df, [xcol, ycol])

    color_by = params.get("color_by")
    if isinstance(color_by, str) and color_by:
        require_columns(df, [color_by])

    fig = px.scatter(
        df,
        x=xcol,
        y=ycol,
        color=color_by if isinstance(color_by, str) and color_by else None,
        opacity=float(params.get("opacity", 0.85) or 0.85),
    )

    if not (isinstance(color_by, str) and color_by):
        fig.update_traces(marker=dict(color=_base_color(params)))

    if str(params.get("trendline", "")).lower() in {"linear", "ols"}:
        _add_linear_trendline(fig, df, xcol, ycol)

    _add_regression_and_analysis(fig, df, xcol, ycol, params)

    if param_bool(params, "log_x", False):
        validate_positive_for_log(coerce_numeric(df[xcol]).dropna().values, axis_label=xcol)
        fig.update_xaxes(type="log")
    if param_bool(params, "log_y", False):
        validate_positive_for_log(coerce_numeric(df[ycol]).dropna().values, axis_label=ycol)
        fig.update_yaxes(type="log")

    return fig


@register_plot("line")
def line(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    xcol = params.get("x")
    ycol = params.get("y")
    if not isinstance(xcol, str) or not isinstance(ycol, str) or not xcol or not ycol:
        raise ValueError("'x' and 'y' are required")
    require_columns(df, [xcol, ycol])

    markers = param_bool(params, "markers", False)

    x = df[xcol]
    if pd.api.types.is_object_dtype(x) or pd.api.types.is_string_dtype(x):
        x = coerce_datetime(x)

    dff = df.copy()
    dff[xcol] = x

    color_by = params.get("color_by")
    if isinstance(color_by, str) and color_by:
        require_columns(dff, [color_by])

    fig = px.line(
        dff,
        x=xcol,
        y=ycol,
        color=color_by if isinstance(color_by, str) and color_by else None,
        markers=markers,
    )

    if not (isinstance(color_by, str) and color_by):
        fig.update_traces(line=dict(color=_base_color(params), width=3))

    # Advanced regression/analysis only makes sense for numeric x.
    _add_regression_and_analysis(fig, dff, xcol, ycol, params)

    if param_bool(params, "log_x", False):
        validate_positive_for_log(coerce_numeric(dff[xcol]).dropna().values, axis_label=xcol)
        fig.update_xaxes(type="log")
    if param_bool(params, "log_y", False):
        validate_positive_for_log(coerce_numeric(dff[ycol]).dropna().values, axis_label=ycol)
        fig.update_yaxes(type="log")

    return fig


def _gaussian_kde_1d(x: np.ndarray, grid: np.ndarray, bandwidth: Optional[float]) -> np.ndarray:
    # Lightweight KDE (no SciPy dependency). Gaussian kernel.
    x = np.asarray(x, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n = x.size

    if bandwidth is None or bandwidth <= 0:
        # Scott's rule
        std = np.std(x, ddof=1)
        bandwidth = 1.06 * std * (n ** (-1 / 5)) if std > 0 else 1.0
        bandwidth = float(max(bandwidth, 1e-6))

    u = (grid[:, None] - x[None, :]) / bandwidth
    k = np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
    return k.mean(axis=1) / bandwidth


def _overlay_distributions(fig: go.Figure, x: np.ndarray, normalize: Any, params: Mapping[str, Any]) -> None:
    names = param_list_str(params, "compare_distributions")
    if not names:
        return

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return

    # Decide scaling.
    is_density = str(normalize).lower() in {"density", "probability density"} or str(normalize).lower() == "density"

    grid = np.linspace(np.min(x), np.max(x), 400)

    for name in names:
        nm = name.strip().lower()
        fr = fit_distribution(x, nm)
        if fr is None:
            continue
        y = pdf(fr.name, fr.params, grid)
        if y is None:
            continue

        if not is_density:
            # Approx scale PDF to histogram counts.
            nbins = int(params.get("nbins", 30) or 30)
            bin_width = (grid.max() - grid.min()) / max(nbins, 1)
            y = y * x.size * bin_width

        fig.add_trace(
            go.Scatter(
                x=grid,
                y=y,
                mode="lines",
                name=f"fit: {fr.name}",
                line=dict(width=3, dash="dash"),
            )
        )


def _add_linear_trendline(fig: go.Figure, df: pd.DataFrame, xcol: str, ycol: str) -> None:
    x = coerce_numeric(df[xcol]).to_numpy(dtype=float)
    y = coerce_numeric(df[ycol]).to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 5:
        return

    m, b = np.polyfit(x, y, deg=1)
    xs = np.linspace(np.min(x), np.max(x), 200)
    ys = m * xs + b

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="trend (linéaire)",
            line=dict(color="#111111", width=3),
        )
    )


def _parse_stats_lines(params: Mapping[str, Any]) -> list[str]:
    v = params.get("stats_lines")
    if v is None or v is False:
        return []
    if isinstance(v, list):
        items = [str(x).strip().lower() for x in v]
    else:
        items = [s.strip().lower() for s in str(v).split(",") if s.strip()]
    return [it for it in items if it]


def _stat_value(x: np.ndarray, token: str) -> Optional[float]:
    token = token.strip().lower()
    if x.size == 0:
        return None
    if token == "mean":
        return float(np.mean(x))
    if token in {"median", "med"}:
        return float(np.median(x))

    # Quantile tokens: q0.25, q25, p25
    if token.startswith("q") or token.startswith("p"):
        qraw = token[1:]
        try:
            q = float(qraw)
        except Exception:
            return None
        if q > 1:
            q = q / 100.0
        if not (0.0 <= q <= 1.0):
            return None
        return float(np.quantile(x, q))

    return None


def _add_stats_lines_1d(
    fig: go.Figure,
    x_values: np.ndarray,
    params: Mapping[str, Any],
    *,
    y_max: Optional[float] = None,
) -> None:
    tokens = _parse_stats_lines(params)
    if not tokens:
        return

    x = np.asarray(x_values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return

    if y_max is None:
        # Try to infer from existing traces.
        y_max = None
        for tr in fig.data:
            y = getattr(tr, "y", None)
            if y is None:
                continue
            try:
                ym = float(np.nanmax(np.asarray(y, dtype=float)))
            except Exception:
                continue
            if np.isfinite(ym):
                y_max = max(y_max or 0.0, ym)
        if y_max is None:
            y_max = 1.0

    line_color = str(params.get("stats_color") or "#333333")

    for tok in tokens:
        v = _stat_value(x, tok)
        if v is None or not np.isfinite(v):
            continue
        fig.add_trace(
            go.Scatter(
                x=[v, v],
                y=[0, float(y_max)],
                mode="lines",
                name=tok,
                line=dict(color=line_color, width=2, dash="dot"),
                hovertemplate=f"{tok}: {v:.4g}<extra></extra>",
            )
        )


def _parse_regression_kind(params: Mapping[str, Any]) -> Optional[int]:
    kind = str(params.get("regression", "")).strip().lower()
    if not kind:
        return None
    if kind in {"linear", "lin", "ols"}:
        return 1
    if kind in {"poly2", "quadratic"}:
        return 2
    if kind in {"poly3", "cubic"}:
        return 3
    if kind.startswith("poly"):
        try:
            return int(kind.replace("poly", ""))
        except Exception:
            return None
    return None


def _add_regression_and_analysis(
    fig: go.Figure,
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    params: Mapping[str, Any],
) -> None:
    deg = _parse_regression_kind(params)
    if deg is None:
        return

    x = coerce_numeric(df[xcol]).to_numpy(dtype=float)
    y = coerce_numeric(df[ycol]).to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < max(6, deg + 2):
        return

    # Fit polynomial regression.
    coeffs = np.polyfit(x, y, deg=deg)
    p = np.poly1d(coeffs)

    xs = np.linspace(float(np.min(x)), float(np.max(x)), 400)
    ys = p(xs)

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=f"régression (deg {deg})",
            line=dict(color="#111111", width=3),
        )
    )

    # Gradient/angle visualization along the regression curve.
    if param_bool(params, "show_gradient", False):
        dp = p.deriv(m=1)
        slopes = dp(xs)
        mode = str(params.get("gradient_mode", "angle")).strip().lower()  # angle|slope
        if mode == "slope":
            z = slopes
            z_title = "pente"
        else:
            z = np.degrees(np.arctan(slopes))
            z_title = "angle (deg)"

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=f"gradient ({z_title})",
                marker=dict(
                    size=6,
                    color=z,
                    colorscale="RdBu",
                    showscale=True,
                    colorbar=dict(title=z_title),
                ),
                hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<br>g=%{marker.color:.4g}<extra></extra>",
            )
        )

    # Highlight min/max on fitted curve.
    if param_bool(params, "highlight_extrema", False):
        i_min = int(np.nanargmin(ys))
        i_max = int(np.nanargmax(ys))
        for label, i in [("min", i_min), ("max", i_max)]:
            fig.add_trace(
                go.Scatter(
                    x=[float(xs[i])],
                    y=[float(ys[i])],
                    mode="markers+text",
                    name=f"{label} (fit)",
                    marker=dict(size=12, color="#d62728" if label == "max" else "#1f77b4", symbol="star"),
                    text=[label],
                    textposition="top center",
                    hovertemplate=f"{label} (fit)<br>x=%{{x:.4g}}<br>y=%{{y:.4g}}<extra></extra>",
                )
            )

    # Highlight top residuals (outliers) among observed points.
    top_k = param_int(params, "highlight_top_residuals", 0) or 0
    if top_k > 0:
        y_hat = p(x)
        resid = y - y_hat
        idx = np.argsort(np.abs(resid))[-top_k:]
        fig.add_trace(
            go.Scatter(
                x=x[idx],
                y=y[idx],
                mode="markers+text",
                name=f"top résidus (k={top_k})",
                marker=dict(size=12, color="#ff7f0e", symbol="triangle-up"),
                text=[f"r={float(resid[i]):.3g}" for i in idx],
                textposition="top center",
                hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<br>res=%{text}<extra></extra>",
            )
        )

    # Tangents at chosen x.
    tangent_xs = param_list_float(params, "tangent_at")
    if tangent_xs:
        dp = p.deriv(m=1)
        x_min, x_max = float(np.min(x)), float(np.max(x))
        span = (x_max - x_min) * 0.18
        for x0 in tangent_xs:
            if not np.isfinite(x0) or not (x_min <= x0 <= x_max):
                continue
            y0 = float(p(x0))
            m = float(dp(x0))
            x1, x2 = x0 - span, x0 + span
            y1, y2 = y0 + m * (x1 - x0), y0 + m * (x2 - x0)
            fig.add_trace(
                go.Scatter(
                    x=[x1, x2],
                    y=[y1, y2],
                    mode="lines",
                    name=f"tangente @ {x0:g}",
                    line=dict(color="#444444", width=2, dash="dash"),
                    hovertemplate=f"x0={x0:g}<br>slope={m:.4g}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[x0],
                    y=[y0],
                    mode="markers",
                    name=f"point tangence @ {x0:g}",
                    marker=dict(size=10, color="#444444", symbol="x"),
                    showlegend=False,
                )
            )

    # Stationary points: p'(x)=0
    if param_bool(params, "show_stationary_points", False) and deg >= 2:
        dp = p.deriv(m=1)
        roots = np.roots(dp)
        _add_real_roots_as_markers(fig, p, roots, x_min=float(np.min(x)), x_max=float(np.max(x)), name="stationnaire")

    # Inflection points: p''(x)=0
    if param_bool(params, "show_inflection_points", False) and deg >= 3:
        ddp = p.deriv(m=2)
        roots = np.roots(ddp)
        _add_real_roots_as_markers(fig, p, roots, x_min=float(np.min(x)), x_max=float(np.max(x)), name="inflexion")

    # Fixed points: p(x)=x => p(x)-x=0
    if param_bool(params, "show_fixed_points", False):
        q = p - np.poly1d([1, 0])
        roots = np.roots(q)
        _add_real_roots_as_markers(fig, p, roots, x_min=float(np.min(x)), x_max=float(np.max(x)), name="point fixe", symbol="diamond")


def _add_real_roots_as_markers(
    fig: go.Figure,
    p: np.poly1d,
    roots: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    name: str,
    symbol: str = "circle",
) -> None:
    xs: list[float] = []
    ys: list[float] = []
    for r in np.asarray(roots):
        if not np.isfinite(r):
            continue
        if abs(np.imag(r)) > 1e-6:
            continue
        xr = float(np.real(r))
        if x_min <= xr <= x_max:
            xs.append(xr)
            ys.append(float(p(xr)))
    if not xs:
        return
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name=name,
            marker=dict(size=11, color="#d62728", symbol=symbol),
            hovertemplate=f"{name}<br>x=%{{x:.4g}}<br>y=%{{y:.4g}}<extra></extra>",
        )
    )


@register_plot("missing_values")
def missing_values(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    """Bar chart of missing values by column."""
    top_n = param_int(params, "top_n", 30) or 30
    include_zero = param_bool(params, "include_zero", False)

    miss = df.isna().sum().sort_values(ascending=False)
    if not include_zero:
        miss = miss[miss > 0]

    if top_n and top_n > 0:
        miss = miss.head(top_n)

    if miss.empty:
        raise ValueError("Aucune valeur manquante détectée")

    fig = px.bar(x=miss.index.astype(str), y=miss.values)
    fig.update_traces(marker_color=_base_color(params))
    fig.update_layout(xaxis_title="Colonne", yaxis_title="Valeurs manquantes")
    return fig


@register_plot("rolling_mean")
def rolling_mean(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    """Time series line + rolling mean.

    Expects:
      - x: datetime-like column
      - y: numeric column
    """
    xcol = params.get("x")
    ycol = params.get("y")
    if not isinstance(xcol, str) or not isinstance(ycol, str) or not xcol or not ycol:
        raise ValueError("'x' and 'y' are required")
    require_columns(df, [xcol, ycol])

    window = param_int(params, "window", 10) or 10
    window = max(int(window), 2)
    markers = param_bool(params, "markers", False)

    x = coerce_datetime(df[xcol])
    y = coerce_numeric(df[ycol])

    dff = pd.DataFrame({xcol: x, ycol: y}).dropna()
    if dff.empty:
        raise ValueError("Not enough data after dropping missing values")

    dff = dff.sort_values(xcol)
    roll = dff[ycol].rolling(window=window, min_periods=max(2, window // 3)).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dff[xcol],
            y=dff[ycol],
            mode="lines+markers" if markers else "lines",
            name=ycol,
            line=dict(color=_base_color(params), width=2),
            opacity=float(params.get("opacity", 0.65) or 0.65),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff[xcol],
            y=roll,
            mode="lines",
            name=f"Moyenne mobile ({window})",
            line=dict(color="#111827", width=3),
        )
    )

    fig.update_layout(xaxis_title=xcol, yaxis_title=ycol)

    if param_bool(params, "log_y", False):
        validate_positive_for_log(dff[ycol].dropna().values, axis_label=ycol)
        fig.update_yaxes(type="log")

    return fig


@register_plot("lag_plot")
def lag_plot(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    """Lag plot y(t) vs y(t-lag)."""
    ycol = params.get("y")
    if not isinstance(ycol, str) or not ycol:
        raise ValueError("'y' is required")
    require_columns(df, [ycol])

    lag = param_int(params, "lag", 1) or 1
    lag = max(int(lag), 1)

    y = finite_1d(coerce_numeric(df[ycol]).dropna().values)
    if y.size <= lag + 2:
        raise ValueError("Not enough data for the requested lag")

    x_lag = y[:-lag]
    y_now = y[lag:]

    fig = px.scatter(x=x_lag, y=y_now)
    fig.update_traces(marker=dict(color=_base_color(params), opacity=float(params.get("opacity", 0.85) or 0.85)))
    fig.update_layout(xaxis_title=f"{ycol}(t-{lag})", yaxis_title=f"{ycol}(t)")
    return fig


@register_plot("acf")
def acf(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    """Autocorrelation function (ACF) bar chart."""
    ycol = params.get("y")
    if not isinstance(ycol, str) or not ycol:
        raise ValueError("'y' is required")
    require_columns(df, [ycol])

    x = finite_1d(coerce_numeric(df[ycol]).dropna().values)
    if x.size < 6:
        raise ValueError("Not enough data for ACF")

    nlags = param_int(params, "nlags", 40) or 40
    nlags = max(int(nlags), 1)
    nlags = min(nlags, int(x.size) - 1)

    lags = np.arange(1, nlags + 1)
    acfs: list[float] = []
    for k in lags:
        a = x[:-k]
        b = x[k:]
        if a.size < 3:
            break
        c = float(np.corrcoef(a, b)[0, 1])
        acfs.append(c)

    if not acfs:
        raise ValueError("ACF computation failed")

    fig = px.bar(x=lags[: len(acfs)], y=acfs)
    fig.update_traces(marker_color=_base_color(params))
    fig.update_layout(xaxis_title="Lag", yaxis_title="Autocorrélation")
    return fig
