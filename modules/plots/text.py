from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .engine import register_plot
from .utils import param_int, param_list_str, require_columns


_TOKEN_RE = re.compile(r"[\w']+", flags=re.UNICODE)


def _base_color(params: Mapping[str, Any]) -> str:
    c = params.get("color")
    return str(c) if c else "#1f77b4"


def _normalize_stopwords(stopwords: Any) -> set:
    if stopwords is None:
        return set()
    if isinstance(stopwords, str):
        # allow comma-separated.
        return {w.strip().lower() for w in stopwords.split(",") if w.strip()}
    if isinstance(stopwords, list):
        return {str(w).strip().lower() for w in stopwords if str(w).strip()}
    return {str(stopwords).strip().lower()}


def _iter_text(series: pd.Series) -> Iterable[str]:
    for v in series.dropna().astype(str).values:
        if v.strip():
            yield v


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _ngrams(tokens: Sequence[str], n: int) -> Iterable[str]:
    if n <= 1:
        yield from tokens
        return
    for i in range(0, max(len(tokens) - n + 1, 0)):
        yield " ".join(tokens[i : i + n])


@register_plot("top_words")
def top_words(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    col = params.get("text")
    if not isinstance(col, str) or not col:
        raise ValueError("'text' is required")
    require_columns(df, [col])

    top_n = param_int(params, "top_n", 30) or 30
    ngram = param_int(params, "ngram", 1) or 1
    min_token_len = param_int(params, "min_token_len", 2) or 2

    stopwords = _normalize_stopwords(params.get("stopwords"))

    counts: Counter = Counter()
    for text in _iter_text(df[col]):
        toks = [t for t in _tokens(text) if len(t) >= min_token_len and t not in stopwords]
        counts.update(_ngrams(toks, ngram))

    if not counts:
        raise ValueError("No tokens found (check stopwords/min length)")

    items = counts.most_common(top_n)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig = px.bar(x=labels, y=values)
    fig.update_traces(marker_color=_base_color(params))
    fig.update_layout(xaxis_title=f"{'ngram' if ngram > 1 else 'mot'}", yaxis_title="Fréquence")
    return fig


@register_plot("wordcloud")
def wordcloud(df: pd.DataFrame, params: Mapping[str, Any]) -> go.Figure:
    """Wordcloud as a Plotly image.

    Requires optional dependency: wordcloud (pip install wordcloud)
    """
    col = params.get("text")
    if not isinstance(col, str) or not col:
        raise ValueError("'text' is required")
    require_columns(df, [col])

    stopwords = _normalize_stopwords(params.get("stopwords"))
    max_words = param_int(params, "max_words", 150) or 150

    text = "\n".join(_iter_text(df[col]))

    try:
        from wordcloud import WordCloud  # type: ignore
    except Exception as e:
        raise ValueError("Optional dependency missing: install 'wordcloud' to use wordcloud plot") from e

    wc = WordCloud(
        width=1400,
        height=800,
        background_color="white",
        stopwords=stopwords,
        max_words=max_words,
        collocations=False,
    ).generate(text)

    img = wc.to_array()

    fig = px.imshow(img)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(coloraxis_showscale=False)
    return fig
