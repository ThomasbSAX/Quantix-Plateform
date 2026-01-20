"""Tiny local demo (optional) to validate plot generation.

This file is safe to ignore in production. It helps you quickly test the module.
"""

import pandas as pd

try:
    # When your project root is on PYTHONPATH.
    from modules.plots import make_figure  # type: ignore
except Exception:
    # Fallback for running this file in isolation.
    from .engine import make_figure


def main():
    df = pd.DataFrame(
        {
            "x": [1, 2, 2, 3, 4, 4, 4, 5],
            "y": [2, 1, 3, 5, 4, 6, 5, 7],
            "cat": ["A", "A", "B", "B", "A", "C", "C", "C"],
            "txt": ["Bonjour ENS", "Data science", "bonjour data", "science science", "ENS", "plotly", "flask", "html"],
        }
    )

    fig = make_figure(
        df,
        {
            "plot_type": "histogram",
            "x": "x",
            "compare_distributions": ["normal"],
            "title": "Histogramme + fit normal",
        },
    )
    fig.show()


if __name__ == "__main__":
    main()
