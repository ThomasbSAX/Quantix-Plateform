"""Smoke-test des endpoints Lab Query/Export + rendu Plot Studio filtré.

Usage:
  python3 scripts/smoke_test_lab_query.py

Ce script:
- upload un CSV de démo en session
- fait un preview filtré via /api/lab/query
- exporte le sous-ensemble via /api/lab/export-query
- rend un plot Plot Studio sur sous-ensemble via /api/plots/render (spec.query)
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    import app as flask_app_module

    client = flask_app_module.app.test_client()

    csv_path = Path("data/demo_math_lab_timeseries.csv")
    if not csv_path.exists():
        raise SystemExit(f"CSV introuvable: {csv_path}")

    with csv_path.open("rb") as f:
        r = client.post(
            "/upload",
            data={"file": (f, csv_path.name)},
            content_type="multipart/form-data",
        )
    j = r.get_json() or {}
    assert r.status_code == 200 and j.get("success") is True, j

    payload = {
        "q": "",
        "filters": [{"column": "region", "op": "equals", "value": "Nord"}],
        "limit": 50,
    }

    r = client.post(
        "/api/lab/query",
        data=json.dumps(payload),
        content_type="application/json",
    )
    j = r.get_json() or {}
    assert r.status_code == 200 and j.get("success") is True, j
    assert "filtered_rows" in j and "total_rows" in j, j
    assert isinstance(j.get("rows"), list), j

    print(
        "OK /api/lab/query",
        f"filtered_rows={j.get('filtered_rows')}",
        f"total_rows={j.get('total_rows')}",
        f"returned={len(j.get('rows') or [])}",
    )

    r = client.post(
        "/api/plots/render",
        data=json.dumps({"plot_type": "histogram", "x": "ventes", "nbins": 20, "query": payload}),
        content_type="application/json",
    )
    j = r.get_json() or {}
    assert r.status_code == 200 and j.get("success") is True, j
    assert j.get("plotly_json"), j

    print("OK /api/plots/render (filtré)")

    r = client.post(
        "/api/lab/export-query",
        data=json.dumps(payload),
        content_type="application/json",
    )
    j = r.get_json() or {}
    assert r.status_code == 200 and j.get("success") is True, j
    assert j.get("download_url"), j

    print("OK /api/lab/export-query", j.get("download_url"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
