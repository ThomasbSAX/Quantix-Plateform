from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    # Import Flask app
    import app as quantix_app

    flask_app = quantix_app.app
    flask_app.testing = True

    demo_path = Path(__file__).resolve().parents[1] / "data" / "demo_math_lab.csv"
    if not demo_path.exists():
        raise SystemExit(f"Demo CSV introuvable: {demo_path}")

    with flask_app.test_client() as client:
        # 1) upload
        with demo_path.open("rb") as f:
            resp = client.post(
                "/upload",
                data={"file": (f, demo_path.name)},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200, resp.data
        j = resp.get_json()
        assert j and j.get("success") is True, j

        # 2) session status
        st = client.get("/api/session/status").get_json()
        assert st and st.get("success") is True
        assert st.get("has_file") is True
        assert st.get("is_data_file") is True

        # 3) dataset info
        info = client.get("/api/lab/dataset-info").get_json()
        assert info and info.get("success") is True, info

        numeric_cols = info.get("numeric_columns") or []
        cat_cols = info.get("categorical_columns") or []

        # 4) catalog
        cat = client.get("/api/plots/catalog").get_json()
        assert cat and cat.get("success") is True, cat

        # 5) render a default plot
        if len(numeric_cols) >= 2:
            spec = {"plot_type": "corr_heatmap"}
        elif len(numeric_cols) >= 1:
            spec = {"plot_type": "histogram", "x": numeric_cols[0]}
        elif len(cat_cols) >= 1:
            spec = {"plot_type": "count", "x": cat_cols[0]}
        else:
            spec = {"plot_type": "missing_values"}

        r = client.post(
            "/api/plots/render",
            data=json.dumps(spec),
            content_type="application/json",
        )
        assert r.status_code == 200, r.data
        out = r.get_json()
        assert out and out.get("success") is True, out
        assert out.get("plotly_json"), "plotly_json manquant"

        # basic sanity: plotly_json should be a JSON dict when decoded
        decoded = json.loads(out["plotly_json"])
        assert isinstance(decoded, dict)
        assert "data" in decoded and "layout" in decoded

    print("OK: upload + dataset-info + catalog + render Plotly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
