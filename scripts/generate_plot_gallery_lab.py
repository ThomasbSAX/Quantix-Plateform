from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "reports" / "plot_gallery"
PLOT_STUDIO_DIR = OUT_DIR / "plot_studio"
API_DIR = OUT_DIR / "api_tools"


@dataclass
class PlotResult:
    plot_type: str
    status: str  # ok|skipped|error
    out_file: Optional[Path] = None
    message: str = ""
    spec: Optional[Dict[str, Any]] = None


def _prefer(cols: List[str], preferred: List[str]) -> Optional[str]:
    for p in preferred:
        if p in cols:
            return p
    return cols[0] if cols else None


def _build_spec(plot_type: str, info: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    cols = info.get("columns") or []
    numeric = info.get("numeric_columns") or []
    categorical = info.get("categorical_columns") or []
    text_cols = info.get("text_columns") or []
    datetime_cols = info.get("datetime_columns") or []

    # Heuristiques "business" (si présentes dans le CSV)
    num_x = _prefer(numeric, ["ventes", "profit", "cout", "quantite", "remise", "age_client", "satisfaction"])
    num_y = _prefer([c for c in numeric if c != num_x], ["profit", "ventes", "cout", "quantite", "remise", "age_client", "satisfaction"])
    cat = _prefer(categorical, ["region", "produit", "canal"])
    text = _prefer(text_cols, ["commentaire"])
    dt = _prefer(datetime_cols, ["date"]) or ("date" if "date" in cols else None)

    # Plots sans paramètres obligatoires
    if plot_type in {"corr_heatmap", "missing_values"}:
        return {"plot_type": plot_type}, ""

    if plot_type in {"histogram", "ecdf", "kde", "qqplot_normal"}:
        if not num_x:
            return None, "Aucune colonne numérique disponible"
        spec: Dict[str, Any] = {"plot_type": plot_type, "x": num_x}
        # options minimalistes
        if plot_type == "histogram":
            spec["nbins"] = 30
        return spec, ""

    if plot_type in {"box", "violin"}:
        if not num_x:
            return None, "Aucune colonne numérique (y)"
        spec = {"plot_type": plot_type, "y": num_x}
        if cat:
            spec["x"] = cat
        return spec, ""

    if plot_type in {"scatter", "line", "density2d_heatmap", "density2d_contour"}:
        if not (num_x and num_y):
            return None, "Besoin de 2 colonnes numériques (x,y)"
        spec = {"plot_type": plot_type, "x": num_x, "y": num_y}
        if cat:
            spec["color_by"] = cat
        return spec, ""

    if plot_type == "bar":
        if not (cat and num_x):
            return None, "Besoin d'une colonne catégorielle (x) et numérique (y)"
        return {"plot_type": "bar", "x": cat, "y": num_x}, ""

    if plot_type == "count":
        if not cat:
            return None, "Aucune colonne catégorielle/texte pour compter"
        return {"plot_type": "count", "x": cat, "top_n": 30}, ""

    if plot_type == "rolling_mean":
        if not (dt and num_x):
            return None, "Besoin d'une colonne date/temps (x) et numérique (y)"
        return {"plot_type": "rolling_mean", "x": dt, "y": num_x, "window": 3}, ""

    if plot_type in {"lag_plot", "acf"}:
        if not num_x:
            return None, "Aucune colonne numérique (y)"
        return {"plot_type": plot_type, "y": num_x}, ""

    if plot_type in {"parallel_coordinates", "scatter_matrix"}:
        dims = numeric[:6]
        if len(dims) < 2:
            return None, "Besoin d'au moins 2 colonnes numériques"
        spec = {"plot_type": plot_type, "dimensions": dims}
        # parallel_coordinates exige un color numérique (continu). scatter_matrix accepte du catégoriel.
        if plot_type == "scatter_matrix" and cat:
            spec["color_by"] = cat
        return spec, ""

    if plot_type == "radar":
        if not cat:
            return None, "Besoin d'une colonne group_by catégorielle"
        metrics = numeric[:5]
        if len(metrics) < 2:
            return None, "Besoin d'au moins 2 métriques numériques"
        return {"plot_type": "radar", "group_by": cat, "metrics": metrics, "agg": "mean", "top_n": 8}, ""

    if plot_type in {"top_words", "wordcloud"}:
        if not text:
            return None, "Aucune colonne texte"
        spec = {"plot_type": plot_type, "text": text}
        if plot_type == "top_words":
            spec.update({"top_n": 30, "ngram": 1, "min_token_len": 2})
        return spec, ""

    return None, "Plot type non géré par le générateur"


def _write_plot_html(out_path: Path, title: str, plotly_payload: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload_json = json.dumps(plotly_payload)
    safe_title = html.escape(title)

    html_doc = f"""<!doctype html>
<html lang=\"fr\">
<head>
  <meta charset=\"utf-8\"/>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
  <title>{safe_title}</title>
  <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Inter', Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 16px; background: #fafafa; }}
    .card {{ background: white; border: 1px solid #e5e5e5; border-radius: 10px; padding: 14px; }}
    .meta {{ color: #6b7280; font-size: 13px; margin-top: 6px; }}
    #plot {{ width: 100%; height: 620px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; font-size: 12px; background:#0b1020; color:#d1d5db; padding: 12px; border-radius: 8px; overflow:auto; }}
  </style>
</head>
<body>
  <div class=\"card\">
    <div style=\"font-weight:600; color:#111827;\">{safe_title}</div>
    <div class=\"meta\">Rendu via /api/plots/render (Plot Studio)</div>
    <div id=\"plot\"></div>
    <details style=\"margin-top:12px;\"><summary style=\"cursor:pointer;\">Voir le JSON Plotly</summary>
      <pre>{html.escape(payload_json)}</pre>
    </details>
  </div>

<script>
  const fig = {payload_json};
  Plotly.newPlot('plot', fig.data, fig.layout, {{responsive: true}});
</script>
</body>
</html>"""

    out_path.write_text(html_doc, encoding="utf-8")


def _write_json(out_path: Path, obj: Any) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    import app as quantix_app

    # CSV d'entrée: argument ou fallback sur un CSV timeseries plus riche
    import sys
    if len(sys.argv) > 1 and sys.argv[1].strip():
        demo_path = (PROJECT_ROOT / sys.argv[1]).resolve() if not Path(sys.argv[1]).is_absolute() else Path(sys.argv[1]).resolve()
    else:
        demo_path = PROJECT_ROOT / "data" / "demo_math_lab_timeseries.csv"
        if not demo_path.exists():
            demo_path = PROJECT_ROOT / "data" / "demo_math_lab.csv"

    if not demo_path.exists():
        raise SystemExit(f"CSV de démo introuvable: {demo_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_STUDIO_DIR.mkdir(parents=True, exist_ok=True)
    API_DIR.mkdir(parents=True, exist_ok=True)

    flask_app = quantix_app.app
    flask_app.testing = True

    results: List[PlotResult] = []

    with flask_app.test_client() as client:
        # Upload demo
        with demo_path.open("rb") as f:
            up = client.post(
                "/upload",
                data={"file": (f, demo_path.name)},
                content_type="multipart/form-data",
            )
        if up.status_code != 200 or not (up.get_json() or {}).get("success"):
            raise SystemExit(f"Upload failed: {up.status_code} {up.data!r}")

        info = client.get("/api/lab/dataset-info").get_json() or {}
        if not info.get("success"):
            raise SystemExit(f"dataset-info failed: {info}")

        summary = client.get("/api/lab/summary").get_json() or {}
        _write_json(API_DIR / "lab_summary.json", summary)

        catalog_resp = client.get("/api/plots/catalog").get_json() or {}
        if not catalog_resp.get("success"):
            raise SystemExit(f"catalog failed: {catalog_resp}")

        plot_types = (catalog_resp.get("catalog") or {}).get("plot_types") or {}
        for plot_type in sorted(plot_types.keys()):
            spec, why = _build_spec(plot_type, info)
            if not spec:
                results.append(PlotResult(plot_type=plot_type, status="skipped", message=why))
                continue

            r = client.post(
                "/api/plots/render",
                data=json.dumps(spec),
                content_type="application/json",
            )
            data = r.get_json() or {}
            if r.status_code != 200 or not data.get("success"):
                results.append(
                    PlotResult(
                        plot_type=plot_type,
                        status="error",
                        message=data.get("error") or f"HTTP {r.status_code}",
                        spec=spec,
                    )
                )
                continue

            try:
                fig_payload = json.loads(data.get("plotly_json") or "{}")
            except Exception as e:
                results.append(
                    PlotResult(
                        plot_type=plot_type,
                        status="error",
                        message=f"plotly_json invalide: {e}",
                        spec=spec,
                    )
                )
                continue

            label = (plot_types.get(plot_type) or {}).get("label") or plot_type
            out_file = PLOT_STUDIO_DIR / f"{plot_type}.html"
            _write_plot_html(out_file, f"{label} ({plot_type})", fig_payload)
            results.append(PlotResult(plot_type=plot_type, status="ok", out_file=out_file, spec=spec))

        # Tester quelques endpoints "maths/sciences" historiques (matplotlib) pour avoir des PNG
        api_tools = [
            ("pca", "/api/sciences/pca", {"n_components": 2}),
            ("ica", "/api/sciences/ica", {"n_components": 2}),
            ("afc", "/api/sciences/afc", {}),
            ("spectral_summary", "/api/sciences/spectral-summary", {}),
            ("statistical_describe", "/api/sciences/statistical-describe", {}),
        ]
        api_results = []
        for name, url, payload in api_tools:
            try:
                resp = client.post(url, data=json.dumps(payload), content_type="application/json")
                j = resp.get_json() or {}
                _write_json(API_DIR / f"{name}.json", j)
                api_results.append({"name": name, "url": url, "status": "ok" if j.get("success") else "error", "error": j.get("error")})

                # Si image URL, télécharger et sauvegarder
                for k in ["plot_url", "dendrogram_url"]:
                    if j.get(k) and isinstance(j.get(k), str) and j[k].startswith("/uploads/"):
                        img_resp = client.get(j[k])
                        if img_resp.status_code == 200 and img_resp.data:
                            img_name = j[k].split("/uploads/")[-1]
                            out_img = API_DIR / img_name
                            out_img.write_bytes(img_resp.data)
            except Exception as e:
                api_results.append({"name": name, "url": url, "status": "error", "error": str(e)})

        _write_json(API_DIR / "api_tools_index.json", api_results)

    # index.html
    ok = [r for r in results if r.status == "ok"]
    skipped = [r for r in results if r.status == "skipped"]
    errors = [r for r in results if r.status == "error"]

    try:
        src_label = demo_path.relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        src_label = str(demo_path)

    def _row(r: PlotResult) -> str:
        if r.status == "ok" and r.out_file:
            rel = r.out_file.relative_to(OUT_DIR).as_posix()
            return f"<tr><td><code>{html.escape(r.plot_type)}</code></td><td style='color:#059669;'>OK</td><td><a href='{html.escape(rel)}' target='_blank'>ouvrir</a></td><td><code style='font-size:12px;'>{html.escape(json.dumps(r.spec or {}, ensure_ascii=False))}</code></td></tr>"
        if r.status == "skipped":
            return f"<tr><td><code>{html.escape(r.plot_type)}</code></td><td style='color:#6b7280;'>SKIP</td><td></td><td>{html.escape(r.message)}</td></tr>"
        return f"<tr><td><code>{html.escape(r.plot_type)}</code></td><td style='color:#b91c1c;'>ERROR</td><td></td><td>{html.escape(r.message)}<br/><code style='font-size:12px;'>{html.escape(json.dumps(r.spec or {}, ensure_ascii=False))}</code></td></tr>"

    index_html = f"""<!doctype html>
<html lang=\"fr\">
<head>
  <meta charset=\"utf-8\"/>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
  <title>Quantix — Galerie Plot Studio</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Inter', Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 16px; background: #fafafa; }}
    .card {{ background: white; border: 1px solid #e5e5e5; border-radius: 10px; padding: 14px; margin-bottom: 14px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #f3f4f6; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #fff; z-index: 1; }}
    code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }}
    .meta {{ color: #6b7280; font-size: 13px; }}
  </style>
</head>
<body>
  <div class=\"card\">
    <div style=\"font-weight:700; color:#111827; font-size:18px;\">Galerie des visualisations (Plot Studio)</div>
        <div class=\"meta\">Généré depuis <code>{html.escape(src_label)}</code>. OK: {len(ok)} • SKIP: {len(skipped)} • ERROR: {len(errors)}</div>
    <div class=\"meta\">Résumé descriptif JSON: <a href=\"api_tools/lab_summary.json\" target=\"_blank\">ouvrir</a></div>
  </div>

  <div class=\"card\">
    <div style=\"font-weight:600; color:#111827; margin-bottom:8px;\">Résultats</div>
    <div style=\"overflow:auto;\">
      <table>
        <thead><tr><th>plot_type</th><th>statut</th><th>fichier</th><th>notes/spec</th></tr></thead>
        <tbody>
          {''.join(_row(r) for r in results)}
        </tbody>
      </table>
    </div>
  </div>

  <div class=\"card\">
    <div style=\"font-weight:600; color:#111827; margin-bottom:8px;\">API “maths/sciences” (JSON + PNG si dispo)</div>
    <div class=\"meta\">Voir <a href=\"api_tools/api_tools_index.json\" target=\"_blank\">api_tools_index.json</a> et les fichiers dans <code>reports/plot_gallery/api_tools/</code>.</div>
  </div>
</body>
</html>"""

    (OUT_DIR / "index.html").write_text(index_html, encoding="utf-8")
    print(f"OK: galerie générée dans {OUT_DIR}")
    print(f"Ouvrir: {OUT_DIR / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
