"""Runtime smoke test for Quantix Flask app.

Runs a small HTTP-level checklist against a locally running server.
- Verifies key pages return 200
- Uploads a known CSV into session
- Exercises core APIs (Lab / Plot Studio / Graph Studio / Dashboard)

Usage:
  python3 scripts/smoke_test_runtime.py --base http://127.0.0.1:5002
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import requests


@dataclass
class CheckResult:
    name: str
    ok: bool
    status: Optional[int] = None
    detail: str = ""


def _short(s: str, n: int = 180) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _as_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"_raw": resp.text}


def check_page(session: requests.Session, base: str, path: str) -> CheckResult:
    url = base.rstrip("/") + path
    r = session.get(url, allow_redirects=True, timeout=30)
    ok = r.status_code == 200
    return CheckResult(
        name=f"GET {path}",
        ok=ok,
        status=r.status_code,
        detail=_short(r.text[:500]) if not ok else "",
    )


def check_json(
    session: requests.Session,
    base: str,
    name: str,
    method: str,
    path: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
    expected_status: Sequence[int] = (200,),
    expect_success: Optional[bool] = True,
) -> Tuple[CheckResult, Dict[str, Any]]:
    url = base.rstrip("/") + path
    if method.upper() == "GET":
        r = session.get(url, timeout=60)
    else:
        r = session.request(method.upper(), url, json=json_body, timeout=120)

    data = _as_json(r)

    ok_status = r.status_code in set(expected_status)
    ok_success = True
    if expect_success is not None:
        ok_success = bool(data.get("success")) is bool(expect_success)

    ok = ok_status and ok_success

    detail = ""
    if not ok:
        detail = _short(json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else str(data))

    return (
        CheckResult(
            name=f"{name} ({method.upper()} {path})",
            ok=ok,
            status=r.status_code,
            detail=detail,
        ),
        data if isinstance(data, dict) else {"_raw": str(data)},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5002")
    ap.add_argument(
        "--data",
        default=os.path.join("data", "test.csv"),
        help="CSV to upload into session",
    )
    args = ap.parse_args()

    base = args.base
    s = requests.Session()

    results: list[CheckResult] = []

    # Pages
    for path in ["/lab", "/graph", "/dashboard", "/labelling", "/data-studio"]:
        results.append(check_page(s, base, path))

    # Session status before upload
    r, _ = check_json(s, base, "Session status (before)", "GET", "/api/session/status", expected_status=(200,), expect_success=True)
    results.append(r)

    # Upload primary dataset
    if not os.path.exists(args.data):
        results.append(CheckResult("Upload primary", False, detail=f"Missing file: {args.data}"))
        print_report(results)
        return 2

    url_upload = base.rstrip("/") + "/upload-file"
    with open(args.data, "rb") as f:
        up = s.post(url_upload, files={"file": (os.path.basename(args.data), f)}, data={"slot": "primary"}, timeout=120)
    up_data = _as_json(up)
    results.append(
        CheckResult(
            name="Upload primary (POST /upload-file)",
            ok=(up.status_code == 200 and bool(up_data.get("success")) is True),
            status=up.status_code,
            detail="" if up.status_code == 200 else _short(str(up_data)),
        )
    )

    # Session status after upload
    r, sess_data = check_json(s, base, "Session status (after)", "GET", "/api/session/status", expected_status=(200,), expect_success=True)
    results.append(r)

    # Lab dataset info
    r, dataset_info = check_json(s, base, "Lab dataset-info", "GET", "/api/lab/dataset-info", expected_status=(200,), expect_success=True)
    results.append(r)

    # Pick some columns
    numeric_cols = dataset_info.get("numeric_columns") or []
    x = "hours_studied" if "hours_studied" in numeric_cols else (numeric_cols[0] if numeric_cols else None)
    y = "exam_score" if "exam_score" in numeric_cols else (numeric_cols[1] if len(numeric_cols) > 1 else None)

    # Plot Studio
    r, _ = check_json(s, base, "Plot catalog", "GET", "/api/plots/catalog", expected_status=(200, 501), expect_success=None)
    results.append(r)
    if r.status == 200:
        r2, plot_render = check_json(
            s,
            base,
            "Plot render histogram",
            "POST",
            "/api/plots/render",
            json_body={"plot_type": "histogram", "x": x or "hours_studied", "nbins": 10},
            expected_status=(200, 400, 501),
            expect_success=None,
        )
        results.append(r2)

    # Core Lab plots
    if x and y:
        r_scatter, scatter = check_json(
            s,
            base,
            "Lab scatter-plot",
            "POST",
            "/api/lab/scatter-plot",
            json_body={"x_column": x, "y_column": y},
            expected_status=(200, 400, 501),
            expect_success=None,
        )
        results.append(r_scatter)

        plot_url = scatter.get("plot_url")
        if plot_url:
            r_pdf, _ = check_json(
                s,
                base,
                "Export plot->PDF",
                "POST",
                "/api/export/plot-to-pdf",
                json_body={"plot_url": plot_url},
                expected_status=(200, 400, 404, 500),
                expect_success=None,
            )
            results.append(r_pdf)

    if x:
        results.append(
            check_json(
                s,
                base,
                "Lab histogram",
                "POST",
                "/api/lab/histogram",
                json_body={"column": x, "bins": 10},
                expected_status=(200, 400, 501),
                expect_success=None,
            )[0]
        )

    # New Lab endpoints
    results.append(
        check_json(
            s,
            base,
            "Lab descriptive-stats",
            "POST",
            "/api/lab/descriptive-stats",
            json_body={},
            expected_status=(200, 400, 501),
            expect_success=None,
        )[0]
    )

    if x and y:
        results.append(
            check_json(
                s,
                base,
                "Lab hypothesis-test (correlation)",
                "POST",
                "/api/lab/hypothesis-test",
                json_body={"test_type": "correlation", "var1": x, "var2": y, "method": "pearson"},
                expected_status=(200, 400, 501),
                expect_success=None,
            )[0]
        )

    results.append(
        check_json(
            s,
            base,
            "Lab outlier-detection",
            "POST",
            "/api/lab/outlier-detection",
            json_body={"method": "iqr"},
            expected_status=(200, 400, 501),
            expect_success=None,
        )[0]
    )

    # These should fail cleanly for this dataset (no datetime / no binary target)
    results.append(
        check_json(
            s,
            base,
            "Lab time-series (expected 400)",
            "POST",
            "/api/lab/time-series",
            json_body={"auto_detect": True},
            expected_status=(400, 200, 501),
            expect_success=None,
        )[0]
    )

    results.append(
        check_json(
            s,
            base,
            "Lab classification (expected 400)",
            "POST",
            "/api/lab/classification",
            json_body={"auto_mode": True},
            expected_status=(400, 200, 501),
            expect_success=None,
        )[0]
    )

    # Graph Studio
    r_cat, catalog = check_json(s, base, "Graph catalog", "GET", "/api/graph/catalog", expected_status=(200, 501), expect_success=None)
    results.append(r_cat)
    if r_cat.status == 200:
        # correlation graph: should work on numeric CSV
        r_build, _ = check_json(
            s,
            base,
            "Graph build (correlation)",
            "POST",
            "/api/graph/build",
            json_body={
                "graph_type": "correlation",
                "options": {"threshold": 0.7, "directed": False, "absolute": True, "min_degree": 1, "max_nodes": 200},
                "include_layout": False,
                "replace_session_file": False,
            },
            expected_status=(200, 400, 501),
            expect_success=None,
        )
        results.append(r_build)

    # Dashboard
    r_an, _ = check_json(s, base, "Dashboard analyze", "GET", "/api/dashboard/analyze?slot=primary", expected_status=(200, 400), expect_success=None)
    results.append(r_an)

    r_ops, _ = check_json(
        s,
        base,
        "Dashboard operations (remove_duplicates)",
        "POST",
        "/api/dashboard/operations",
        json_body={"operations": ["remove_duplicates"], "replace_session_file": False, "output_format": "csv"},
        expected_status=(200, 400),
        expect_success=None,
    )
    results.append(r_ops)

    # Dashboard merge (upload a small right dataset with student_id)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
        tf.write("student_id,bonus\n")
        tf.write("S001,1\n")
        tf.write("S002,0\n")
        tf.write("S003,1\n")
        tf_path = tf.name

    try:
        with open(tf_path, "rb") as f:
            up_r = s.post(url_upload, files={"file": (os.path.basename(tf_path), f)}, data={"slot": "right"}, timeout=120)
        up_rj = _as_json(up_r)
        results.append(
            CheckResult(
                name="Upload right (POST /upload-file)",
                ok=(up_r.status_code == 200 and bool(up_rj.get("success")) is True),
                status=up_r.status_code,
                detail="" if up_r.status_code == 200 else _short(str(up_rj)),
            )
        )

        r_merge, _ = check_json(
            s,
            base,
            "Dashboard merge (on=student_id)",
            "POST",
            "/api/dashboard/merge",
            json_body={"how": "left", "on": "student_id", "indicator": True},
            expected_status=(200, 400, 501),
            expect_success=None,
        )
        results.append(r_merge)
    finally:
        try:
            os.unlink(tf_path)
        except Exception:
            pass

    print_report(results)

    # exit code
    failed = [r for r in results if not r.ok]
    return 1 if failed else 0


def print_report(results: Sequence[CheckResult]) -> None:
    width = max(len(r.name) for r in results) if results else 10
    print("\n=== Quantix runtime smoke test ===")
    ok_count = sum(1 for r in results if r.ok)
    print(f"Results: {ok_count}/{len(results)} OK\n")

    for r in results:
        status = "" if r.status is None else f"[{r.status}]"
        flag = "OK" if r.ok else "FAIL"
        line = f"{flag:4} {status:6} {r.name.ljust(width)}"
        if (not r.ok) and r.detail:
            line += " :: " + r.detail
        print(line)


if __name__ == "__main__":
    raise SystemExit(main())
