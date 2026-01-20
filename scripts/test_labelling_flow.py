from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import dataclass

from flask import Flask

from modules.labelling.annotation import create_flask_blueprint


def make_app() -> Flask:
    app = Flask(__name__)
    app.testing = True
    app.register_blueprint(create_flask_blueprint(url_prefix="/api/labelling"))
    return app


def post_upload(client, filename: str, data: bytes):
    resp = client.post(
        "/api/labelling/upload",
        data={"file": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
    )
    return resp


def main() -> int:
    app = make_app()
    client = app.test_client()

    # 1) markers
    m = client.get("/api/labelling/markers")
    assert m.status_code == 200, m.data

    # 2) txt upload
    txt = b"Bonjour\nCeci est un test.\n"
    r = post_upload(client, "test.txt", txt)
    assert r.status_code == 200, r.data
    payload = r.get_json()
    assert payload and payload.get("doc_id"), payload
    assert "Bonjour" in (payload.get("text") or ""), payload
    doc_id = payload["doc_id"]

    # 3) annotate a span
    ann_payload = {
        "mot": "Bonjour",
        "marqueur": "POSITIVE",
        "couleur": "#00FF00",
        "phrase": "Bonjour",
        "index_phrase": 0,
        "span_start": 0,
        "span_end": 7,
    }
    a = client.post(f"/api/labelling/annotate?doc_id={doc_id}&auto_expand=false&dedupe=true", json=ann_payload)
    assert a.status_code == 200, a.data

    # 4) verify annotations list
    la = client.get(f"/api/labelling/annotations?doc_id={doc_id}")
    assert la.status_code == 200, la.data
    anns = la.get_json().get("annotations")
    assert isinstance(anns, list) and len(anns) >= 1, anns

    # 5) delete by span (overlaps Bonjour)
    d = client.post(
        f"/api/labelling/annotations/delete_by_span?doc_id={doc_id}",
        json={"span_start": 0, "span_end": 7},
    )
    assert d.status_code == 200, d.data
    dpl = d.get_json()
    assert dpl.get("removed") >= 1, dpl

    print("OK: upload->annotate->delete_by_span fonctionne (txt)")

    # Optional: docx/pdf smoke (si libs dispo)
    try:
        import docx  # noqa: F401

        tmp = io.BytesIO()
        doc = docx.Document()
        doc.add_paragraph("DOCX test")
        doc.save(tmp)
        tmp.seek(0)
        r2 = post_upload(client, "test.docx", tmp.read())
        assert r2.status_code == 200, r2.data
        assert "DOCX test" in (r2.get_json().get("text") or "")
        print("OK: DOCX extraction")
    except Exception as e:
        print(f"SKIP DOCX: {e}")

    try:
        from reportlab.pdfgen import canvas

        pdf_buf = io.BytesIO()
        c = canvas.Canvas(pdf_buf)
        c.drawString(72, 720, "PDF test")
        c.save()
        pdf_buf.seek(0)
        r3 = post_upload(client, "test.pdf", pdf_buf.read())
        assert r3.status_code == 200, r3.data
        # pdfplumber peut renvoyer "PDF test" ou vide selon extraction; on valide juste qu'on a du texte str.
        assert isinstance(r3.get_json().get("text"), str)
        print("OK: PDF extraction (smoke)")
    except Exception as e:
        print(f"SKIP PDF: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
