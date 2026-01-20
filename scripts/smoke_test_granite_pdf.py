from __future__ import annotations

from pathlib import Path
import tempfile


def make_minimal_pdf_bytes(text: str) -> bytes:
    """Create a tiny 1-page PDF without external dependencies."""

    content = (f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET\n").encode("latin-1", "replace")

    objs: list[tuple[int, bytes]] = []

    def add(n: int, body: bytes) -> None:
        objs.append((n, body))

    add(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    add(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")

    page = b"<< /Type /Page /Parent 2 0 R "
    page += b"/MediaBox [0 0 612 792] "
    page += b"/Resources << /Font << /F1 4 0 R >> >> "
    page += b"/Contents 5 0 R >>"
    add(3, page)

    add(4, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    stream = b"<< /Length " + str(len(content)).encode("ascii") + b" >>\n"
    stream += b"stream\n" + content + b"endstream"
    add(5, stream)

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {0: 0}

    for n, body in objs:
        offsets[n] = len(out)
        out += f"{n} 0 obj\n".encode("ascii")
        out += body + b"\nendobj\n"

    xref_start = len(out)
    out += b"xref\n"
    out += f"0 {len(objs) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for n in range(1, len(objs) + 1):
        out += f"{offsets[n]:010d} 00000 n \n".encode("ascii")

    out += b"trailer\n"
    out += f"<< /Size {len(objs) + 1} /Root 1 0 R >>\n".encode("ascii")
    out += b"startxref\n"
    out += f"{xref_start}\n".encode("ascii")
    out += b"%%EOF\n"

    return bytes(out)


def main() -> int:
    tmp_dir = Path(tempfile.gettempdir())
    pdf_path = tmp_dir / "quantix_granite_test.pdf"
    pdf_path.write_bytes(make_minimal_pdf_bytes("Hello Granite"))

    print("PDF créé:", pdf_path)
    print("Taille:", pdf_path.stat().st_size, "bytes")

    # 1) Test via modules.converter.pdf2markdown
    try:
        from modules.converter.pdf2markdown import pdf_to_markdown_granite

        md_path = pdf_to_markdown_granite(pdf_path)
        print("\n[pdf2markdown] OK:", md_path)
        preview = Path(md_path).read_text(encoding="utf-8")[:400]
        print("Aperçu markdown:\n", preview)
    except Exception as e:
        print("\n[pdf2markdown] FAIL:", type(e).__name__, str(e))

    # 2) Test via modules.converter.graniteIBM
    try:
        from modules.converter.graniteIBM import granite

        out = granite(str(pdf_path), output_format="markdown")
        print("\n[graniteIBM] OK:", out)
        preview = Path(out).read_text(encoding="utf-8")[:400]
        print("Aperçu markdown:\n", preview)
    except Exception as e:
        print("\n[graniteIBM] FAIL:", type(e).__name__, str(e))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
