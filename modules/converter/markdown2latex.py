from __future__ import annotations

from pathlib import Path
import re
from typing import List, Optional


# ==========================================================
# LaTeX utils
# ==========================================================

def latex_escape(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("\\", r"\textbackslash{}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("$", r"\$")
            .replace("#", r"\#")
            .replace("_", r"\_")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("~", r"\textasciitilde{}")
            .replace("^", r"\textasciicircum{}")
    )


def render_inline(text: str) -> str:
    """Markdown inline → LaTeX (ordre correct)"""
    if not text:
        return ""

    text = latex_escape(text)

    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"\*(.+?)\*", r"\\textit{\1}", text)
    text = re.sub(r"`(.+?)`", r"\\texttt{\1}", text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"\\href{\2}{\1}", text)

    return text


# ==========================================================
# Block model
# ==========================================================

class Block:
    pass


class Heading(Block):
    def __init__(self, level: int, text: str):
        self.level = level
        self.text = text


class Paragraph(Block):
    def __init__(self, text: str):
        self.text = text


class CodeBlock(Block):
    def __init__(self, code: str, language: str):
        self.code = code
        self.language = language or "text"


class ListBlock(Block):
    def __init__(self, items: List[str], ordered: bool):
        self.items = items
        self.ordered = ordered


class Image(Block):
    def __init__(self, alt: str, path: str):
        self.alt = alt
        self.path = path


class Table(Block):
    def __init__(self, rows: List[List[str]]):
        self.rows = rows


class Math(Block):
    def __init__(self, content: str, display: bool):
        self.content = content
        self.display = display


# ==========================================================
# Markdown parser
# ==========================================================

def parse_markdown(text: str) -> List[Block]:
    lines = text.splitlines()
    blocks: List[Block] = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Headings
        if m := re.match(r"(#{1,6})\s+(.*)", line):
            blocks.append(Heading(len(m.group(1)), m.group(2)))
            i += 1
            continue

        # Code block
        if line.startswith("```"):
            lang = line[3:].strip()
            buf = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            blocks.append(CodeBlock("\n".join(buf), lang))
            i += 1
            continue

        # Math display
        if line.strip() == "$$":
            buf = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                buf.append(lines[i])
                i += 1
            blocks.append(Math("\n".join(buf), display=True))
            i += 1
            continue

        # Image
        if m := re.match(r"!\[(.*?)\]\((.*?)\)", line):
            blocks.append(Image(m.group(1), m.group(2)))
            i += 1
            continue

        # List
        if re.match(r"[-*]\s+", line):
            items = []
            while i < len(lines) and re.match(r"[-*]\s+", lines[i]):
                items.append(lines[i][2:])
                i += 1
            blocks.append(ListBlock(items, ordered=False))
            continue

        if re.match(r"\d+\.\s+", line):
            items = []
            while i < len(lines) and re.match(r"\d+\.\s+", lines[i]):
                items.append(lines[i].split(". ", 1)[1])
                i += 1
            blocks.append(ListBlock(items, ordered=True))
            continue

        # Table
        if line.startswith("|") and "|" in line[1:]:
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                row = [c.strip() for c in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", c or "") for c in row):
                    rows.append(row)
                i += 1
            if len(rows) >= 2:
                blocks.append(Table(rows))
            continue

        # Paragraph
        if line.strip():
            buf = [line]
            i += 1
            while i < len(lines) and lines[i].strip():
                buf.append(lines[i])
                i += 1
            blocks.append(Paragraph(" ".join(buf)))
            continue

        i += 1

    return blocks


# ==========================================================
# Renderer
# ==========================================================

def render_latex(blocks: List[Block]) -> str:
    out: List[str] = []

    for b in blocks:
        if isinstance(b, Heading):
            cmd = ["section", "subsection", "subsubsection", "paragraph", "subparagraph"][b.level - 1]
            out.append(f"\\{cmd}{{{latex_escape(b.text)}}}\n")

        elif isinstance(b, Paragraph):
            out.append(render_inline(b.text) + "\n\n")

        elif isinstance(b, CodeBlock):
            out.append(
                "\\begin{lstlisting}[language=%s]\n%s\n\\end{lstlisting}\n\n"
                % (b.language.capitalize(), b.code)
            )

        elif isinstance(b, ListBlock):
            env = "enumerate" if b.ordered else "itemize"
            out.append(f"\\begin{{{env}}}\n")
            for it in b.items:
                out.append(f"\\item {render_inline(it)}\n")
            out.append(f"\\end{{{env}}}\n\n")

        elif isinstance(b, Image):
            out.append(
                "\\begin{figure}[h]\n"
                "\\centering\n"
                f"\\includegraphics[width=0.8\\linewidth]{{{b.path}}}\n"
                f"\\caption{{{latex_escape(b.alt)}}}\n"
                "\\end{figure}\n\n"
            )

        elif isinstance(b, Table):
            cols = len(b.rows[0])
            out.append("\\begin{tabular}{" + "c" * cols + "}\n\\hline\n")
            for r in b.rows:
                out.append(" & ".join(latex_escape(c) for c in r) + " \\\\\n")
            out.append("\\hline\n\\end{tabular}\n\n")

        elif isinstance(b, Math):
            if b.display:
                out.append("\\[\n" + b.content + "\n\\]\n\n")
            else:
                out.append(f"${b.content}$")

    return "".join(out)


# ==========================================================
# Public API
# ==========================================================

def markdown_to_latex(md_path: str | Path) -> str:
    md_path = Path(md_path)
    text = md_path.read_text(encoding="utf-8")

    blocks = parse_markdown(text)
    body = render_latex(blocks)

    return body


def convert(md_path: str | Path, tex_path: str | Path, **kwargs) -> Path:
    """Wrapper compatibilité: Markdown -> LaTeX (.tex)."""

    tex_path = Path(tex_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    body = markdown_to_latex(md_path)
    tex_path.write_text(body, encoding="utf-8")
    return tex_path


__all__ = ["markdown_to_latex", "convert"]
