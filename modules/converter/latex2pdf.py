#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Optional


@dataclass(frozen=True)
class LatexBuildResult:
    pdf_path: Path
    log_path: Path
    workdir: Path


class Latex2PDF:
    """
    Robust LaTeX → PDF builder.

    Features:
    - Isolated build directory
    - Arbitrary number of external figures
    - Automatic \\graphicspath injection
    - latexmk → engine fallback
    """

    def __init__(
        self,
        engine: str = "pdflatex",
        bib_tool: str = "auto",
        runs: int = 0,
        halt_on_error: bool = True,
        synctex: bool = False,
        shell_escape: bool = False,
        keep_workdir: bool = False,
        timeout_s: int = 240,
    ) -> None:
        self.engine = engine.lower()
        self.bib_tool = bib_tool.lower()
        self.runs = runs
        self.halt_on_error = halt_on_error
        self.synctex = synctex
        self.shell_escape = shell_escape
        self.keep_workdir = keep_workdir
        self.timeout_s = timeout_s

    # -------------------------
    # Utils
    # -------------------------

    @staticmethod
    def _which(cmd: str) -> Optional[str]:
        return shutil.which(cmd)

    @staticmethod
    def _read_text(p: Path, limit: int = 2_000_000) -> str:
        try:
            data = p.read_bytes()
            if len(data) > limit:
                data = data[-limit:]
            return data.decode("utf-8", errors="replace")
        except Exception:
            return ""

    @staticmethod
    def _copy_project(src: Path, dst: Path) -> None:
        ignore = shutil.ignore_patterns(
            ".git", "__pycache__", "*.aux", "*.log", "*.out",
            "*.toc", "*.lof", "*.lot", "*.bbl", "*.blg",
            "*.synctex*", "*.fdb_latexmk", "*.fls",
        )
        for item in src.iterdir():
            if item.name.startswith("."):
                continue
            target = dst / item.name
            if item.is_dir():
                shutil.copytree(item, target, ignore=ignore, dirs_exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)

    # -------------------------
    # Figures handling
    # -------------------------

    @staticmethod
    def _inject_graphicspath(tex_path: Path, rel_fig_dir: str) -> None:
        content = tex_path.read_text(encoding="utf-8", errors="replace")

        if "\\graphicspath" in content:
            return  # user already manages it

        injection = f"\n\\graphicspath{{{{{rel_fig_dir}/}}}}\n"

        if "\\documentclass" in content:
            content = content.replace(
                "\\documentclass",
                "\\documentclass" + injection,
                1,
            )
        else:
            content = injection + content

        tex_path.write_text(content, encoding="utf-8")

    # -------------------------
    # Build
    # -------------------------

    def build(
        self,
        main_tex: str | Path,
        *,
        figures: Sequence[str | Path] | None = None,
        out_pdf: str | Path | None = None,
    ) -> LatexBuildResult:

        main_tex = Path(main_tex).resolve()
        if not main_tex.exists() or main_tex.suffix != ".tex":
            raise FileNotFoundError(main_tex)

        src_root = main_tex.parent.resolve()
        out_pdf = Path(out_pdf or main_tex.with_suffix(".pdf")).resolve()
        out_pdf.parent.mkdir(parents=True, exist_ok=True)

        # Detect bibliography tool
        text = self._read_text(main_tex).lower()
        if self.bib_tool == "auto":
            if "biblatex" in text:
                bib_tool = "biber"
            elif "\\bibliography" in text:
                bib_tool = "bibtex"
            else:
                bib_tool = "none"
        else:
            bib_tool = self.bib_tool

        tmp = None
        try:
            if self.keep_workdir:
                workdir = Path(tempfile.mkdtemp(prefix="latex2pdf_"))
            else:
                tmp = tempfile.TemporaryDirectory(prefix="latex2pdf_")
                workdir = Path(tmp.name)

            # Copy LaTeX project
            self._copy_project(src_root, workdir)

            rel_main = main_tex.relative_to(src_root)
            build_main = workdir / rel_main
            cwd = build_main.parent

            # Handle figures
            if figures:
                fig_dir = workdir / "_quantix_figures"
                fig_dir.mkdir(parents=True, exist_ok=True)
                for fig in figures:
                    fig = Path(fig)
                    if fig.exists():
                        shutil.copy2(fig, fig_dir / fig.name)
                self._inject_graphicspath(build_main, "_quantix_figures")

            # latexmk
            if self._which("latexmk"):
                cmd = ["latexmk", "-pdf", "-interaction=nonstopmode"]
                if self.halt_on_error:
                    cmd.append("-halt-on-error")
                if bib_tool == "biber":
                    cmd.append("-usebiber")
                if self.shell_escape:
                    cmd.append("-shell-escape")
                cmd.append(build_main.name)

                subprocess.run(
                    cmd,
                    cwd=cwd,
                    timeout=self.timeout_s,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            else:
                # fallback engines
                for _ in range(max(2, self.runs or 2)):
                    subprocess.run(
                        [self.engine, "-interaction=nonstopmode", build_main.name],
                        cwd=cwd,
                        timeout=self.timeout_s,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

            built_pdf = cwd / build_main.with_suffix(".pdf").name
            if not built_pdf.exists():
                raise RuntimeError("Compilation terminée sans PDF")

            shutil.copy2(built_pdf, out_pdf)
            log_path = cwd / build_main.with_suffix(".log").name

            return LatexBuildResult(
                pdf_path=out_pdf,
                log_path=log_path if log_path.exists() else cwd,
                workdir=workdir,
            )

        finally:
            if tmp is not None:
                tmp.cleanup()


__all__ = ["Latex2PDF", "LatexBuildResult"]
