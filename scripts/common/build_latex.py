#!/usr/bin/env python3
"""Build the selected document in the editor's temporary preview directory."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import re
import subprocess

from experiment_paths import repo_root


def build(source: Path, outdir: Path) -> int:
    source = source.resolve(strict=True)
    outdir = outdir.resolve()
    text = source.read_text()
    lablog = repo_root() / "docs/lablog"
    is_report = source.is_relative_to(lablog / "reports")
    # Reports intentionally resolve style, figures and data from docs/lablog.
    cwd = lablog if is_report else source.parent
    program = re.search(r"^%\s*!\s*TeX\s+program\s*=\s*(\w+)", text, re.I | re.M)
    if program and program[1].lower() in {"pdflatex", "xelatex", "lualatex"}:
        engine = program[1].lower()
    else:
        unicode_fonts = re.search(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\b(?:fontspec|kotex)\b", text)
        engine = "xelatex" if is_report or unicode_fonts else "pdflatex"
    mode = {"pdflatex": "-pdf", "xelatex": "-xelatex", "lualatex": "-lualatex"}[engine]
    outdir.mkdir(parents=True, exist_ok=True)
    command = [
        "latexmk", mode, "-synctex=1", "-interaction=nonstopmode",
        "-halt-on-error", "-file-line-error", f"-outdir={outdir}", str(source),
    ]
    # Two remote editor windows may save the same document concurrently.
    with (outdir / f".{source.stem}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(f"Building {source.name} with {engine}; working directory: {cwd}", flush=True)
        result = subprocess.run(command, cwd=cwd)
        if result.returncode:
            return result.returncode
        log = (outdir / f"{source.stem}.log").read_text(errors="replace")
        missing = [line for line in log.splitlines() if "Missing character:" in line]
        if missing:
            print("Missing glyphs in the preview PDF:\n" + "\n".join(missing[:20]))
            return 1
        pdf = outdir / f"{source.stem}.pdf"
        print(f"Preview PDF: {pdf}", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(build(args.source, args.outdir))
