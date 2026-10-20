# English manuscript for Overleaf

The folder root contains only `main.tex` and `ref.bib` (plus an optional `.gitignore`). Edit these sources. Supporting files live in `_support/`; paper figures live in `figures/`.

Upload the manuscript folder with `_support/acmart.cls` and `_support/ACM-Reference-Format.bst` when using Overleaf. `main.tex` resolves these paths directly. Compiled manuscript and lablog PDFs are not stored in the repository. Figure inputs, reference papers and presentation PDFs are retained.

For Cursor connected to the server, `.vscode/settings.json` compiles the selected document on save. `scripts/common/build_latex.py` runs `latexmk` with SyncTeX inside LaTeX Workshop's `%TMPDIR%`, using a separate subdirectory per source. It does not copy PDFs beside the source. Open the desired `.tex` and use `Ctrl+Alt+B` for the first build, then `Ctrl+Alt+V` for preview. Later saves refresh the viewer. Close an old preview and reopen it from the intended `.tex` after changing output settings. Deleting the temporary cache removes that preview; rebuilding recreates it from the source. `make -C docs/lablog/_support` validates reports in automatically removed temporary directories.

Korean lablog reports explicitly name themselves with `% !TeX root = YYYYMMDD.tex` because the shared style contains their `\documentclass`. This avoids falling back to a previously opened manuscript, including in multi-root workspaces where a folder-level detection setting may not be read. Their `% !LW recipe=lablog XeLaTeX` header selects XeLaTeX and resolves style/figure paths from `docs/lablog/`. Existing reports and the `write_record` generator include both headers; retain them and update the root filename when copying or renaming a report. The workspace files also select `\begin{document}` as the root indicator. The default recipe uses pdfLaTeX for this English manuscript.

`_support/sources.json` records evidence and revision provenance. `references/Haider_ETD.pdf` and `history/from_slides/` are reference materials, not current manuscript inputs. The paper title remains **TAILGUARD: An Uncertainty-Driven Ensemble Against Long-Tailed Cyber Threats**; this layout change does not change the scientific content.

Korean experiment records: [lablog](../../lablog/log.md). Presentations: [slides](../../slides/). Follow the paper rules in [AGENTS.md](../../../AGENTS.md).
