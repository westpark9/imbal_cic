#!/usr/bin/env python3
"""Render every CSV artifact in experiment run dir(s) to a color-coded PNG
table next to it (same stem, .png) via exp_utils.render_table_png.

Read-only companion viewer: experiment scripts stay frozen (CLAUDE.md — never
retrofit a script that already has runs in results/), and this restores the
numbered CSV+PNG pair convention retroactively for runs that wrote CSVs only
(exp22's first runs). New experiment scripts can call render_dir() at the end
of their artifact block instead.

    python scripts/render_run_tables.py tabpfn/results/<run_dir> [more dirs...]
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_utils import render_table_png  # noqa: E402

# columns where closer to 1.0 is better (render_table_png: blue >=0.9,
# yellow >=0.7, red below). Everything else stays uncolored.
HIGH_GOOD = ("precision", "recall", "f1", "sign_accuracy", "pearson_r",
             "override_precision", "accept_rate", "frac_positive")
MAX_ROWS = 150  # a table PNG beyond this is unreadable anyway


def render_dir(run_dir):
    made = []
    for name in sorted(os.listdir(run_dir)):
        if not name.endswith(".csv"):
            continue
        path = os.path.join(run_dir, name)
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # unreadable csv -> skip, keep going
            print(f"skip {name}: {exc}")
            continue
        if df.empty:
            print(f"skip {name}: empty")
            continue
        title = name
        if len(df) > MAX_ROWS:
            df = df.head(MAX_ROWS)
            title = f"{name} (first {MAX_ROWS} rows)"
        # render_table_png iterates rows (iterrows coerces mixed numeric rows
        # to float64, printing ints as 1842794.0000). Pre-format integer
        # columns as thousands-separated strings; float columns stay numeric
        # so the fmt and the high_good coloring still apply to them.
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].map(lambda v: f"{v:,}")
        png = path[:-4] + ".png"
        render_table_png(df, png, title=title, high_good=HIGH_GOOD,
                         fmt="{:.4f}")
        made.append(png)
        print(f"wrote {png}")
    return made


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for run_dir in sys.argv[1:]:
        if not os.path.isdir(run_dir):
            raise SystemExit(f"not a directory: {run_dir}")
        print(f"--- {run_dir} ---")
        render_dir(run_dir)


if __name__ == "__main__":
    main()
