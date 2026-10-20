#!/usr/bin/env python3
"""Build the current compact main slides and appendix from the shared generator."""
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("build_1015_compact.py")), run_name="__main__")
