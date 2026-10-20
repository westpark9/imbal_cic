#!/usr/bin/env python3
"""EXP75 entry point; shared replay implementation is explicitly in EXP74."""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents
             if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)
from run_adaptation import main

if __name__ == '__main__':
    raise SystemExit(main())
