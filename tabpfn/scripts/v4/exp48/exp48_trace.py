"""EXP48 observations reuse EXP47 hooks with explicit ToN vocabulary."""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

from exp47_trace import Trace as CICTrace, instrument


class Trace(CICTrace):
    def record(self, stage, v):
        super().record(stage, v)
        if stage == 'inputs':
            assert v['args'].target_dataset == 'ton_iot'
            assert 'benign' in self.meta['class_names']
            self.meta['protected_classes'] = ['ddos', 'dos']
            self.meta['dataset'] = 'ton_iot'
            self.meta['experiment'] = 'EXP48'
