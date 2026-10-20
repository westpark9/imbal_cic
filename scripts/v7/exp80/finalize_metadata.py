#!/usr/bin/env python3
"""Replace inherited budget-100 metadata with expanded-budget provenance."""
from pathlib import Path
import sys,json,argparse,hashlib
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)
from exp61_utils import write,sha

p=argparse.ArgumentParser();p.add_argument('run',type=Path);run=p.parse_args().run
cfg=json.loads((run/'config.json').read_text());result=json.loads((run/'results.json').read_text());assert result['state']=='complete'
for ds in cfg['datasets']:
    for n in cfg['new_budgets']:
        dest=run/f'new_{n}'/'data'/ds;meta=json.loads((dest/'manifest.json').read_text())
        selected=[j for j in result['jobs'] if j['dataset']==ds and j['new_per_class']==n]
        overlap={j['test_rows']-j['feature_disjoint_diagnostic']['rows'] for j in selected};assert len(overlap)==1
        meta.setdefault('inherited_metadata',{k:meta[k] for k in ['new_plans_sha256','test_feature_overlap_with_released_rows'] if k in meta})
        meta.update(plans_sha256=sha(dest/'plans.json'),new_plans_sha256=sha(dest/'plans.json'),
            test_feature_overlap_with_released_rows=overlap.pop(),metadata_finalized=True)
        write(dest/'manifest.json',meta)
write(run/'metadata_finalization.json',dict(complete=True,predictions_changed=False,reason='Update inherited budget-100 plan hashes and feature-overlap counts'))
print('Expanded-budget metadata finalized')
