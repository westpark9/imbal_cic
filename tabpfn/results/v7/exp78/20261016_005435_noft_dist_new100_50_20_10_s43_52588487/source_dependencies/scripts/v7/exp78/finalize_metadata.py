#!/usr/bin/env python3
"""Finalize subset-specific provenance without changing predictions or scores."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap,repo_root
bootstrap(_root)
ROOT=repo_root(__file__)
import argparse,json,shutil
from exp61_utils import write,sha


def finalize(run):
    data=json.loads((run/'results.json').read_text());assert data['state']=='complete'
    cfg=json.loads((run/'config.json').read_text());source=Path(cfg['source'])
    updates=[]
    for ds in cfg['datasets']:
        for n in [50,20,10]:
            folder=run/f'new_{n}'/'data'/ds;path=folder/'manifest.json'
            meta=json.loads(path.read_text());jobs=[j for j in data['jobs'] if j['dataset']==ds and j['new_per_class']==n]
            overlaps={j['test_rows']-j['feature_disjoint_diagnostic']['rows'] for j in jobs};assert len(overlaps)==1
            if 'inherited_metadata' not in meta:
                meta['inherited_metadata']={k:meta[k] for k in ['plans_sha256','source_plans_sha256','test_feature_overlap_with_released_rows']}
            meta.update(plans_sha256=sha(folder/'plans.json'),source_plans_sha256=sha(source/'data'/ds/'plans.json'),
                test_feature_overlap_with_released_rows=overlaps.pop(),metadata_finalizer='scripts/v7/exp78/finalize_metadata.py')
            write(path,meta);updates.append(str(path.relative_to(run)))
    write(run/'metadata_finalization.json',dict(updated=updates,reason='Replace inherited budget-100 plan hashes and feature-overlap counts with values for each nested subset; all predictions and scores preserved'))
    for sourcefile in (ROOT/'scripts/v7/exp78').glob('*.py'):
        target=run/'source_dependencies'/sourcefile.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==sha(sourcefile)
        else:shutil.copy2(sourcefile,target)
    hashes={str(p.relative_to(run/'source_dependencies')):sha(p) for p in (run/'source_dependencies').rglob('*.py')}
    write(run/'source_dependencies.json',hashes)
    print('Finalized metadata for',len(updates),'nested subsets')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);finalize(p.parse_args().run)
