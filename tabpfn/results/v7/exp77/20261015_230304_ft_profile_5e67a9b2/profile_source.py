#!/usr/bin/env python3
"""Initial IDS weight adaptation, context updates, and final-state evaluation."""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import bootstrap, repo_root
bootstrap(_root)
ROOT = repo_root(__file__)
import argparse
import gc
import hashlib
import json
import time
import numpy as np
from exp61_utils import write, sha
from replay_models import timed, make_base, local_request
from replay_protocol import confusion_metrics, dist_adjust_current_batch
from scenario_data import assign_new_roles
from run_context_policies import replace_context

PREVIOUS = ROOT/'tabpfn/results/v7/exp76/20261015_212937_context1000_new100_s43_1c62d14c'
CHECKPOINT = ROOT/'tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt'
METHODS = ['xgb','pfn_pretrained','pfn_ft_accumulate','pfn_ft_fifo','pfn_ft_similar','pfn_ft_random','pfn_ft_fixed','pfn_ft_repeated','ours','boostpfn']


def weights_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_data(out, ds, seed):
    old = PREVIOUS/'data'/ds
    manifest = json.loads((old/'manifest.json').read_text())
    base = Path(manifest['source'])
    arr = {k: np.load(base/f'{k}.npy', mmap_mode='r') for k in ['train_X','train_y','train_hash','train_ids','test_X','test_y','test_hash','test_ids']}
    plans = json.loads((old/'plans.json').read_text())
    # Persist feature-group-disjoint roles as new labeled rows arrive.
    known_ft = {}; known_ours = {}; ft = {'train': [], 'val': []}
    ours = {r: [] for r in ['global','expert','route','tune','cal']}
    for s, plan in enumerate(plans):
        ids = np.asarray(plan['incoming'], dtype='int64')
        for target, known, fractions in [(ft, known_ft, (.9,.1)), (ours, known_ours, (.4,.2,.2,.08,.12))]:
            split = assign_new_roles(ids, arr['train_y'], arr['train_hash'], known, seed+s*104729, tuple(target), fractions)
            for role, values in split.items(): target[role].extend(values.tolist())
        plan['ft_roles'] = {r: list(v) for r,v in ft.items()}
        plan['ours_roles'] = {r: list(v) for r,v in ours.items()}
        for roles in (plan['ft_roles'], plan['ours_roles']):
            assert sorted(sum(roles.values(), [])) == sorted(plan['released'])
            for i,r in enumerate(roles):
                assert set(arr['train_y'][roles[r]]) == set(arr['train_y'][plan['released']]), (ds,s,r)
                for q in list(roles)[i+1:]:
                    assert not np.intersect1d(arr['train_hash'][roles[r]],arr['train_hash'][roles[q]]).size
    assert not np.intersect1d(arr['train_ids'][plans[-1]['released']], arr['test_ids']).size
    dest = out/'data'/ds; dest.mkdir(parents=True, exist_ok=True)
    write(dest/'plans.json', plans)
    write(dest/'manifest.json', dict(manifest, evaluation='single final model on complete fixed test in original file order',
        source_plans_sha256=sha(old/'plans.json'), new_plans_sha256=sha(dest/'plans.json'),
        validation_within_released_budget=True, test_labels_used_for_training=False))
    return arr, manifest['class_names'], plans


def finetune(out, cfg, arr, plan, source):
    import torch
    from tabpfn.finetuning import FinetunedTabPFNClassifier
    from tabpfn.model_loading import save_tabpfn_model
    out.mkdir(parents=True, exist_ok=True)
    if (out/'audit.json').exists(): return out/'weights.ckpt', json.loads((out/'audit.json').read_text())
    tr = np.asarray(plan['ft_roles']['train']); va = np.asarray(plan['ft_roles']['val'])
    classes = np.unique(arr['train_y'][plan['released']])
    labels = lambda ids: np.searchsorted(classes, arr['train_y'][ids])
    ft = FinetunedTabPFNClassifier(device='cuda:0', epochs=cfg['epochs'], learning_rate=1e-5,
        weight_decay=.01, n_finetune_ctx_plus_query_samples=1024, finetune_ctx_query_split_ratio=.2,
        n_estimators_finetune=1, n_estimators_validation=1, n_estimators_final_inference=4,
        early_stopping=True, early_stopping_patience=3, eval_metric='log_loss', random_state=cfg['seed'],
        use_lr_scheduler=False, save_checkpoint_interval=None,
        extra_classifier_kwargs={'model_path': str(source), 'n_preprocessing_jobs': 1, 'auto_scale_n_estimators': False})
    before = torch.load(source, map_location='cpu', weights_only=False)['state_dict']
    torch.cuda.reset_peak_memory_stats()
    _, elapsed = timed(lambda: ft.fit(np.asarray(arr['train_X'][tr]), labels(tr),
        X_val=np.asarray(arr['train_X'][va]), y_val=labels(va), output_dir=out/'training'))
    save_tabpfn_model(ft.finetuned_estimator_, out/'weights.ckpt')
    after = torch.load(out/'weights.ckpt', map_location='cpu', weights_only=False)['state_dict']
    changed = [k for k in after if k in before and not torch.equal(after[k], before[k])]
    audit = dict(seconds=elapsed, source=str(source), checkpoint=str(out/'weights.ckpt'),
        before_sha256=weights_digest(before), after_sha256=weights_digest(after), changed_tensors=len(changed),
        total_tensors=len(after), train_indices=tr.tolist(), validation_indices=va.tolist(),
        train_rows=len(tr), validation_rows=len(va), classes=classes.tolist(),
        epochs_max=cfg['epochs'], learning_rate=1e-5, optimizer='AdamW', loss='cross_entropy',
        best_epoch_selection='validation log_loss; patience 3; no test access',
        peak_torch_gib=torch.cuda.max_memory_allocated()/2**30)
    write(out/'audit.json', audit)
    del ft, before, after; gc.collect(); torch.cuda.empty_cache()
    return out/'weights.ckpt', audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--profile',action='store_true');p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--datasets',nargs='+',default=['cic2018','toniot']);p.add_argument('--methods',nargs='+',default=METHODS)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cfg=dict(seed=43,threads=8,batch=8192,prepare_batch=4096,experts=4,epochs=a.epochs,
        checkpoint=str(CHECKPOINT),source=str(PREVIOUS),protocol='exp77.1',methods=a.methods,datasets=a.datasets,
        evaluation='final_full_test',initial_finetuning=True,html_publication=False)
    write(a.out/'config.json',cfg)
    import torch
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    for ds in a.datasets:
        arr,names,plans=load_data(a.out,ds,cfg['seed'])
        checkpoint,audit=finetune(a.out/'finetuning'/ds/'initial',cfg,arr,plans[0],CHECKPOINT)
        if a.profile:
            ids=np.asarray(plans[-1]['released']); classes=np.unique(arr['train_y'][ids])
            model=make_base('tabpfn',dict(cfg,checkpoint=str(checkpoint)),len(classes))
            _,fit_seconds=timed(lambda:model.fit(arr['train_X'][ids],np.searchsorted(classes,arr['train_y'][ids])))
            _,cold=timed(lambda:model.predict_proba(arr['test_X'][:cfg['batch']]))
            _,warm=timed(lambda:model.predict_proba(arr['test_X'][cfg['batch']:5*cfg['batch']]))
            write(a.out/f'{ds}_profile.json',dict(finetuning=audit,fit_seconds=fit_seconds,first_batch_seconds=cold,
                warm_rows=4*cfg['batch'],warm_seconds=warm,estimated_full_inference_seconds=warm/(4*cfg['batch'])*len(arr['test_y'])))
            del model;gc.collect();torch.cuda.empty_cache()
        else:
            run_dataset(a.out,cfg,ds,arr,names,plans,checkpoint,audit)


if __name__=='__main__':main()
