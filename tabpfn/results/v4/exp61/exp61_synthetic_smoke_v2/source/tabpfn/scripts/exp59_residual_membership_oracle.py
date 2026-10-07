#!/usr/bin/env python3
"""Residual-membership oracle on consistent single-seed expert banks.

Reuse EXP57 only after exact identity checks; otherwise compute a fresh bank
with exp59_fresh_worker.py. Test truth enters residual membership and scoring only;
it never selects an expert by its correctness or repairs an expert mistake.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import types

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'tabpfn/results/20260922_143529_exp57_expert_quality_s42_44'
ARMS = ['designed', 'matched_random', 'balanced_random']


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                              default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def confusion(y, pred, C):
    return np.bincount(np.asarray(y, dtype=np.int64) * C + pred,
                       minlength=C*C).reshape(C, C)


def scores(cm):
    support = cm.sum(1)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = support - tp
    den = support + cm.sum(0)
    f1 = np.divide(2*tp, den, out=np.zeros(len(tp)), where=den > 0)
    precision = np.divide(tp, tp+fp, out=np.zeros(len(tp)), where=tp+fp > 0)
    recall = np.divide(tp, support, out=np.zeros(len(tp)), where=support > 0)
    return dict(macro_f1=float(f1.mean()), accuracy=float(tp.sum()/max(1, cm.sum())),
                present_macro_f1=float(f1[support > 0].mean()) if np.any(support) else None,
                support=support, TP=tp, FP=fp, FN=fn, f1=f1, precision=precision, recall=recall)


def best_region_constants(counts):
    """Exact joint Macro-F1 optimum over C**K constant-per-region mappings.

    This is a null diagnostic with the same label-assisted membership as the
    residual oracle. It is deliberately stronger than a train-majority rule.
    """
    counts = np.asarray(counts, dtype=np.int64)
    K, C = counts.shape
    if C**K > 10_000_000:
        raise ValueError('Constant enumeration exceeds the declared small-K design')
    support = counts.sum(0)
    rows = counts.sum(1)
    best, mapping = -1., None
    for choice in itertools.product(range(C), repeat=K):
        tp = np.zeros(C, dtype=np.int64)
        predicted = np.zeros(C, dtype=np.int64)
        for k, c in enumerate(choice):
            tp[c] += counts[k, c]
            predicted[c] += rows[k]
        den = support + predicted
        value = float(np.divide(2*tp, den, out=np.zeros(C), where=den > 0).mean())
        if value > best + 1e-15:
            best, mapping = value, np.asarray(choice, dtype=np.int16)
    return mapping, best


def assigned_prediction(experts, regions):
    experts = np.asarray(experts)
    regions = np.asarray(regions)
    if experts.ndim != 2 or len(regions) != len(experts):
        raise ValueError('Need N x K predictions and N fixed region IDs')
    if np.any(regions < 0) or np.any(regions >= experts.shape[1]):
        raise ValueError('Invalid region ID')
    return experts[np.arange(len(regions)), regions]


class Replay:
    def __init__(self, source, out):
        self.source, self.out = source, out
        self.started = time.time()

    def progress(self, phase, **extra):
        obj = dict(phase=phase, elapsed_seconds=time.time()-self.started, **extra)
        write(self.out/'progress.json', obj)
        print('EXP59 ' + json.dumps(obj), flush=True)

    def run_bank_diagnostics(self, v):
        self.progress('verify_frozen_inputs')
        audit = {}
        with np.load(self.source/'shared_context_ids.npz') as z:
            for name, key in [('global_context', 'g_idx'), ('anchor', 'anchor_idx'), ('expert_pool', 'exp_idx')]:
                np.testing.assert_array_equal(z[name], v[key])
                audit[name + '_ids_exact'] = True
        with np.load(self.source/'evaluation_identity.npz') as z:
            np.testing.assert_array_equal(z['ids'], v['eval_idx'])
            np.testing.assert_array_equal(z['y'], v['y_eval'])
            audit['test_ids_and_labels_exact'] = True
        old_tune = np.load(self.source/'probabilities/global_tune.npy', mmap_mode='r')
        audit['global_tune_max_abs_difference'] = float(np.max(np.abs(old_tune-v['p0_tune_raw'])))
        np.testing.assert_allclose(old_tune, v['p0_tune_raw'], rtol=0, atol=1e-6)
        self.progress('reconstruct_full_residual_clusters', global_tune_difference=audit['global_tune_max_abs_difference'])
        bank = v['build_bank'](4)
        mu = bank['mu']
        old = np.load(self.source/'fixed_regions.npz')
        audit['observable_centroid_max_abs_difference'] = float(np.max(np.abs(mu[:, :v['sig'].obs_dim]-old['centroids'])))
        np.testing.assert_allclose(mu[:, :v['sig'].obs_dim], old['centroids'], rtol=0, atol=1e-5)
        for k, expert in enumerate(bank['experts']):
            existing = np.load(self.source/f'contexts/designed_e{k+1}.npy')
            np.testing.assert_array_equal(existing, v['exp_idx'][expert['block_rows']])
            audit[f'expert{k+1}_context_ids_exact'] = True
        state = {'full_centroids': mu, 'w_bal': v['w_bal'], 'train_regions': bank['assign'],
                 'train_labels': v['y_exp'], 'train_ids': v['exp_idx'],
                 'train_residual_clip': np.asarray(v['r_max']), 'global_temperature': np.asarray(v['temp'])}
        for name, (mean, std) in v['sig'].stats.items():
            state[name+'_mean'] = mean
            state[name+'_std'] = std
        np.savez(self.out/'residual_state.npz', **state)
        write(self.out/'reconstruction_audit.json', audit)
        self.progress('reconstruct_test_representation', context_ids_exact=True)
        v['embed_stage']['tag'] = 'exp59/test_residual'
        z = v['phi'].transform(v['X_eval'])
        np.save(self.out/'test_z.npy', z)
        g_raw = np.load(self.source/'probabilities/global_test.npy', mmap_mode='r')
        p0 = v['corr0'].correct(g_raw, 0., v['temp'])
        # The same observable projection must reproduce the old evaluation
        # exactly before we attach the old expert predictions to new regions.
        observable = v['sig'].observable(z, p0)
        regions_obs = v['sq_dist_to_centroids'](observable, mu[:, :v['sig'].obs_dim]).argmin(1)
        audit['observable_test_assignment_mismatches'] = int(np.count_nonzero(regions_obs != old['test']))
        np.testing.assert_array_equal(regions_obs, old['test'])
        r = v['balanced_ce'](p0, v['y_eval'], v['w_bal'])
        signature = v['sig'].full(z, p0, v['y_eval'], np.minimum(r, v['r_max']))
        distance = v['sq_dist_to_centroids'](signature, mu)
        regions = distance.argmin(1).astype(np.int16)
        np.save(self.out/'residual_test_regions.npy', regions)
        np.save(self.out/'residual_test_distances.npy', distance)
        audit.update(full_residual_vs_observable_changed_rows=int(np.count_nonzero(regions != regions_obs)),
                     test_rows=len(regions), train_only_scaling=True,
                     residual_clip=float(v['r_max']), global_temperature=float(v['temp']),
                     reconstruction_seconds=time.time()-self.started)
        write(self.out/'reconstruction_audit.json', audit)
        v['manifest_df'].to_csv(self.out/'split_manifest.csv', index=False)
        v['pool_audit'].to_csv(self.out/'train_pool_partition.csv', index=False)
        write(self.out/'RECONSTRUCTION_COMPLETE.json', audit)
        self.progress('reconstruction_complete')


def reconstruct(source, out):
    import faulthandler
    faulthandler.enable(all_threads=True)
    frozen = SOURCE/'source/tabpfn/scripts'
    sys.path.insert(0, str(frozen))
    sys.path.insert(0, str(SOURCE/'source/scripts'))
    from nfv3_conflict_clean import install_clean_loader
    import nfv3_v3_common as core
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(16)
    torch.set_num_interop_threads(4)
    job = json.loads((source/'job.json').read_text())
    args = job['args'].copy()
    args.update(out_root=str(out/'replay'), models_dir=str(out/'replay/models'),
                resume_dir=str(out/'replay/resume'))
    install_clean_loader(core, args['clean_manifest'])
    code = (source/'executed_prefix.py').read_text()
    target = '            clf = make_clf(np.concatenate([anchor[0], X_exp[top]]), yk)'
    if code.count(target) != 1:
        raise ValueError('Frozen prefix no longer has the expected expert fit site')
    code = code.replace(target, '            clf = None  # EXP59: reuse frozen expert predictions')
    # Functions local to the frozen run are exposed to its diagnostic hook.
    target_return = '    return _exp57.run_bank_diagnostics(locals())'
    if code.count(target_return) != 1:
        raise ValueError('Expected EXP57 diagnostic hook')
    code = code.replace(target_return,
                        '    return _exp57.run_bank_diagnostics(dict(locals(), sq_dist_to_centroids=sq_dist_to_centroids, balanced_ce=balanced_ce))')
    (out/'executed_replay.py').write_text(code)
    module = types.ModuleType('exp59_frozen_base')
    module.__file__ = str(frozen/'nfv3_v3_exp31_c0alloc.py')
    module._exp57 = Replay(source, out)
    sys.modules[module.__name__] = module
    exec(compile(code, str(out/'executed_replay.py'), 'exec'), module.__dict__)
    with threadpool_limits(16):
        module.run_exp29(types.SimpleNamespace(**args))


def analyze(source, out, arms=None):
    arms = ARMS if arms is None else list(arms)
    if not arms or any(arm not in ARMS for arm in arms):
        raise ValueError('Expected at least one declared, completed context arm')
    started = time.time()
    design = json.loads((source/'design.json').read_text())
    names, C, K = design['class_names'], design['C'], design['K']
    with np.load(source/'evaluation_identity.npz') as z:
        y = z['y'].astype(np.int64)
    regions = np.load(out/'residual_test_regions.npy')
    counts = np.bincount(regions.astype(np.int64)*C+y, minlength=K*C).reshape(K, C)
    mapping, optimum = best_region_constants(counts)
    constant = mapping[regions]
    np.testing.assert_allclose(scores(confusion(y, constant, C))['macro_f1'], optimum, atol=1e-14)
    np.save(out/'best_region_constant_pred.npy', constant)
    write(out/'constant_reference.json', dict(mapping=mapping.tolist(), labels=[names[c] for c in mapping],
              macro_f1=optimum, exact_mappings_evaluated=C**K,
              rule='joint exact test Macro-F1 maximum over one fixed class per frozen residual region'))
    train = np.load(out/'residual_state.npz')
    train_counts = np.bincount(train['train_regions'].astype(np.int64)*C+train['train_labels'], minlength=K*C).reshape(K, C)
    compositions = []
    for k in range(K):
        for split, hist in [('train_expert_pool', train_counts[k]), ('test', counts[k])]:
            p = hist/max(1, hist.sum())
            compositions.append(dict(split=split, region=k+1, rows=int(hist.sum()), classes_present=int((hist>0).sum()),
                 dominant_class=names[hist.argmax()], dominant_fraction=float(p.max()),
                 entropy=float(-(p[p>0]*np.log(p[p>0])).sum()), **{n:int(hist[c]) for c,n in enumerate(names)}))
    pd.DataFrame(compositions).to_csv(out/'residual_region_composition.csv', index=False)
    g = np.load(source/'predictions/global_raw.npy')
    np.testing.assert_array_equal(g,np.load(source/'probabilities/global_test.npy',mmap_mode='r').argmax(1))
    global_score = scores(confusion(y, g, C))
    summary, perclass, region_rows, region_pc, changes, cm_rows = [], [], [], [], [], []

    def append(model, pred, arm='', oracle=False):
        cm = confusion(y, pred, C)
        s = scores(cm)
        row = dict(model=model, arm=arm, macro_f1=s['macro_f1'], accuracy=s['accuracy'], rows=len(y))
        if oracle:
            row.update(delta_global=s['macro_f1']-global_score['macro_f1'],
                       delta_constant=s['macro_f1']-optimum,
                       delta_oracle=s['macro_f1']-max(global_score['macro_f1'], optimum))
        summary.append(row)
        for c, n in enumerate(names):
            perclass.append(dict(model=model, arm=arm, **{'class':n}, **{m:s[m][c] for m in ['support','TP','FP','FN','precision','recall','f1']}))
            for d, nn in enumerate(names):
                cm_rows.append(dict(model=model, actual=n, predicted=nn, rows=int(cm[c,d])))
        return s

    append('global', g)
    append('best_region_constant', constant)
    reused_files = [source/f for f in ['design.json','evaluation_identity.npz','shared_context_ids.npz','summary.csv',
                                      'per_class.csv','probability_metrics.csv','context_compositions.csv',
                                      'predictions/global_raw.npy','probabilities/global_test.npy']]
    original_summary = pd.read_csv(source/'summary.csv').set_index('model')
    for arm in arms:
        preds = []
        for k in range(K):
            file = source/f'predictions/{arm}_e{k+1}_raw.npy'
            pred = np.load(file)
            p = np.load(source/f'probabilities/{arm}_e{k+1}_test.npy', mmap_mode='r')
            np.testing.assert_array_equal(pred, p.argmax(1))
            measured = append(f'{arm}_e{k+1}', pred, arm)
            np.testing.assert_allclose(measured['macro_f1'], original_summary.loc[f'{arm}_e{k+1}_raw','macro_f1'], atol=1e-12)
            preds.append(pred)
            reused_files.extend([file, source/f'probabilities/{arm}_e{k+1}_test.npy', source/f'contexts/{arm}_e{k+1}.npy'])
        E = np.stack(preds, axis=1)
        oracle = assigned_prediction(E, regions)
        np.save(out/f'{arm}_residual_oracle_pred.npy', oracle)
        append(f'{arm}_residual_oracle', oracle, arm, True)
        for k in range(K):
            mask = regions == k
            local_y = y[mask]
            if not len(local_y):
                continue
            models = [('global', g), *[(f'e{j+1}', preds[j]) for j in range(K)],
                      ('joint_best_constant', constant), ('local_majority_constant', np.full(len(y), counts[k].argmax(), dtype=np.int16))]
            for model, pred in models:
                s = scores(confusion(local_y, pred[mask], C))
                region_rows.append(dict(arm=arm, region=k+1, model=model, rows=int(mask.sum()),
                    classes_present=int((counts[k]>0).sum()), designated=(model==f'e{k+1}'),
                    macro_f1_all_classes=s['macro_f1'], present_macro_f1=s['present_macro_f1'], accuracy=s['accuracy']))
                for c,n in enumerate(names):
                    region_pc.append(dict(arm=arm,region=k+1,model=model,**{'class':n},
                        **{m:s[m][c] for m in ['support','TP','FP','FN','precision','recall','f1']}))
        for c,n in enumerate(names):
            mask = y == c
            changes.append(dict(arm=arm, **{'class':n}, rows=int(mask.sum()),
                global_correct=int(((g==y)&mask).sum()), oracle_correct=int(((oracle==y)&mask).sum()),
                fixed=int(((g!=y)&(oracle==y)&mask).sum()), harmed=int(((g==y)&(oracle!=y)&mask).sum())))
    for file, rows in [('summary',summary),('per_class',perclass),('region_metrics',region_rows),
                       ('region_per_class',region_pc),('oracle_changes',changes),('confusion',cm_rows)]:
        pd.DataFrame(rows).to_csv(out/(file+'.csv'), index=False)
    for f in ['probability_metrics.csv','context_compositions.csv','fixed_operating_points.csv']:
        if (source/f).exists():shutil.copy2(source/f, out/f)
    write(out/'source_manifest.json', dict(source=str(source), dataset=design['dataset'], seed=design['seed'],K=K,
        cache_origin='fresh EXP59 bank' if source.name=='fresh_bank' else 'completed EXP57 bank',
        reused={str(p.relative_to(source)):sha(p) for p in reused_files},
        new_evaluation='full frozen residual membership, unchanged expert predictions, exact constant null',
        no_scorer_verifier_refit=True, test_role=design['test_role']))
    write(out/'COMPLETE.json',dict(dataset=design['dataset'],seed=design['seed'],K=K,
        arms=arms,all_contexts_complete=arms==ARMS,rows=len(y),analysis_seconds=time.time()-started,
        global_macro_f1=global_score['macro_f1'],constant_macro_f1=optimum,
        oracle=[r for r in summary if 'delta_oracle' in r]))


def worker(root, dataset):
    source = SOURCE/f'{dataset}_s43'
    out = root/f'{dataset}_s43'
    out.mkdir(parents=True, exist_ok=True)
    try:
        if not (out/'RECONSTRUCTION_COMPLETE.json').exists():
            reconstruct(source, out)
        analyze(source, out)
    except BaseException:
        write(out/'ERROR.json',dict(traceback=traceback.format_exc()))
        raise


def controller(root):
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__,root/'experiment_source.py')
    completed=[]
    started=time.time()
    for dataset in ['cic2018','toniot']:
        job=f'{dataset}_s43';out=root/job;out.mkdir(exist_ok=True)
        if (out/'COMPLETE.json').exists():
            completed.append(job);continue
        env={**os.environ,'OMP_NUM_THREADS':'16','MKL_NUM_THREADS':'16','OPENBLAS_NUM_THREADS':'16',
             'NUMEXPR_NUM_THREADS':'16','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONUNBUFFERED':'1',
             'PYTHONFAULTHANDLER':'1','CUDA_VISIBLE_DEVICES':'0'}
        with (out/'worker.log').open('a') as log:
            if (root/'cache_replay_decision.json').exists():
                command=[sys.executable,'-u',str(Path(__file__).with_name('exp59_fresh_worker.py')),'--root',str(root),'--dataset',dataset]
            else:
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root),'--stage','worker','--dataset',dataset]
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            write(root/'status.json',dict(state='running',active_job=job,worker_pid=child.pid,
                  controller_pid=os.getpid(),completed=completed,started_epoch=started))
            code=child.wait()
        if code:
            write(root/'status.json',dict(state='needs_recovery',job=job,exit_code=code,completed=completed))
            raise SystemExit(code)
        completed.append(job)
    write(root/'status.json',dict(state='complete',completed=completed,seconds=time.time()-started))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['controller','worker','analyze'],required=True)
    parser.add_argument('--dataset',choices=['cic2018','toniot'])
    args=parser.parse_args()
    if args.stage=='controller':controller(args.root.resolve())
    elif args.stage=='worker':worker(args.root.resolve(),args.dataset)
    else:
        out=args.root.resolve()/f'{args.dataset}_s43'
        source=out/'fresh_bank' if (out/'fresh_bank/COMPLETE.json').exists() else SOURCE/f'{args.dataset}_s43'
        analyze(source,out)
