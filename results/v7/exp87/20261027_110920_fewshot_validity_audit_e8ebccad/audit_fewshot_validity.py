"""Audit EXP86/87 artifacts and run bounded CIC2018 diagnostic controls.

Exploratory diagnostics on the existing test; never a new research version or
an untouched validation set. Original runs are read-only.
"""
from pathlib import Path
import sys

_root = next(p for p in Path(__file__).resolve().parents
             if (p / 'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root / 'scripts/common'))
from experiment_paths import repo_root

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

ROOT = repo_root(__file__)
SEEDS = [43, 44, 45, 46, 47]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def one(pattern):
    paths = list(ROOT.glob(pattern))
    assert len(paths) == 1, (pattern, paths)
    return paths[0]


def metrics(y, p, classes):
    # Independent sklearn implementation, including false positives from all rows.
    cm = confusion_matrix(y, p, labels=classes)
    f = np.divide(2 * np.diag(cm), cm.sum(0) + cm.sum(1),
                  out=np.zeros(len(classes)), where=(cm.sum(0) + cm.sum(1)) != 0)
    return dict(macro_f1=float(f.mean()), per_class_f1=f.tolist(), confusion=cm.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    exp86 = one('tabpfn/results/v7/exp86/*/results.json').parent
    curve = one('tabpfn/results/v7/exp87/*full_learning*/results.json').parent
    pilot = ROOT / read(curve / 'results.json')['source_run']
    cfg = dict(started_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
               source_exp86=str(exp86), source_curve=str(curve), source_pilot=str(pilot),
               seeds=SEEDS, diagnostic_budgets=[2, 20], dataset='cic2018',
               evaluation='Existing full attack-only test; exploratory, no tuning or model selection',
               permission='2026-10-27 user requested review of data difficulty, feature bias, and code',
               benign_train_rows=0, benign_test_rows=0, research_version='v7',
               proxy_group_rationale='Protocol/service and host-stack candidates; association is not proof of spuriousness',
               permutation_repeats_per_seed=10, finetuning=False)
    write(out / 'config.json', cfg)
    write(out / 'environment.json', {k: importlib.metadata.version(k)
                                   for k in ['numpy', 'pandas', 'scikit-learn', 'xgboost']})
    shutil.copy2(__file__, out / Path(__file__).name)

    audit = []
    cached = {}
    for origin in [exp86, curve]:
        for j in read(origin / 'results.json')['jobs']:
            ds = j['dataset']
            if ds not in cached:
                md = read(exp86 / 'data' / ds / 'manifest.json')
                base = Path(md['test_base'])
                cached[ds] = (md, np.load(base / 'test_y.npy', mmap_mode='r'),
                              np.sort(np.load(base / 'test_ids.npy')))
            md, all_y, test_ids = cached[ds]
            names = md['class_names']
            classes = np.array([i for i, name in enumerate(names) if name != 'benign'])
            pp = Path(j['predictions_path'])
            pred = np.load(pp, mmap_mode='r')
            assert sha(pp) == read(pp.parent / 'COMPLETE.json')['predictions_sha256']
            ids = np.load(pp.parent / 'train_source_ids.npy')
            np.testing.assert_array_equal(ids, j['train_source_ids'])
            assert sha(pp.parent / 'train_source_ids.npy') == j['train_ids_sha256']
            assert len(ids) == len(np.unique(ids)) == j['n'] * len(classes) == j['context_rows']
            positions = np.searchsorted(test_ids, ids)
            assert not np.any(test_ids[np.minimum(positions, len(test_ids)-1)] == ids)
            counts = np.asarray(j['train_class_counts'])
            np.testing.assert_array_equal(counts[classes], np.full(len(classes), j['n']))
            assert counts[names.index('benign')] == 0
            if origin == exp86:
                y = all_y[all_y != names.index('benign')]
                p = pred[all_y != names.index('benign')]
                saved = j['phase_a']
                cm = confusion_matrix(all_y, pred, labels=np.arange(len(names)))
                np.testing.assert_array_equal(cm, j['phase_b']['confusion'])
            else:
                y = np.load(pilot / 'data' / ds / 'test_y.npy', mmap_mode='r')
                p = pred
                saved = j['metrics']
            calculated = metrics(y, p, classes)
            np.testing.assert_array_equal(calculated['confusion'], np.asarray(saved['confusion'])[np.ix_(classes, classes)])
            assert abs(calculated['macro_f1'] - saved['macro_f1']) < 1e-12
            audit.append(dict(experiment='exp86' if origin == exp86 else 'exp87', dataset=ds,
                              seed=j['seed'], n=j['n'], method=j['method'], context_rows=len(ids),
                              verified_macro_f1=calculated['macro_f1'], passed=True,
                              predictions_sha256=sha(pp)))
        print('artifact audit complete:', origin.name, len(audit), flush=True)
    pd.DataFrame(audit).to_csv(out / 'artifact_checks.csv', index=False)

    d = pilot / 'data/cic2018'
    md = read(d / 'manifest.json')
    clean = Path(md['clean_manifest']).parent
    clean_md = read(clean / 'manifest.json')
    names = clean_md['feature_names']
    classes = np.asarray(md['attack_classes'])
    tx = np.load(d / 'test_X.npy', mmap_mode='r')
    ty_global = np.load(d / 'test_y.npy')
    ty = np.searchsorted(classes, ty_global)
    test_hash = np.load(d / 'test_hash.npy')
    clean_ids = np.load(clean / 'source_idx.npy', mmap_mode='r')
    full_hash = np.load(clean / 'vector_hash.npy', mmap_mode='r')
    train_ids = np.load(clean / 'train_idx.npy')
    train_hash = full_hash[np.searchsorted(clean_ids, train_ids)]
    _, unique_test = np.unique(test_hash, return_index=True)
    pool_overlap = np.isin(test_hash, np.unique(train_hash))
    context_hashes = []
    for seed in SEEDS:
        sd = d / f'seed_{seed}'
        context_hashes.extend(np.load(sd / 'hashes.npy').tolist())
        previous = None
        for n in [2, 5, 10, 20, 40, 80, 160, 320, 450]:
            ii = np.load(sd / f'n_{n}_indices.npy')
            ids = np.load(sd / 'source_ids.npy')[ii]
            if previous is not None:
                assert np.isin(previous, ids).all()
            previous = ids
            jobs = [j for j in read(curve / 'results.json')['jobs']
                    if j['dataset'] == 'cic2018' and j['seed'] == seed and j['n'] == n]
            assert len(jobs) == 2 and jobs[0]['train_source_ids'] == jobs[1]['train_source_ids'] == ids.tolist()
    fixed_disjoint = ~np.isin(test_hash, np.unique(context_hashes))
    overlap = dict(test_attack_rows=len(ty), test_unique_feature_vectors=len(unique_test),
                   test_rows_matching_any_clean_train_vector=int(pool_overlap.sum()),
                   fixed_disjoint_rows=int(fixed_disjoint.sum()),
                   fixed_disjoint_definition='Exclude union of every five-seed balanced n450 context vector; same mask for all budgets/models',
                   hash_caveat='Grouping uses historical uint64 hashes, not near-duplicate or host/session groups',
                   classes=[])
    for c, label in enumerate(classes):
        mask = ty == c
        overlap['classes'].append(dict(name=md['class_names'][label], test_rows=int(mask.sum()),
             unique_vectors=len(np.unique(test_hash[mask])),
             any_train_overlap_rows=int((mask & pool_overlap).sum()),
             fixed_disjoint_rows=int((mask & fixed_disjoint).sum())))
    write(out / 'overlap.json', overlap)
    np.save(out / 'fixed_disjoint_mask.npy', fixed_disjoint)
    np.save(out / 'unique_test_positions.npy', unique_test)
    secondary = []
    for j in read(curve / 'results.json')['jobs']:
        if j['dataset'] != 'cic2018':
            continue
        pp = np.searchsorted(classes, np.load(j['predictions_path']))
        for label, mask in [('unique_vectors', unique_test), ('fixed_disjoint', fixed_disjoint)]:
            mm = metrics(ty[mask], pp[mask], np.arange(len(classes)))
            secondary.append(dict(seed=j['seed'], n=j['n'], method=j['method'],
                                  evaluation=label, rows=len(ty[mask]), **mm))
    write(out / 'secondary_metrics.json', secondary)

    proxy = ['PROTOCOL', 'L7_PROTO', 'MIN_TTL', 'MAX_TTL', 'TCP_WIN_MAX_IN',
             'TCP_WIN_MAX_OUT', 'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_TYPE',
             'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE']
    groups = dict(all_features=list(range(len(names))),
                  without_proxy=[i for i, x in enumerate(names) if x not in proxy],
                  proxy_only=[names.index(x) for x in proxy],
                  ttl_only=[names.index(x) for x in ['MIN_TTL', 'MAX_TTL']],
                  without_ttl=[i for i, x in enumerate(names) if x not in ['MIN_TTL', 'MAX_TTL']])
    cfg['feature_groups'] = {k: [names[i] for i in v] for k, v in groups.items()}
    write(out / 'config.json', cfg)
    result = []
    replay = []
    for seed in SEEDS:
        sd = d / f'seed_{seed}'
        X = np.load(sd / 'X.npy')
        y = np.searchsorted(classes, np.load(sd / 'y.npy'))
        for n in [2, 20]:
            ix = np.load(sd / f'n_{n}_indices.npy')
            xtr, ytr = X[ix], y[ix]
            original = next(j for j in read(curve / 'results.json')['jobs']
                            if j['dataset'] == 'cic2018' and j['seed'] == seed and j['n'] == n and j['method'] == 'xgb')
            model_path = Path(original['predictions_path']).parent / 'model.ubj'
            restored = XGBClassifier()
            restored.load_model(model_path)
            restored.set_params(device='cpu', n_jobs=8)
            restored_pred = restored.predict(tx)
            original_pred = np.searchsorted(classes, np.load(original['predictions_path']))
            disagreement = int((restored_pred != original_pred).sum())
            assert disagreement == 0
            replay.append(dict(seed=seed, n=n, prediction_disagreements=disagreement))
            variants = [(name, columns, None) for name, columns in groups.items()]
            variants.append(('depth3_tree', groups['all_features'], None))
            if n == 2:
                variants += [(f'label_permutation_{k}', groups['all_features'], k) for k in range(10)]
            for label, cols, perm in variants:
                begin = time.time()
                job = out / 'jobs' / f's{seed}_n{n}_{label}'
                job.mkdir(parents=True)
                yy = ytr if perm is None else np.random.default_rng(271000 + seed*100 + perm).permutation(ytr)
                if label == 'depth3_tree':
                    model = DecisionTreeClassifier(max_depth=3, random_state=seed)
                else:
                    model = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=.05,
                        subsample=.8, colsample_bytree=.8, min_child_weight=1, reg_lambda=1,
                        objective='multi:softprob', num_class=len(classes), eval_metric='mlogloss',
                        tree_method='hist', device='cuda:0', n_jobs=8, random_state=seed)
                model.fit(xtr[:, cols], yy)
                p = model.predict(np.asarray(tx[:, cols]))
                np.save(job / 'predictions.npy', p)
                np.save(job / 'fit_y.npy', yy)
                np.save(job / 'train_source_ids.npy', np.load(sd / 'source_ids.npy')[ix])
                mm = metrics(ty, p, np.arange(len(classes)))
                assert abs(mm['macro_f1'] - f1_score(ty, p, labels=np.arange(len(classes)), average='macro')) < 1e-12
                split_count = None
                if label == 'depth3_tree':
                    (job / 'tree.txt').write_text(export_text(model, feature_names=[names[i] for i in cols]))
                else:
                    model.save_model(job / 'model.ubj')
                    split_count = int((model.get_booster().trees_to_dataframe().Feature != 'Leaf').sum())
                record = dict(seed=seed, n=n, variant=label, feature_count=len(cols),
                              seconds=time.time()-begin, tree_splits=split_count, **mm)
                if label == 'all_features':
                    record['refit_prediction_disagreements'] = int((p != original_pred).sum())
                write(job / 'results.json', record)
                result.append(record)
                write(out / 'diagnostics.json', result)
            print('diagnostics complete', seed, n, len(result), flush=True)
    write(out / 'saved_xgb_replay.json', replay)
    pd.DataFrame([{k:v for k,v in j.items() if k not in ['confusion','per_class_f1']}
                  for j in result]).to_csv(out / 'diagnostics.csv', index=False)
    write(out / 'COMPLETE.json', dict(finished_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
          artifact_checks=len(audit), saved_xgb_replays=len(replay), diagnostic_fits=len(result),
          elapsed_seconds=time.time()-start, paper_adoption='not_adopted_exploratory'))
    print(json.dumps(read(out / 'COMPLETE.json')), flush=True)


if __name__ == '__main__':
    main()
