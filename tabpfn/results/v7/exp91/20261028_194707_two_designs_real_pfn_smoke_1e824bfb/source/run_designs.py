#!/usr/bin/env python3
"""Train/evaluate the two proposals from an explicit NPZ data contract.

Required: X_l, y_l, X_u, X_test. Optional: y_test, ids_l, ids_u, ids_test.
Use --smoke for a tiny synthetic integration check with the real TabPFN backend.
Use scripts/results_versions.py run for registered, reproducible executions.
"""
from __future__ import annotations

from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import repo_root
ROOT = repo_root(__file__)

import argparse
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import json
import shutil
import time
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from common import TabPFNFactory, matrix, row_groups, reject_overlap
from teacher_student import StudentConfig, TeacherStudentClassifier
from pair_residual import PairConfig, PairResidualClassifier


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def demo(seed):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 1.2, size=(3, 6))
    def draw(n):
        y = np.repeat(np.arange(3), n)
        return (centres[y]+rng.normal(0, 1, size=(len(y), 6))).astype('float32'), y
    X, y = draw(6)
    U, _ = draw(12)
    T, yt = draw(8)
    return dict(X_l=X, y_l=y, X_u=U, X_test=T, y_test=yt,
                ids_l=np.arange(len(X)), ids_u=np.arange(len(U))+100,
                ids_test=np.arange(len(T))+1000)


def predict_batches(model, X, batch_size):
    if not len(X): return np.empty((0, len(model.classes_)))
    return np.concatenate([model.predict_proba(X[i:i+batch_size]) for i in range(0, len(X), batch_size)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', choices=['a', 'b', 'both'], default='both')
    parser.add_argument('--config', type=Path, help='JSON with student and/or pair dataclass overrides')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--benign-class', type=int, default=0)
    parser.add_argument('--categorical-columns', type=int, nargs='*', default=[])
    parser.add_argument('--estimators', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1024, help='PFN/query inference batch size')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--allow-test-feature-overlap', action='store_true',
                        help='Only for a declared historical split; always audit overlap')
    args = parser.parse_args()
    if args.smoke == bool(args.input):
        parser.error('Provide exactly one of --input or --smoke')
    if args.batch_size < 1 or args.estimators < 1 or args.threads < 1:
        parser.error('Batch size, estimator count and threads must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        raise FileExistsError('Use a new managed run directory')
    torch.set_num_threads(args.threads)
    started = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    if args.smoke:
        np.savez_compressed(out/'input_smoke.npz', **demo(args.seed))
        args.input = out/'input_smoke.npz'
    data = np.load(args.input, allow_pickle=False)
    missing = set(['X_l', 'y_l', 'X_u', 'X_test'])-set(data.files)
    if missing: raise ValueError(f'Missing NPZ keys: {sorted(missing)}')
    X = matrix(data['X_l'])
    U = matrix(data['X_u'], features=X.shape[1], allow_empty=True)
    T = matrix(data['X_test'], features=X.shape[1])
    y = np.asarray(data['y_l'])
    reject_overlap(X, U)
    overlap = np.intersect1d(row_groups(np.concatenate([X, U])), row_groups(T))
    if len(overlap) and not args.allow_test_feature_overlap:
        raise ValueError('Train/U and test feature groups overlap; fix split or declare historical diagnostic')
    ids = {}
    for key, count in [('ids_l', len(X)), ('ids_u', len(U)), ('ids_test', len(T))]:
        if key in data:
            value = data[key]
            if value.ndim != 1 or len(value) != count or len(np.unique(value)) != count:
                raise ValueError(f'Invalid/duplicate row IDs: {key}')
            ids[key] = value
            np.save(out/f'{key}.npy', value)
    for a, b in [('ids_l','ids_u'),('ids_l','ids_test'),('ids_u','ids_test')]:
        if a in ids and b in ids and np.intersect1d(ids[a], ids[b]).size:
            raise ValueError(f'Overlapping row IDs: {a}/{b}')
    overrides = json.loads(args.config.read_text()) if args.config else {}
    sc = dict(overrides.get('student', {}))
    pc = dict(overrides.get('pair', {}))
    sc.update(seed=args.seed, device=args.device, categorical_columns=tuple(args.categorical_columns),
              predict_batch_size=args.batch_size)
    pc.update(seed=args.seed, batch_size=args.batch_size)
    if args.smoke:
        sc.update(hidden=32, embedding=16, warmup_epochs=1, rounds=1, epochs_per_round=1, batch_size=24)
        pc.update(max_experts=1, folds=2)
    student_cfg, pair_cfg = StudentConfig(**sc), PairConfig(**pc)
    student_cfg.validate(); pair_cfg.validate()
    factory = TabPFNFactory(str(args.checkpoint.resolve()), args.device,
        1 if args.smoke else args.estimators, args.seed, tuple(args.categorical_columns))
    source = out/'source'; source.mkdir(exist_ok=True)
    code_hashes = {}
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path, source/path.name); code_hashes[path.name] = sha(path)
    if args.config: shutil.copy2(args.config, source/'config_override.json')
    protocol = ROOT/'configs/experiments/exp91_protocol.json'
    if protocol.exists(): shutil.copy2(protocol, source/protocol.name)
    metadata = dict(started_at=started, research_version='v7', experiment_id='exp91',
        purpose='synthetic_implementation_smoke' if args.smoke else 'explicit_input_experiment',
        input_path=str(args.input.resolve()), input_sha256=sha(args.input), checkpoint_sha256=sha(args.checkpoint),
        label_rows=len(X), label_counts={str(c):int((y==c).sum()) for c in np.unique(y)},
        unlabelled_rows=len(U), test_rows=len(T), train_test_feature_group_overlap=len(overlap),
        supplied_row_id_keys=list(ids), additional_validation_labels=0,
        test_label_access='Only metrics after both fit and predict; no parameter selection',
        student_config=asdict(student_cfg), pair_config=asdict(pair_cfg), factory=asdict(factory),
        source_sha256=code_hashes, python=sys.version, packages={p:importlib.metadata.version(p)
        for p in ['torch','numpy','scipy','scikit-learn','joblib','tabpfn']})
    write(out/'config.json', metadata)
    outputs = {}
    for name in (['a','b'] if args.model=='both' else [args.model]):
        print(f'Fitting proposal {name}', flush=True)
        model = TeacherStudentClassifier(factory, student_cfg) if name=='a' else PairResidualClassifier(factory, args.benign_class, pair_cfg)
        sync(); start = time.perf_counter()
        if name=='a': model.fit(X, y, U)
        else: model.fit(X, y)
        sync(); fit_time = time.perf_counter()-start
        start = time.perf_counter()
        p = predict_batches(model, T, args.batch_size)
        sync(); inference_time = time.perf_counter()-start
        np.save(out/f'{name}_probabilities.npy', p)
        np.save(out/f'{name}_classes.npy', model.classes_)
        export = out/('student.joblib' if name=='a' else 'pair_model')
        model.save(export)
        loaded = (TeacherStudentClassifier if name=='a' else PairResidualClassifier).load(export, device=args.device)
        restored = loaded.predict_proba(T[:min(8, len(T))])
        np.testing.assert_allclose(restored, p[:len(restored)], atol=2e-5, rtol=2e-5)
        if name=='b':
            model.audit_['final_backbone_hashes'] = model.verify_frozen()
            loaded.close()
        write(out/f'{name}_audit.json', model.audit_)
        if name=='a': write(out/'a_training_history.json', model.history_)
        outputs[name] = dict(fit_seconds=fit_time, inference_seconds=inference_time,
            probabilities_sha256=sha(out/f'{name}_probabilities.npy'), export_reload_verified=True,
            selected_experts=len(model.steps_) if name=='b' else None)
        if 'y_test' in data:
            truth = data['y_test']  # first label access, after model selection and prediction
            if truth.ndim != 1 or len(truth)!=len(T) or not np.isin(truth, model.classes_).all():
                raise ValueError('Evaluation labels not aligned with trained classes')
            predicted = model.classes_[p.argmax(1)]
            attacks = model.classes_[model.classes_ != args.benign_class]
            normal = truth == args.benign_class
            outputs[name]['metrics'] = {'attack_macro_f1': f1_score(truth,predicted,labels=attacks,average='macro',zero_division=0),
                'benign_fpr': float(np.mean(predicted[normal]!=args.benign_class)) if normal.any() else None,
                'confusion_matrix': confusion_matrix(truth,predicted,labels=model.classes_).tolist(),
                'interpretation': 'synthetic plumbing check only' if args.smoke else 'this NPZ test population only'}
        if name=='b': model.close()
        del model, loaded
    data.close()
    write(out/'summary.json', outputs)
    write(out/'COMPLETE.json', dict(status='complete', completed_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        purpose=metadata['purpose'], models=list(outputs), ids_performance_claim=False if args.smoke else None))
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
