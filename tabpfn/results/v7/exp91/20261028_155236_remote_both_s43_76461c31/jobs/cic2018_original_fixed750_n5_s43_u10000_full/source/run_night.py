#!/usr/bin/env python3
"""Run prepared EXP91 conditions sequentially, with portable paths and metadata.

Register this runner with scripts/results_versions.py run. Use --check before
launching. No test population, benign label, or categorical schema is guessed.
"""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents
             if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root/'scripts/common'))
from experiment_paths import repo_root
ROOT = repo_root(__file__)

import argparse
import json
import shlex
import shutil
import subprocess
import traceback
from prepare_inputs import INPUTS, CHECKPOINT, DATASETS, SOURCE_HASH, filename, sha, stamp, write


def report_run(out, config, jobs, summaries, errors):
    """Append factual Korean run results without replacing prior same-day records."""
    from lablog import write_record, build_report
    day = stamp()[:10].replace('-', '')
    destination = ROOT/'docs/lablog/evidence'/day/'exp91_remote.txt'
    target = ROOT/'docs/lablog/reports'/f'{day}_exp91.tex'
    existing = target.read_text() if target.is_file() else ''
    tag = f'% EXP91_RUN: {out.name}'
    if tag in existing:
        return
    rows = ['| 조건 | 모델 | 공격 Macro-F1 | 정상 오탐률 | 학습 초 | 추론 초 |',
            '|---|---|---:|---:|---:|---:|']
    for name, models in summaries.items():
        for model, result in models.items():
            metrics = result.get('metrics', {})
            f1 = metrics.get('attack_macro_f1')
            fpr = metrics.get('benign_fpr')
            rows.append(f'| {name} | {model} | {f1:.6f} | {fpr:.6f} | '
                        f'{result["fit_seconds"]:.3f} | {result["inference_seconds"]:.3f} |')
    text = '\n\n'.join([
        '# EXP91: 원격 장비의 두 설계 실행 기록',
        f'## 실행 {out.name}',
        f'두 제안 분류기의 실제 자료 실행을 기록한다. 완료 조건은 {len(summaries)}/{len(jobs)}개이며 '
        f'실행 모델은 {config["model"]}, seed는 {config["seeds"]}, 공격별 라벨 예산은 {config["budgets"]}이다. '
        f'정상 라벨은 750개, 비라벨 학습 행은 {config["unlabelled"]}개다. 실제 실행 경로는 {out.relative_to(ROOT)}이다.',
        '원래 시나리오별 시간 분할과 정제 전 특징을 사용한다. 각 모델은 정상 포함 전체 원래 test를 평가한다. '
        '기존에 검토한 분할이며 train/비라벨과 test 사이의 동일 특징 중복을 유지하고 그 수를 감사 파일에 기록한다. '
        '따라서 이 결과를 독립 분할 검증이나 문헌 대비 우위로 해석하지 않는다.',
        '\n'.join(rows),
        f'실패 상태는 {errors}이며 각 조건의 stdout.log에서 원인을 확인한다. '
        '실패 조건은 동일한 설정으로 재개하고, 설정이나 소스를 수정할 때에는 새 실행으로 등록한다. '
        '후속 분석은 클래스별 오류와 전문가 채택 여부, 정상 오탐, 같은 정보 예산의 기반 모델 및 절제 대조를 확인한다.',
        f'출처: {out.relative_to(ROOT)}/summary.json 및 각 jobs/ 하위 config.json·input_manifest.json·감사 파일.'
    ])
    result = write_record(destination, text, summary=f'원격 실행 {out.name}: {len(summaries)}/{len(jobs)} 조건 완료; '
                          '정제 전·정상 750·전체 원래 test, 성능 해석 및 비교 대조 검토 필요', compile_pdf=False)
    generated = result.read_text()
    if existing:
        body = generated.split('\\begin{document}', 1)[1].rsplit('\\end{document}', 1)[0]
        body = '\n'.join(body.splitlines()[2:])  # omit second report title
        result.write_text(existing.rsplit('\\end{document}', 1)[0]+'\n'+tag+'\n'+body+'\n\\end{document}\n')
    else:
        result.write_text(generated.replace('\\end{document}', tag+'\n\\end{document}'))
    if shutil.which('xelatex'):
        build_report(result)
    else:
        write(out/'report_status.json', dict(report=str(result.relative_to(ROOT)), compile='pending: XeLaTeX unavailable'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=INPUTS)
    p.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--budgets', type=int, nargs='+', default=[2, 3, 5, 10, 20])
    p.add_argument('--seeds', type=int, nargs='+', default=[43])
    p.add_argument('--unlabelled', type=int, default=10000)
    p.add_argument('--model', choices=['a', 'b', 'both'], default='both')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--estimators', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--config', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    if not args.check and args.output is None:
        p.error('--output is required unless using --check')
    if not all(1 <= n <= 20 for n in args.budgets) or args.unlabelled < 0:
        p.error('Budgets must be 1..20 and unlabelled count must be nonnegative')
    jobs, missing = [], []
    if not args.checkpoint.is_file():
        missing.append(str(args.checkpoint))
    if args.config and not args.config.is_file():
        missing.append(str(args.config))
    for n in args.budgets:
        for seed in args.seeds:
            for dataset in args.datasets:
                source = args.input_dir/filename(dataset, n, seed, args.unlabelled)
                meta = source.with_suffix('.json')
                missing.extend(str(f) for f in [source, meta] if not f.is_file())
                if source.is_file() and meta.is_file():
                    md = json.loads(meta.read_text())
                    if (md['source_sha256'] != SOURCE_HASH or md['dataset'] != dataset
                            or md['n'] != n or md['seed'] != seed or md['unlabelled_rows'] != args.unlabelled
                            or md['bytes'] != source.stat().st_size):
                        raise ValueError(f'Input manifest disagrees with request: {meta}')
                    jobs.append((source, md))
    prepare_command = [sys.executable, 'scripts/v7/exp91/prepare_inputs.py',
        '--output', str(args.input_dir), '--datasets', *args.datasets,
        '--budgets', *map(str, args.budgets), '--seeds', *map(str, args.seeds),
        '--unlabelled', str(args.unlabelled)]
    print(json.dumps(dict(at=stamp(), ready=not missing, missing=missing, prepared_conditions=len(jobs),
        model=args.model, input_directory=str(args.input_dir), checkpoint=str(args.checkpoint),
        prepare_command=shlex.join(prepare_command)), ensure_ascii=False, indent=2), flush=True)
    if missing:
        raise SystemExit(2)
    if args.check:
        return
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/'jobs').mkdir(exist_ok=True)
    source_dir = out/'source'
    config = dict(model=args.model, datasets=args.datasets, budgets=args.budgets, seeds=args.seeds,
        unlabelled=args.unlabelled, checkpoint_sha256=sha(args.checkpoint), device=args.device,
        estimators=args.estimators, batch_size=args.batch_size, threads=args.threads,
        override_sha256=sha(args.config) if args.config else None,
        input_sha256=[m['npz_sha256'] for _, m in jobs],
        source_sha256={s.name: sha(s) for s in Path(__file__).parent.glob('*.py')})
    if (out/'night_config.json').exists():
        if json.loads((out/'night_config.json').read_text()) != config:
            raise ValueError('Resume configuration or source changed; use a new managed run')
    else:
        write(out/'night_config.json', config)
    source_dir.mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob('*.py'):
        if not (source_dir/source.name).exists():
            shutil.copy2(source, source_dir/source.name)
    summaries, errors = {}, []
    try:
        for i, (input_path, md) in enumerate(jobs):
            name = input_path.stem
            target = out/'jobs'/name
            target.mkdir(exist_ok=True)
            write(out/'progress.json', dict(at=stamp(), phase='verify_input', condition=name,
                  completed=len(summaries), total=len(jobs)))
            if sha(input_path) != md['npz_sha256']:
                raise ValueError(f'Input checksum changed: {input_path}')
            if (target/'COMPLETE.json').is_file():
                summaries[name] = json.loads((target/'summary.json').read_text())
                continue
            command = [sys.executable, '-u', str(Path(__file__).with_name('run_designs.py')),
                '--input', str(input_path.resolve()), '--checkpoint', str(args.checkpoint.resolve()),
                '--output', str(target), '--model', args.model, '--device', args.device,
                '--seed', str(md['seed']), '--benign-class', str(md['benign_class']),
                '--estimators', str(args.estimators), '--batch-size', str(args.batch_size),
                '--threads', str(args.threads), '--categorical-columns',
                *map(str, md['categorical_columns'])]
            if md['historical_test_feature_overlap_allowed']:
                command.append('--allow-test-feature-overlap')
            if args.config:
                command.extend(['--config', str(args.config.resolve())])
            shutil.copy2(input_path.with_suffix('.json'), target/'input_manifest.json')
            write(target/'command.json', command)
            write(out/'progress.json', dict(at=stamp(), phase='running', condition=name,
                  completed=len(summaries), total=len(jobs)))
            print(f'[{stamp()}] {i+1}/{len(jobs)} {name} models={args.model}', flush=True)
            with (target/'stdout.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                errors.append(dict(condition=name, returncode=result.returncode))
                write(out/'failures.json', errors)
                print(f'Failed {name}; see {target / "stdout.log"}', flush=True)
            else:
                summaries[name] = json.loads((target/'summary.json').read_text())
            write(out/'summary.json', summaries)
    finally:
        write(out/'progress.json', dict(at=stamp(), phase='finished' if len(summaries)==len(jobs) else 'partial',
              completed=len(summaries), total=len(jobs), failed=errors))
        write(out/'summary.json', summaries)
        try:
            report_run(out, config, jobs, summaries, errors)
        except Exception:
            (out/'report_error.txt').write_text(traceback.format_exc())
            print(f'Report generation needs review: {out / "report_error.txt"}', flush=True)
    if errors:
        raise SystemExit(1)
    write(out/'COMPLETE.json', dict(at=stamp(), conditions=len(summaries), models=args.model,
          evidence_adoption='unreviewed', evaluation='full original test; historical split'))
    print(f'[{stamp()}] Complete: {out}', flush=True)


if __name__ == '__main__':
    main()
