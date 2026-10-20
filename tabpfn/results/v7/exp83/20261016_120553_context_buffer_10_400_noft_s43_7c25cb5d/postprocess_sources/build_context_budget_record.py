#!/usr/bin/env python3
"""Export auditable research tables and figures without editing slides or HTML."""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root / 'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from exp61_utils import write, sha
from context_budget_sweep import BUDGETS, read

METHODS = ['xgb', 'pfn_accumulate', 'pfn_fifo', 'pfn_random', 'pfn_similar']
LABELS = {'xgb': 'XGB 누적 재학습', 'pfn_accumulate': 'PFN 누적', 'pfn_fifo': 'PFN FIFO',
          'pfn_random': 'PFN 무작위 삭제', 'pfn_similar': 'PFN 유사도 삭제'}
COLORS = ['#4b5563', '#2563eb', '#d97706', '#059669', '#c026d3']
MARKERS = ['s', 'o', '^', 'D', 'v']
DATASETS = {'cic2018': 'CIC2018', 'toniot': 'ToN'}


def style():
    font = Path('/home/user/.local/share/fonts/Pretendard/Pretendard-Regular.otf')
    if font.exists():
        font_manager.fontManager.addfont(font)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})


def save(fig, dest):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(dest.with_suffix('.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--out', type=Path, default=_root / 'docs/research/20261016')
    a = p.parse_args()
    run, out = a.run.resolve(), a.out
    verification = read(run / 'verification.json')
    assert verification['passed'] and verification['complete']
    result = read(run / 'results.json')
    assert result['state'] == 'complete' and len(result['jobs']) == 60
    rows, class_rows, stage_rows = [], [], []
    jobs = {(j['dataset'], j['new_per_class'], j['method']): j for j in result['jobs']}
    for ds in DATASETS:
        metadata = read(run / 'data' / ds / 'manifest.json')
        names = metadata['class_names']
        y = np.load(Path(metadata['source']) / 'train_y.npy', mmap_mode='r')
        for n in BUDGETS:
            plans = read(run / f'new_{n}' / 'data' / ds / 'plans.json')
            initial = set(y[plans[0]['incoming']])
            benign = names.index('benign')
            initial_attacks = initial - {benign}
            new_attacks = set(range(len(names))) - initial
            cumulative = jobs[ds, n, 'pfn_accumulate']['metrics']['macro_f1'] * 100
            xgb = jobs[ds, n, 'xgb']['metrics']['macro_f1'] * 100
            for method in METHODS:
                j = jobs[ds, n, method]
                counts = np.asarray(j['final_context_counts'])
                metrics = j['metrics']
                f1 = metrics['macro_f1'] * 100
                row = dict(dataset=ds, new_per_class=n, method=method,
                           released_rows=j['label_budget'], context_rows=j['context_rows'],
                           macro_f1_pct=f1, delta_vs_cumulative_pp=f1-cumulative, delta_vs_xgb_pp=f1-xgb,
                           benign_fpr_pct=metrics['benign_false_positive_rate'] * 100,
                           update_seconds=j['update_seconds'],
                           update_selection_seconds=sum(f['selection_seconds'] for f in j['fits'][1:]),
                           update_fit_seconds=sum(f['fit_seconds'] for f in j['fits'][1:]),
                           update_construct_seconds=sum(f['construct_seconds'] for f in j['fits'][1:]),
                           cost_reused=j['cost_reused'], cost_source=j.get('cost_source', str(run / f'new_{n}' / 'jobs' / f'{ds}_{method}' / 'results.json')),
                           context_benign=int(counts[benign]),
                           context_initial_attacks=int(counts[list(initial_attacks)].sum()),
                           context_new_attacks=int(counts[list(new_attacks)].sum()),
                           initial_attack_mean_f1_pct=float(np.mean([metrics['classes'][c]['f1'] for c in initial_attacks])) * 100,
                           new_attack_mean_f1_pct=float(np.mean([metrics['classes'][c]['f1'] for c in new_attacks])) * 100,
                           original_order_prediction_mismatches=j.get('original_order_prediction_mismatches'),
                           test_rows=j['test_rows'], prediction_path=j['predictions_path'])
                rows.append(row)
                for c in sorted(range(len(names)), key=lambda c: -metrics['classes'][c]['support']):
                    item = metrics['classes'][c]
                    class_rows.append(dict(dataset=ds, new_per_class=n, method=method, class_name=names[c],
                                           test_support=item['support'], context_count=int(counts[c]),
                                           precision_pct=item['precision'] * 100, recall_pct=item['recall'] * 100,
                                           f1_pct=item['f1'] * 100, TP=item['TP'], FP=item['FP'], FN=item['FN']))
                for s, fit in enumerate(j['fits']):
                    c = np.bincount(y[fit['context_indices']], minlength=len(names))
                    stage_rows.append(dict(dataset=ds, new_per_class=n, method=method, stage=s,
                                           released_rows=fit['released_rows'], incoming_rows=len(plans[s]['incoming']),
                                           context_rows=fit['context_rows'], removed_rows=len(fit['removed_indices']),
                                           context_benign=int(c[benign]), context_initial_attacks=int(c[list(initial_attacks)].sum()),
                                           context_new_attacks=int(c[list(new_attacks)].sum()),
                                           update_seconds=fit['update_seconds'], selection_seconds=fit['selection_seconds'],
                                           construct_seconds=fit['construct_seconds'], fit_seconds=fit['fit_seconds'],
                                           cost_in_main_total=s > 0))
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'exp83_budget_values.csv', index=False)
    pd.DataFrame(class_rows).to_csv(out / 'exp83_per_class.csv', index=False)
    pd.DataFrame(stage_rows).to_csv(out / 'exp83_stage_values.csv', index=False)
    style()
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    for column, ylabel, suffix, note in [
        ('macro_f1_pct', '최종 전체 test Macro-F1 (%)', 'performance',
         '초기 1,000개 고정 · FT 없음 · 버퍼 1,000개 · seed 43 · 모든 조건 동일한 전체 test'),
        ('update_seconds', '후속 갱신 시간 합계 (초)', 'cost',
         '초기 준비·추론 제외 · 선택+모델 구성+fit · 누적/XGB: 기존 측정, 버퍼: 이번 측정')]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
        x = np.arange(len(BUDGETS))
        for ax, (ds, display) in zip(axes, DATASETS.items()):
            subset = frame[frame.dataset == ds]
            for method, color, marker in zip(METHODS, COLORS, MARKERS):
                series = subset[subset.method == method].set_index('new_per_class').loc[BUDGETS, column]
                ax.plot(x, series, label=LABELS[method], color=color, marker=marker,
                        lw=2, ms=5, ls='--' if method == 'xgb' else '-')
            ax.set_title(display, fontsize=15, fontweight='bold')
            ax.set_xticks(x, BUDGETS)
            ax.set_xlabel('신규 공격 종류별 추가 사례 수 (개)')
            ax.grid(axis='y', alpha=.2)
            ax.set_xlim(-.15, 5.15)
            if column == 'update_seconds':
                ax.set_ylim(0, float(frame[column].max()) * 1.1)
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(.5, 1.02), frameon=False)
        fig.text(.5, .01, note, ha='center', fontsize=10, color='#4b5563')
        fig.tight_layout(rect=(0, .05, 1, .92))
        save(fig, figures / f'exp83_context_{suffix}')
    # Numeric annotations in a companion heatmap make every plotted score explicit.
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    vmin, vmax = frame.macro_f1_pct.min(), frame.macro_f1_pct.max()
    for ax, (ds, display) in zip(axes, DATASETS.items()):
        mat = frame[frame.dataset == ds].pivot(index='method', columns='new_per_class', values='macro_f1_pct').loc[METHODS, BUDGETS]
        im = ax.imshow(mat, cmap='Blues', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(display, fontsize=15, fontweight='bold')
        ax.set_xticks(range(6), BUDGETS)
        ax.set_yticks(range(5), [LABELS[m] for m in METHODS])
        ax.set_xlabel('신규 공격 종류별 사례 수 (개)')
        for r in range(5):
            for c in range(6):
                value = float(mat.iloc[r, c])
                ax.text(c, r, f'{value:.2f}', ha='center', va='center', fontsize=11,
                        color='white' if value > vmin + .6 * (vmax-vmin) else '#111827')
    fig.subplots_adjust(left=.15, right=.88, bottom=.17, top=.90, wspace=.12)
    colorbar_axis = fig.add_axes([.91, .17, .015, .73])
    fig.colorbar(im, cax=colorbar_axis, label='최종 전체 test Macro-F1 (%)')
    save(fig, figures / 'exp83_context_scores')
    for ds, display in DATASETS.items():
        metadata = read(run / 'data' / ds / 'manifest.json')
        names = metadata['class_names']
        ref = jobs[ds, 100, 'pfn_accumulate']
        class_order = sorted(range(len(names)), key=lambda c: -ref['metrics']['classes'][c]['support'])
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        palette = plt.get_cmap('tab20').colors
        for ax, method in zip(axes.flat, METHODS[1:]):
            counts = np.array([jobs[ds, n, method]['final_context_counts'] for n in BUDGETS])
            bottom = np.zeros(len(BUDGETS))
            for i, c in enumerate(class_order):
                ax.bar(range(6), counts[:, c], bottom=bottom, color=palette[i], label=names[c], width=.75)
                bottom += counts[:, c]
            ax.set_title(LABELS[method])
            ax.set_xticks(range(6), BUDGETS)
            ax.set_ylabel('최종 context 사례 수 (개)')
            ax.set_ylim(0, max(1100, bottom.max() * 1.15))
            for i, total in enumerate(bottom):
                ax.text(i, total + bottom.max() * .015, f'{int(total):,}', ha='center', fontsize=9)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=5, loc='lower center', bbox_to_anchor=(.5, .005), frameon=False, fontsize=9)
        fig.suptitle(f'{display} · 정책별 최종 context 구성', fontsize=16)
        fig.text(.5, .085, '가로축: 신규 공격 종류별 사례 수 · 클래스 순서: 전체 test 평가 표본 수 내림차순', ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .12, 1, .95))
        save(fig, figures / f'exp83_context_counts_{ds}')
    lines = ['# 신규 공격 표본 수별 context 누적·버퍼 유지 비교', '',
             '초기 1,000개와 동일한 신규 사례를 사용해 FT 없는 TabPFN v3의 누적·FIFO·무작위·유사도 삭제를 비교했다. '
             '신규 공격 종류별 10·20·50·100·200·400개, 두 데이터셋, seed 43 한 번이다. '
             '버퍼는 항상 1,000개를 유지하며 기존 클래스마다 최소 1개를 보호했다. '
             '성능은 마지막 단계 모델의 전체 test 결과이며 중간 단계 예측을 합산하지 않았다.', '',
             '| 데이터 | 신규/클래스 | XGB (%) | PFN 누적 (%) | FIFO (%) | 무작위 (%) | 유사도 (%) |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for ds, display in DATASETS.items():
        for n in BUDGETS:
            values = [jobs[ds, n, method]['metrics']['macro_f1'] * 100 for method in METHODS]
            lines.append(f'| {display} | {n} | ' + ' | '.join(f'{value:.2f}' for value in values) + ' |')
    lines += ['', '관찰과 후속 설계 판단:', '']
    for ds, display in DATASETS.items():
        cumulative = jobs[ds, 400, 'pfn_accumulate']
        fifo = jobs[ds, 400, 'pfn_fifo']
        random = jobs[ds, 400, 'pfn_random']
        similar = jobs[ds, 400, 'pfn_similar']
        names = read(run / 'data' / ds / 'manifest.json')['class_names']
        benign = names.index('benign')
        lines.append(
            f"{display}의 신규 공격당 400개 조건에서 누적/FIFO/무작위/유사도 Macro-F1은 "
            f"{cumulative['metrics']['macro_f1']*100:.2f}/{fifo['metrics']['macro_f1']*100:.2f}/"
            f"{random['metrics']['macro_f1']*100:.2f}/{similar['metrics']['macro_f1']*100:.2f}%다. "
            f"FIFO는 정상 사례를 {fifo['final_context_counts'][benign]}개만 보유하며, "
            f"정상 오탐률은 누적 {cumulative['metrics']['benign_false_positive_rate']*100:.2f}%에서 "
            f"FIFO {fifo['metrics']['benign_false_positive_rate']*100:.2f}%로 달라졌다. "
            '보존량 감소와 품질 저하가 함께 관찰됐지만, 표본 수·선택된 사례·클래스 비율이 함께 변하므로 '
            '하나의 요인만을 분리한 인과 효과로 해석하지 않는다.')
        lines.append('')
    lines += [
        '작은 추가량에서는 일부 버퍼 정책이 누적과 비슷하거나 더 높은 F1을 보였으나, '
        '특정 정책의 우위가 모든 표본 수와 두 데이터셋에서 유지되지는 않았다. '
        '세 버퍼의 크기는 모두 1,000개이므로 이들 사이의 차이는 버퍼 크기 차이만으로 설명할 수 없다. '
        '선택된 사례와 클래스별 배분을 함께 봐야 한다.', '',
        '후속 개선안은 정상·기존 공격의 보존 할당량과 신규 공격의 context 할당량을 함께 제한하는 것이다. '
        '확보한 신규 사례 전부를 반드시 넣는 규칙을 완화하고, 확보한 풀에서 대표 사례를 선택하면 '
        '새 공격이 많이 들어오는 단계에도 기존 사례를 위한 공간을 남길 수 있다. '
        '클래스별 하한이나 배분량은 validation 또는 교차적합으로 정하고 정상 오탐 제약과 함께 검증해야 한다. '
        '이번 실험에는 이 개선안을 적용하지 않았으며, 성능 개선이 입증된 것으로 주장하지 않는다.', '']
    lines += ['', '버퍼 정책의 누적 대비 변화(%p):', '',
              '| 데이터 | 신규/클래스 | FIFO−누적 | 무작위−누적 | 유사도−누적 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for ds, display in DATASETS.items():
        for n in BUDGETS:
            values = [float(frame[(frame.dataset == ds) & (frame.new_per_class == n) & (frame.method == m)].iloc[0].delta_vs_cumulative_pp)
                      for m in METHODS[2:]]
            lines.append(f'| {display} | {n} | ' + ' | '.join(f'{value:+.2f}' for value in values) + ' |')
    lines += ['', '갱신 비용(초)은 초기 단계 이후의 선택·모델 구성·fit 합계다. 초기 준비와 전체 추론은 제외했다. '
              '누적 PFN·XGB의 비용은 기존 실행에서 가져왔고 버퍼 정책의 비용은 이번 실행에서 측정했다. '
              '작은 시간 차이는 실행 시점과 측정 변동의 영향이 있을 수 있어 정책의 확정적인 비용 우위로 해석하지 않는다.', '',
              '| 데이터 | 신규/클래스 | XGB (초) | PFN 누적 (초) | FIFO (초) | 무작위 (초) | 유사도 (초) |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for ds, display in DATASETS.items():
        for n in BUDGETS:
            values = [jobs[ds, n, method]['update_seconds'] for method in METHODS]
            lines.append(f'| {display} | {n} | ' + ' | '.join(f'{value:.3f}' for value in values) + ' |')
    lines += ['', '표본 공급량을 늘리면 누적 정책은 context가 커지고, 버퍼 정책은 기존 사례를 더 많이 제거한다. '
              '두 경우 모두 context의 클래스 비율이 달라지므로 표본 수의 효과와 클래스 구성의 효과를 분리해서 주장할 수 없다. '
              'FIFO는 확보 순서의 대리 기준이며 데이터가 실제로 낡았는지는 측정하지 않았다. '
              '유사도 삭제는 신규 학습 사례와 가까운 기존 사례를 제거하는 정의이며, 중복 제거의 품질 개선을 보장하지 않는다. '
              '개선 방향은 기존 정상·공격의 충분한 보존량과 신규 사례 배분을 함께 제어하는 정책이다. '
              '검증용 데이터 또는 교차적합으로 배분을 정하는 후속 평가가 필요하며, 이번 test 결과로 정책을 튜닝하지 않았다.', '',
              '모든 예측의 혼동행렬과 Macro-F1, 초기 사례 일치, 중첩 표본, 클래스 보존, 삭제 이력, 가중치 불변, 비용 합계를 독립 검증했다. '
              '단일 seed의 탐색 결과이며 표본 선택의 변동성이나 다른 네트워크 환경의 일반화는 검증하지 않았다.', '',
              '누적 PFN의 10·20·50·200·400개 조건도 저장 모델로 전체 혼합 query 평가를 수행했다. '
              '이전 원본 순서 평가와 예측이 달라진 행 수는 CSV의 `original_order_prediction_mismatches`에 기록했다. '
              '이전 결과는 보존하며 이번 비교는 동일한 혼합 평가 순서의 값을 사용한다.', '',
              '[성능 추이](figures/exp83_context_performance.png) · [수치 그래프](figures/exp83_context_scores.png) · '
              '[갱신 비용](figures/exp83_context_cost.png) · [CIC2018 context 구성](figures/exp83_context_counts_cic2018.png) · '
              '[ToN context 구성](figures/exp83_context_counts_toniot.png)', '',
              '[전체 수치 CSV](exp83_budget_values.csv) · [클래스별 CSV](exp83_per_class.csv) · [단계별 CSV](exp83_stage_values.csv) · '
              '[설계](exp83_protocol.md)', '',
              f'실행: `{run.relative_to(_root)}`. 원본 결과와 검증 기록은 해당 실행의 `results.json`, `verification.json`에 있다. '
              '슬라이드·HTML·원고는 수정하지 않았다.', '']
    (out / 'exp83_results.md').write_text('\n'.join(lines))
    paths = [out / 'exp83_budget_values.csv', out / 'exp83_per_class.csv', out / 'exp83_stage_values.csv']
    paths += sorted(figures.glob('exp83_context_*'))
    write(out / 'exp83_artifact_validation.json', dict(passed=True, source_results=str(run / 'results.json'),
          source_results_sha256=sha(run / 'results.json'), verification=str(run / 'verification.json'),
          jobs=60, source_metrics_exact=True, class_sort='descending fixed-test support within dataset/budget/method',
          files={str(p): sha(p) for p in paths}))
    print(json.dumps(dict(complete=True, jobs=len(rows), class_rows=len(class_rows), stage_rows=len(stage_rows))))


if __name__ == '__main__':
    main()
