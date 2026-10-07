#!/usr/bin/env python3
"""Append audited EXP57 results and EXP58 oracle correction to the existing 0918 tab."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from build_exp56_report_section import Markup, table, metric, save_figure

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'tabpfn/results'
LOCAL = RESULTS / '20260922_143529_exp57_expert_quality_s42_44'
ORACLE = RESULTS / 'exp58_six_condition_20260922'
REPORT = ROOT / 'lablog/html_report/scorer_verifier_target_0918.html'
INDEX = ROOT / 'lablog/html_report/post_tabpfn.html'
OUT = ROOT / 'docs/research/20260923'
DATASETS = {'cic2018': 'CIC2018', 'toniot': 'ToN'}
ARMS = ['designed', 'matched_random', 'balanced_random']
ARM_LABELS = ['현재 선택', '클래스 비율 일치 무작위', '같은 크기 균형 무작위']
TABLES = ['summary', 'per_class', 'probability_metrics', 'fixed_operating_points',
          'region_metrics', 'context_compositions', 'paired_bootstrap']
START = '<!-- EXP57_FOLLOWUP_START -->'
END = '<!-- EXP57_FOLLOWUP_END -->'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f(value):
    return f'{value:.4f}' if np.isfinite(value) else '—'


def pct(value):
    return f'{100 * value:.3f}%' if np.isfinite(value) else '—'


def cells(values):
    return [metric(v, max(values)) for v in values]


def load_jobs():
    roots = [LOCAL] + sorted(LOCAL.glob('recovery_*')) + sorted(RESULTS.glob('*exp57_external*'))
    jobs = []
    status = []
    sources = {}
    for ds, title in DATASETS.items():
        for seed in [42, 43, 44]:
            job_id = f'{ds}_s{seed}'
            completed = [r / job_id for r in roots if (r / job_id / 'COMPLETE.json').is_file()]
            if not completed:
                state = '외부 완료 결과 미수신' if seed == 42 else '비정상 종료 · 완성된 세 bank 없음'
                status.append([title, seed, state, '최종 비교에서 제외'])
                continue
            path = completed[0]
            design = json.loads((path / 'design.json').read_text())
            complete = json.loads((path / 'COMPLETE.json').read_text())
            if not (design['dataset'] == ds and design['seed'] == seed and
                    complete['dataset'] == ds and complete['seed'] == seed and
                    design['K'] == 4 and design['K'] < design['C']):
                raise ValueError(f'Unexpected design: {path}')
            job = dict(dataset=ds, title=title, seed=seed, path=path, design=design)
            for name in TABLES:
                job[name] = pd.read_csv(path / f'{name}.csv')
            expected = {'global_raw', 'global_calibrated'}
            for arm in ARMS:
                expected.update(f'{arm}_e{k}_{v}' for k in range(1, 5) for v in ['raw', 'calibrated'])
                expected.update(arm + '_' + v for v in ['fixed_region', 'validation_region_reference', 'region_constant'])
            summary = job['summary'].set_index('model')
            if set(summary.index) != expected or len(summary) != 35:
                raise ValueError(f'Incomplete model table: {path}')
            per = job['per_class']
            base = per[per.model == 'global_raw'].set_index('class').support
            for name in expected:
                q = per[per.model == name].set_index('class').reindex(base.index)
                np.testing.assert_array_equal(q.support, base)
                np.testing.assert_array_equal(q.TP + q.FN, q.support)
                np.testing.assert_allclose(q.f1, 2*q.TP/(2*q.TP+q.FP+q.FN), atol=1e-12)
                np.testing.assert_allclose(q.f1.mean(), summary.loc[name, 'macro_f1'], atol=1e-12)
                np.testing.assert_allclose(q.TP.sum()/q.support.sum(), summary.loc[name, 'accuracy'], atol=1e-12)
            if base.sum() != design['test_rows']:
                raise ValueError(f'Test row count mismatch: {path}')
            comp = job['context_compositions'].set_index(['arm', 'expert'])
            for k in range(1, 5):
                a = comp.loc[('designed', k)]
                b = comp.loc[('matched_random', k)]
                np.testing.assert_array_equal(a[design['class_names']], b[design['class_names']])
                for arm in ARMS:
                    c = comp.loc[(arm, k)]
                    if c.block_rows != a.block_rows or c[design['class_names']].sum() != c.block_rows:
                        raise ValueError(f'Context budget mismatch: {path}, {arm}, e{k}')
            for name in ['probability_metrics', 'fixed_operating_points']:
                expected_rows = 26 * design['C'] * (10 if name == 'fixed_operating_points' else 1)
                if len(job[name]) != expected_rows:
                    raise ValueError(f'Incomplete {name}: {path}')
            job['classes'] = base.sort_values(ascending=False).index.tolist()
            context_equal = {}
            for k in range(1, 5):
                a = path / f'contexts/designed_e{k}.npy'
                b = path / f'contexts/matched_random_e{k}.npy'
                if a.exists() and b.exists():
                    context_equal[f'e{k}'] = bool(np.array_equal(np.sort(np.load(a)), np.sort(np.load(b))))
            jobs.append(job)
            status.append([title, seed, '완료 · 3 bank / 35 평가 조건', f'{complete["seconds"]/60:.1f}분'])
            sources[job_id] = dict(path=str(path.relative_to(ROOT)), context_id_equal_designed_vs_matched=context_equal,
                                  sha256={n:sha(path/n) for n in ['COMPLETE.json', 'design.json'] + [v+'.csv' for v in TABLES]})
    for ds in DATASETS:
        group = [j for j in jobs if j['dataset'] == ds]
        for j in group[1:]:
            a = group[0]['per_class'].query("model == 'global_raw'").set_index('class').support
            b = j['per_class'].query("model == 'global_raw'").set_index('class').support
            np.testing.assert_array_equal(a.sort_index(), b.sort_index())
    if not jobs:
        raise ValueError('No complete EXP57 workers found')
    return jobs, status, sources


def item(job, table_name, model, cl):
    q = job[table_name]
    return q[(q.model == model) & (q['class'] == cl)].iloc[0]


def chosen_from_training(job, cl):
    comp = job['context_compositions'].query("arm == 'designed'").set_index('expert').sort_index()
    ratios = comp[cl] / comp.block_rows
    return int(ratios.idxmax()) if ratios.max() > 0 else None


def focus_rows(jobs):
    rows = []
    for job in jobs:
        targets = ['infiltration', 'web_attacks'] if job['dataset'] == 'cic2018' else ['scanning', 'ransomware', 'xss']
        for cl in job['classes']:
            if cl not in targets:
                continue
            k = chosen_from_training(job, cl)
            if k:
                rows.append((job, cl, k))
    return rows


def setup_fonts():
    path = ROOT / 'lablog/html_report/assets/report_notosans_kr_regular.otf'
    font_manager.fontManager.addfont(str(path))
    font = font_manager.FontProperties(fname=str(path)).get_name()
    plt.rcParams.update({'font.family': font, 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'path',
                         'svg.hashsalt': 'exp57-followup-20260923'})


def figures(jobs, focuses):
    setup_fonts()
    colors = ['#7b8794', '#245e8d', '#d68b38', '#469987']
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5.5, len(focuses)*.65)), layout='constrained')
    labels = [f'{j["title"]} s{j["seed"]} · {cl} / e{k}' for j, cl, k in focuses]
    y = np.arange(len(focuses))
    for ax, field, title in zip(axes, ['AP', 'R_at_FPR_001'], ['AP: 양성·음성 점수 순위', 'R@FPR≤0.1%: 같은 오탐 한도의 recall']):
        for idx, label in enumerate(['Global'] + ARM_LABELS):
            values = [float(item(j, 'probability_metrics', 'global_raw' if idx == 0 else f'{ARMS[idx-1]}_e{k}_raw', cl)[field]) for j, cl, k in focuses]
            ax.barh(y+(idx-1.5)*.18, values, height=.16, label=label, color=colors[idx])
        ax.set_yticks(y, labels if ax == axes[0] else [])
        ax.invert_yaxis(); ax.set_xlim(0, 1.04); ax.set_title(title); ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=4, fontsize=9, frameon=False)
    first = save_figure(fig, 'exp57_context_quality', '학습 context의 해당 클래스 비율로 정한 expert를 같은 seed의 대조군과 비교한다. test 점수로 expert를 고르지 않았다. 각 행은 해당 클래스 전체 test 평가다.')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout='constrained')
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        q = pd.read_csv(ORACLE/ds/'six_condition_summary.csv').set_index('policy')
        labels = ['1 Global', '2 고정 영역', '3 개별 expert', '4 S-only', '5 S+V', '6 Bank oracle']
        experts = q[q.condition == 3].macro_f1
        for i, policy in enumerate(['global', 'fixed_region', None, 'scorer_only', 'scorer_verifier', 'bank_oracle']):
            if policy is None:
                ax.scatter(experts, np.full(len(experts), i), color='#8396a7', s=25)
                ax.hlines(i, experts.min(), experts.max(), color='#8396a7')
                value=experts.max(); label=f'{experts.min():.4f}–{experts.max():.4f}'
            else:
                value=q.loc[policy, 'macro_f1']; label=f'{value:.4f}'
                ax.barh(i, value, color='#245e8d' if i==5 else '#9aaab8', height=.54)
            ax.text(value+.008, i, label, va='center', fontsize=9)
        ax.set_yticks(range(6), labels); ax.invert_yaxis(); ax.set_xlim(0,1.18); ax.set_xticks([0,.25,.5,.75,1]); ax.set_xlabel('Macro-F1'); ax.set_title(title+' · 기존 bank')
    second = save_figure(fig, 'exp58_six_conditions', '기존 다섯 조건과 고정 bank의 정답 참조 상한. 3번은 expert별 단독 적용 점수이며 6번은 행마다 후보 예측을 선택한 값이다. 두 값은 서로 다른 평가를 뜻한다.')
    return first, second


def oracle_section(figure):
    parts = ['<h2 id="oracle-correction">7. 전체 test의 F1과 oracle이 달랐던 이유</h2>',
        '<p>기존 oracle은 담당 expert를 정확히 찾는 평가로 설명됐지만, 실제로는 정답을 확인해 여러 expert 중 맞힌 예측을 고르고, 없으면 global을 유지했다. '
        '반면 개별 expert의 전체 test 적용은 특화 대상 외의 음성까지 모두 분류하므로, 특정 클래스를 넓게 예측할 때 FP가 늘어 F1이 낮아진다. '
        '<b>따라서 높은 oracle과 낮은 개별 F1은 함께 나타날 수 있다.</b> 개별 F1만으로 특화 식별력을 단정하지 않고 AP·동일 FPR recall·영역 내 상수 대조를 함께 확인한다.</p>',
        '<p>6번은 <b>같은 샘플에서 고정된 global·expert가 실제로 출력한 클래스</b>만 후보로 두고, 그 후보 안에서 전체 Macro-F1을 최대화한 참조 상한이다. '
        '맞힐 수 있는 행은 정답을 고르고, 모든 후보가 틀린 행에서도 후보 밖 정답을 만들지 않는다. 남은 오답의 FP 배분까지 최적화하므로 '
        '기존의 “전부 틀리면 global 유지”와 Macro-F1이 다를 수 있다. 클래스별 독립 최댓값을 더한 값이 아니라 하나의 예측 벡터에서 동시에 달성한 값이다.</p>',
        '<p class="note">Test 정답을 사용하는 6번은 배포 성능이나 expert 생성 품질의 주지표가 아니다. 비용·운영 FPR 제약을 적용하지 않은 고정 후보 bank의 상한이다. '
        '기존 다섯 조건의 수치는 유지했다. 여기의 CIC K=8/C=7, ToN K=7/C=10 bank와 아래 EXP57의 새 K=4 bank는 다르다.</p>', figure]
    rows=[];bounds=pd.read_csv(ORACLE/'oracle_summary.csv').query("scope == 'bank'").set_index('dataset')
    for ds,title in DATASETS.items():
        b=bounds.loc[ds]
        rows.append([title,f(b.canonical_global_fallback_f1),f(b.macro_f1_upper_bound),f(b.accuracy_upper_bound),f'{int(b.uncovered_rows):,}'])
    parts.append(table(['기존 bank','이전 global-fallback F1','정정한 Macro-F1 상한','Accuracy 상한','모든 후보가 틀린 행'],rows,'exp58-bounds'))
    for ds,title in DATASETS.items():
        q=pd.read_csv(ORACLE/ds/'six_condition_per_class.csv');base=q[q.policy=='global'].sort_values('support',ascending=False);rows=[]
        for _,r in base.iterrows():
            cl=q[q['class']==r['class']].set_index('policy');e=cl[cl.condition==3].f1
            vals=[cl.loc[p,'f1'] for p in ['global','fixed_region']]+[e.max()]+[cl.loc[p,'f1'] for p in ['scorer_only','scorer_verifier','bank_oracle']]
            best=max(vals);row=[r['class'],f'{int(r.support):,}']
            for i,value in enumerate(vals):
                row.append(Markup(f'{f(e.min())}–{metric(value,best)}') if i==2 else metric(value,best))
            rows.append(row)
        parts += [f'<details><summary>{title} · 여섯 조건의 클래스별 F1</summary>',
                  table(['클래스','Test 수','1 Global','2 고정 영역','3 Expert 범위','4 S-only','5 S+V','6 Bank oracle'],rows,'exp58-classes-'+ds),
                  '<p class="note">6번은 전체 Macro-F1 최적 예측에서의 클래스별 F1이다. 각 클래스만 따로 최적화한 독립 상한은 아니며, 한 클래스의 값이 다른 조건보다 낮을 수도 있다.</p></details>']
    return parts


def result_section(jobs, status, focuses, figure):
    counts={ds:sum(j['dataset']==ds for j in jobs) for ds in DATASETS}
    parts=['<h2 id="exp57-results">8. K=4로 다시 만든 bank와 context 대조</h2>',
           '<p>같은 seed 안에서 global·anchor·잔차 영역·expert별 context 크기를 공유하고, '
           '① 현재 실패 패턴 선택, ② 클래스별 행 수까지 맞춘 무작위 선택, ③ 총 행 수를 맞춘 균형 무작위 선택을 비교했다. '
           '②는 행 선택 방식의 추가 가치를, ③은 클래스 구성 변화의 효과를 살핀다. 두 데이터 모두 <b>K=4&lt;C</b>이며 K는 global을 제외한 expert 수다.</p>',
           f'<p>2026-09-23 현재 검증한 완성본은 <b>CIC2018 {counts["cic2018"]}개 seed, ToN {counts["toniot"]}개 seed</b>다. '
           '아래 집계는 COMPLETE 파일과 35개 평가 조건이 모두 있는 실행만 사용한다. 중단된 bank를 서로 다른 재시도에서 이어 붙이거나 미수신 seed를 평균에 넣지 않았다.</p>',
           table(['데이터','Seed','확인 상태','완료 소요 / 집계'],status,'exp57-status'),
           '<p class="note">각 완성본은 global 1개와 expert 12개를 fit한 결과다. 새 scorer/verifier는 학습하지 않았다. '
           '같은 모순 제거 개발 holdout을 재평가했으며 시간 사분위도 새 독립 test가 아니다. Expert 번호와 centroid는 seed별로 달라질 수 있어 같은 번호를 동일한 expert로 평균내지 않는다.</p>',
           '<h3>Global·고정 영역 적용·상수 대조</h3>']
    rows=[]
    for j in jobs:
        s=j['summary'].set_index('model');values=[s.loc['global_raw','macro_f1']]+[s.loc[a+'_fixed_region','macro_f1'] for a in ARMS]+[s.loc['designed_region_constant','macro_f1']]
        rows.append([j['title'],j['seed']]+cells(values))
    parts += [table(['데이터','Seed','Global','현재 선택 → 고정 영역','비율 일치 무작위 → 고정 영역','균형 무작위 → 고정 영역','영역별 상수'],rows,'exp57-fixed-regions'),
              '<p>이는 새 bank의 기본적인 배정 적합성을 확인한 Macro-F1이다. 기존 다섯 조건의 S/S+V 결과와 같은 실험으로 합치지 않는다. '
              '상수 예측은 tune에서 영역별 최다 클래스를 정해 test에 그대로 적용했다.</p>',
              '<h2 id="exp57-quality">9. 낮은 F1을 식별력 부족으로 단정할 수 있는가?</h2>',
              '<p><b>클래스별로 결론이 다르다.</b> ToN scanning은 두 완료 seed에서 특화 expert의 AP·R@0.1%가 global보다 높다. '
              '동시에 비율 일치 무작위도 비슷하거나 더 높은 값을 보여 현재 행 선택 방식만의 우위와 구분해야 한다. '
              'CIC infiltration은 현재 선택 expert의 AP·R@0.1%가 global보다 낮아, 전체 적용의 FP 증가만으로 설명을 끝낼 수 없다.</p>', figure,
              '<p class="note">그림과 핵심 표의 expert는 학습 특화 block에서 해당 클래스의 비율이 가장 높은 slot으로 정했다(동률이면 작은 번호). '
              'Test 성능을 보고 고른 expert가 아니며, 정답 클래스를 그 expert에 라우팅해 계산한 점수도 아니다. 모든 점수는 양성·음성을 포함한 전체 test에서 측정했다. '
              '클래스는 각 dataset·seed 안에서 test 수 내림차순이다.</p>']
    for field,label in [('AP','AP'),('R_at_FPR_001','R@FPR≤0.1%')]:
        rows=[]
        for j,cl,k in focuses:
            values=[item(j,'probability_metrics','global_raw',cl)[field]]+[item(j,'probability_metrics',f'{a}_e{k}_raw',cl)[field] for a in ARMS]
            support=item(j,'per_class','global_raw',cl).support
            rows.append([j['title'],j['seed'],cl,f'{int(support):,}',f'e{k}']+cells(values))
        parts += [f'<h3>주요 클래스 · {label}</h3>',table(['데이터','Seed','클래스','Test 수','Expert','Global','현재 선택','비율 일치 무작위','균형 무작위'],rows,'exp57-focus-'+field)]
    rows=[]
    for j,cl,k in focuses:
        g=item(j,'per_class','global_raw',cl);e=item(j,'per_class',f'designed_e{k}_raw',cl)
        precision=e.TP/(e.TP+e.FP) if e.TP+e.FP else 0
        rows.append([j['title'],j['seed'],cl,f'e{k}',f(g.f1),f(e.f1),f(precision),f(e.TP/e.support),f'{int(e.FP):,}',f'{int(e.FN):,}'])
    parts += ['<details><summary>같은 expert의 원래 argmax 예측: F1·precision·recall·FP</summary>',
              table(['데이터','Seed','클래스','Expert','G F1','Expert F1','Precision','Recall','FP','FN'],rows,'exp57-raw-fp'),
              '<p>예를 들어 ToN seed 43 scanning e2는 전체 test에서 FP 6,686개로 F1이 0.6042이지만, AP는 0.9145이고 R@0.1%는 0.9003이다. '
              '반면 CIC seed 43 infiltration e3는 FP 865,113개·F1 0.0185이며 AP 0.1422도 global 0.1985보다 낮다. '
              '첫 사례는 분류 임계값·적용 범위와 점수 식별력을 분리해서 볼 근거이고, 둘째는 점수 자체의 개선도 필요한 사례다.</p></details>',
              '<h3>가중 validation에서 정한 FPR 0.1% 운영점의 test 전이</h3>',
              '<p>아래는 test에서 임계값을 다시 고르지 않은 결과다. R@0.1%는 test ROC에서 같은 오탐 한도의 식별력을 비교한 기술적 지표이고, '
              '이 표는 validation의 임계값을 그대로 썼을 때 실제로 그 한도가 유지되는지를 보여준다.</p>',
              '<p class="note">구현상 임계값은 train과 겹치는 행을 제외한 calibration 표본에서 클래스 reference prior로 가중한 음성 분포에 맞췄다. '
              '아래 test FPR은 원래 test 분포의 미가중 FP 비율이다. 두 값의 차이에는 시간 변화뿐 아니라 클래스 비중 차이도 반영될 수 있으므로, '
              '모든 초과를 순수한 시간 drift로 단정하지 않는다.</p>']
    rows=[]
    for j,cl,k in focuses:
        q=j['fixed_operating_points'];q=q[(q['class']==cl)&(q.window=='full_test')&np.isclose(q.validation_FPR_target,.001)].set_index('model')
        g=q.loc['global_raw'];e=q.loc[f'designed_e{k}_raw'];c=q.loc[f'designed_e{k}_calibrated']
        rows.append([j['title'],j['seed'],cl,f'e{k}',f(g.recall),pct(g.FPR),f(e.recall),pct(e.FPR),f'{int(e.FP):,}',f(c.recall),pct(c.FPR)])
    parts += [table(['데이터','Seed','클래스','Expert','G recall','G FPR','Expert recall','Expert FPR','Expert FP','보정 후 recall','보정 후 FPR'],rows,'exp57-transfer'),
              '<p>ToN scanning e2의 raw 임계값은 seed 43·44에서 test FPR 0.248%·0.451%로 0.1%를 초과한다. '
              '따라서 점수 식별력의 이득을 현재 임계값 그대로 사용할 수 있다는 뜻은 아니다. 시간별 calibration과 어려운 음성 표본을 보강한 뒤 '
              '허용 FPR을 지키는지 새 시간 구간에서 확인하는 것이 다음 단계다.</p>']
    return parts


def detail_section(jobs):
    parts=['<h2 id="exp57-details">10. Expert별 근거와 반복 안정성</h2>',
           '<p>아래 원자료 표는 expert 번호 순서를 유지한다. 클래스별 표는 test 샘플 수 내림차순이다. '
           '동일 영역의 모든 expert를 비교하므로 담당 expert보다 다른 expert가 더 적합한지도 확인할 수 있다.</p>']
    for j in jobs:
        prefix=f'{j["dataset"]}-s{j["seed"]}'
        parts += [f'<details><summary>{j["title"]} seed {j["seed"]} · 전체 expert·클래스·시간 구간</summary>']
        rows=[];summary=j['summary'].set_index('model')
        for k in range(1,5):
            vals=[summary.loc['global_raw','macro_f1']]+[summary.loc[f'{a}_e{k}_raw','macro_f1'] for a in ARMS]
            rows.append([f'e{k}']+cells(vals))
        parts += ['<h3>각 expert의 전체 test Macro-F1</h3>',table(['Expert','Global','현재 선택','비율 일치 무작위','균형 무작위'],rows,prefix+'-all-test')]
        rows=[];comp=j['context_compositions'].query("arm=='designed'").set_index('expert')
        for k in range(1,5):
            r=comp.loc[k];rows.append([f'e{k}',f'{int(r.block_rows):,}',f'{int(j["design"]["anchor_rows"]):,}']+[f'{int(r[c]):,}' for c in j['classes']])
        parts += ['<h3>현재 선택의 context 구성</h3>',table(['Expert','특화 block 수','공통 anchor 수']+j['classes'],rows,prefix+'-context'),
                  '<p class="note">표는 특화 block의 클래스 구성이다. 실제 context에는 모든 expert가 공유하는 anchor가 더해진다. '
                  '클래스 비율 일치 무작위의 block 클래스별 행 수가 이 표와 같은지 검증했다.</p>']
        rows=[]
        for cl in j['classes']:
            for k in range(1,5):
                records=[item(j,'probability_metrics','global_raw',cl)]+[item(j,'probability_metrics',f'{a}_e{k}_raw',cl) for a in ARMS]
                rows.append([cl,f'e{k}']+cells([r.AP for r in records])+cells([r.R_at_FPR_001 for r in records]))
        parts += ['<h3>모든 클래스의 식별력 대조 · raw</h3>',table(['클래스','Expert','G AP','현재 AP','비율 일치 AP','균형 AP','G R@0.1%','현재 R','비율 일치 R','균형 R'],rows,prefix+'-all-classes')]
        rows=[]
        region=j['region_metrics'].query("arm=='designed'")
        for k in range(1,5):
            q=region[region.region==k].set_index('model')
            if q.empty:continue
            vals=[q.loc[m,'present_class_macro_f1'] for m in ['global','e1','e2','e3','e4','region_constant']]
            rows.append([f'영역 {k} / e{k}',f'{int(q.iloc[0].rows):,}',int(q.iloc[0].classes_present)]+cells(vals))
        parts += ['<h3>고정된 입력 영역에서 global·모든 expert·상수 비교</h3>',table(['영역 / 담당','Test 수','등장 클래스','Global','e1','e2','e3','e4','영역 상수'],rows,prefix+'-region'),
                  '<p class="note">같은 영역에 실제 등장하는 클래스의 F1 평균이다. 영역 간 또는 전체 test Macro-F1과 직접 비교하지 않는다. '
                  '영역은 정답 없이 입력에서 배정하고, 상수 라벨은 tune에서 고정했다.</p>']
        rows=[];unc=j['paired_bootstrap'].set_index(['model','grouping'])
        for k in range(1,5):
            name=f'designed_e{k}_raw';delta=summary.loc[name,'macro_f1']-summary.loc['global_raw','macro_f1']
            a=unc.loc[(name,'exact_vector')];b=unc.loc[(name,'time_block')]
            rows.append([f'e{k}',f'{delta:+.4f}',f'[{a.delta_macro_f1_low:+.4f}, {a.delta_macro_f1_high:+.4f}]',
                         f'[{b.delta_macro_f1_low:+.4f}, {b.delta_macro_f1_high:+.4f}]'])
        parts += ['<h3>전체 test Macro-F1의 global 대비 차이 · bootstrap 95% 구간</h3>',
                  table(['Expert','ΔMacro-F1','같은 feature 벡터 묶음','시간 블록'],rows,prefix+'-bootstrap'),
                  '<p class="note">200회 paired Poisson cluster bootstrap이다. 무작위 context 대비 차이의 신뢰구간이나 여러 독립 데이터셋에서의 일반화 구간으로 해석하지 않는다.</p>']
        rows=[];q=j['fixed_operating_points']
        for cl in j['classes']:
            for k in range(0,5):
                model='global_raw' if k==0 else f'designed_e{k}_raw'
                a=q[(q['class']==cl)&(q.model==model)&np.isclose(q.validation_FPR_target,.001)].set_index('window')
                vals=[]
                for window in ['full_test']+[f'time_quartile_{v}' for v in range(1,5)]:
                    r=a.loc[window];vals.append(f'{f(r.recall)} / {pct(r.FPR)}')
                rows.append([cl,'G' if k==0 else f'e{k}']+vals)
        parts += ['<h3>가중 validation FPR 0.1% 임계값의 시간 전이 · recall / 실제 FPR</h3>',
                  table(['클래스','모델','전체 test','시간 Q1','시간 Q2','시간 Q3','시간 Q4'],rows,prefix+'-time'),
                  '<p class="note">양성 또는 음성이 없는 구간의 해당 지표는 —로 표시한다. 시간 순서로 나눈 동일 test의 진단이며 미래 독립 평가가 아니다.</p></details>']
    selections=[]
    for j in jobs:
        for p in j['path'].glob('*_selection.json'):
            selections.append(json.loads(p.read_text()))
    beta_zero=sum(r['beta']==0 for r in selections)
    parts += [f'<p>Global과 expert에 같은 tune NLL 기준으로 β·온도를 선택한 {len(selections)}개 fit 중 β=0은 {beta_zero}개다. '
              '현재 완료본에서는 raw와 보정 후 argmax Macro-F1이 같다. β=0이면 양의 온도 변경은 argmax를 바꾸지 않으므로 이 결과는 예상 가능한 동작이다. '
              '확률 순위와 고정 임계값의 결과는 달라질 수 있어 보정 전후 operating-point CSV를 별도로 보존했다.</p>',
              '<p>ToN의 ransomware 특화 block은 seed 43 e3와 seed 44 e4에서 모두 537행이다. 각 실행의 현재 선택과 비율 일치 무작위는 '
              '특화 context ID 집합까지 같아 이 slot에서는 행 선택 알고리즘의 효과를 분리할 수 없다. '
              '그런데 R@0.1%는 두 seed 사이 0.9760→0.0000으로 변했다. Anchor·global·분할 등 함께 달라진 조건을 하나씩 고정하는 반복이 필요하다.</p>']
    return parts


def verdict_section():
    return ['<h2 id="exp57-verdict">11. 현재 확인된 역량과 개선할 대상</h2>',
            table(['대상','이번 결과가 보여주는 것','판정과 다음 조치'],[
                ['ToN scanning','seed 43·44 e2의 AP·저오탐 recall이 global보다 높고, 영역 내 F1도 영역 상수보다 높다. 비율 일치 무작위와는 비슷하며 val 임계값의 test FPR은 목표를 넘는다.',
                 '특화 식별력은 확인됐다. 현재 행 선택의 우월성은 별도이며, 어려운 음성·시간별 calibration으로 운영점 전이를 개선한다.'],
                ['CIC infiltration','seed 43 e3는 전체 적용에서 FP가 많고 AP·R@0.1%도 global보다 낮다. 균형 무작위 e1은 AP 0.2407·R 0.2887로 global 0.1985·0.1521보다 높다.',
                 '양성 중심 context만 강화하는 방식에서 벗어나, 균형 구성·혼동되는 benign 음성을 같은 예산으로 비교한다. 완료 seed를 보충해 재현성을 확인한다.'],
                ['ToN ransomware','특화 block은 고정된 537행인데 seed 43·44의 저오탐 성능이 크게 다르다. 같은 클래스 수 무작위와 context가 같은 slot도 있다.',
                 '현재 상태에서 안정적인 특화 능력으로 판단하지 않는다. Anchor와 global을 고정한 반복으로 변동 원인을 분리한 뒤 음성 구성을 바꾼다.'],
                ['고정 영역과 expert의 적합성','CIC seed 43 영역 3에서는 담당 e3보다 global·상수·e1이 모두 높다. ToN에서도 담당 expert가 항상 영역 내 최고는 아니다.',
                 '잔차 cluster가 곧 담당 expert의 최적 적용 영역이라는 가정을 수정한다. 영역 정의와 context 선택의 적합성을 validation에서 함께 점검한다.'],
                ['현재 선택 방식의 고유 이득','완료된 비교에서 클래스 비율 일치 무작위가 비슷하거나 더 좋은 대상이 있다.',
                 'expert의 유용성과 행 선택 알고리즘의 우위를 구분한다. 클래스 구성을 고정한 선택 비교를 충분한 seed로 완성한다.']],'exp57-verdict'),
            '<p><b>데이터·클래스별로 식별력과 안정성을 나누어 판단한다.</b> '
            'ToN scanning은 구체적인 식별력 이득이 있고, CIC infiltration과 ToN ransomware는 context·안정성 개선의 우선 대상이다. '
            '운영에 충분한지는 허용 FPR·최소 recall을 먼저 정한 뒤, 고정 임계값이 새 시간 데이터에서 그 기준을 만족하는지로 판정해야 한다.</p>',
            '<p>보고서를 완성된 3-seed 비교로 확대하려면 외부 seed 42 결과를 합치고, CIC seed 44의 비정상 종료를 해결해 누락 bank를 완성해야 한다. '
            '배치를 125,000·62,500으로 줄인 재시도도 expert 추론 중 native segmentation fault로 종료됐으므로, '
            '다음 복구는 동일 환경의 배치 축소 반복보다 PyTorch/CUDA 실행 환경 고정과 해당 expert의 독립 프로세스 재현부터 진행한다.</p>']


def update_index():
    text=INDEX.read_text();m=re.search(r'const REPORTS = (.*?);\n',text);reports=json.loads(m.group(1))
    for r in reports:
        if r['id']=='scorer_verifier_target_0918':
            r.update(titleKr='Expert 역량 검증 — Oracle 정정과 K=4 대조 결과',
                     titleEn='Corrected bank oracle and controlled expert evaluation',
                     verdict='09/23 결과 반영 · 완료 seed 범위와 미완료 구분',verdictClass='warn')
    text=text[:m.start(1)]+json.dumps(reports,ensure_ascii=False)+text[m.end(1):]
    pattern=r'(<button class="jump" data-jump="scorer_verifier_target_0918">).*?(</button><p>).*?(</p>)'
    replacement=r'\1<h3>EXP56–58 — Oracle 정정과 expert 생성 품질 대조</h3>\2기존 다섯 조건에 고정 bank의 정답 참조 상한을 추가했다. K=4의 현재 선택·비율 일치 무작위·균형 무작위를 비교하고, 전체 test의 FP와 저오탐 식별력을 구분한다. 완료된 seed만 집계하며 운영점 전이와 반복 안정성의 개선 대상을 제시한다.\3'
    text,n=re.subn(pattern,replacement,text,count=1,flags=re.S)
    if n!=1:raise ValueError('Overview entry not found')
    text=text.replace('09-18<br>09-22 개정','09-18<br>09-23 갱신')
    INDEX.write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    jobs,status,sources=load_jobs()
    print(json.dumps(dict(complete_jobs=list(sources),status=status),ensure_ascii=False,indent=2))
    if args.audit_only:return
    OUT.mkdir(parents=True,exist_ok=True)
    focuses=focus_rows(jobs);quality_fig,oracle_fig=figures(jobs,focuses)
    parts=['<section id="exp57-followup" aria-label="2026-09-23 후속 실험 결과">',
           '<p class="eyebrow">2026-09-23 후속 결과 · EXP57 / EXP58</p>']
    parts+=oracle_section(oracle_fig)+result_section(jobs,status,focuses,quality_fig)+detail_section(jobs)+verdict_section()
    parts += ['<details><summary>계산·검증 출처</summary><p>EXP58: <code>tabpfn/results/exp58_six_condition_20260922</code>. '
              'EXP57의 실제 채택 경로와 CSV SHA-256은 <code>docs/research/20260923/exp57_report_manifest.json</code>에 기록했다. '
              '모든 완료본의 클래스별 TP·FP·FN으로 F1·전체 Macro-F1·Accuracy를 재검산하고, context 대조의 클래스별 행 수·총 예산과 test support 일치를 검증했다.</p></details></section>']
    fragment='\n'.join(parts)+'\n'
    (OUT/'exp57_followup_fragment.html').write_text(fragment)
    manifest=dict(date='2026-09-23',sources=sources,status=status,oracle_root=str(ORACLE.relative_to(ROOT)),
                  all_six_workers_complete=len(jobs)==6,html_report=str(REPORT.relative_to(ROOT)),online_artifact_updated=False)
    (OUT/'exp57_report_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    text=REPORT.read_text()
    block=START+'\n'+fragment+END+'\n'
    if START in text:
        text=re.sub(re.escape(START)+r'.*?'+re.escape(END)+r'\n?',lambda _:block,text,flags=re.S)
    else:
        if text.count('</main>')!=1:raise ValueError('Report main anchor mismatch')
        text=text.replace('</main>',block+'</main>')
    text=text.replace('0918 연구 보고서 · 2026-09-22 평가 설계 수정 · EXP56','0918 연구 보고서 · 2026-09-23 후속 결과 · EXP56–58')
    text=text.replace('정답 없는 다섯 조건 비교, 점수 차이의 원인, 지표별 검증 목적과 생성 품질의 근거를 채울 추가 실험.',
                      '다섯 조건과 정정한 oracle의 비교, K=4 context 대조, 저오탐 식별력·운영점 전이·반복 안정성의 검증.')
    if '6. 정답 참조 bank oracle' not in text:
        pattern=r'(<table data-kind="definitions">.*?)(</tbody>)'
        row='<tr><td>6. 정답 참조 bank oracle</td><td>고정 global·expert의 실제 예측 후보 안에서 전체 Macro-F1을 최대화</td><td>모델 후보 집합의 참조 상한 · 생성 품질 평가에서는 제외</td></tr>'
        text,n=re.subn(pattern,lambda m:m.group(1)+row+m.group(2),text,count=1,flags=re.S)
        if n!=1:raise ValueError('Definition table not found')
    text=text.replace('각 자료가 답하는 질문을 § 4–6에 연결한다.', '기존 설계는 § 4–6, 새 K=4 실행 결과는 § 8–11에 연결한다.')
    text=text.replace('현재 수치는 기존 bank를 이용한 재진단이다. CIC K=8·C=7은 새 제약을 충족하지 않으므로 새 설계의 결과로 주장하지 않는다. ToN K=7·C=10은 개수 조건을 충족한다. 이번에 bank 재학습은 하지 않았다.',
                      '1–6절은 기존 bank를 재진단한 09/22 기록이다. 기존 CIC K=8·C=7과 ToN K=7·C=10의 결과를 아래 새 K=4 bank 결과와 구분한다. Oracle 정정은 7절, K&lt;C로 재학습한 결과는 8–11절에 이어서 제시한다.')
    if 'id="exp57-nav"' not in text:
        text=text.replace('</header>', '</header><nav id="exp57-nav" class="note" aria-label="후속 결과 바로가기">'
                          '<p>09/23 후속 결과: <a href="#oracle-correction">Oracle 정정</a> · '
                          '<a href="#exp57-results">K=4 실험 범위</a> · <a href="#exp57-quality">식별력 대조</a> · '
                          '<a href="#exp57-details">Expert별 근거</a> · <a href="#exp57-verdict">판정·개선 방향</a></p></nav>',1)
    text=text.replace('<h2>5. 현재 결과로 내릴 수 있는 판단</h2>','<h2>5. 기존 bank의 판단 · 09/22 기준</h2>')
    text=text.replace('<h2>6. 추가로 필요한 실험과 각각의 판정 기준</h2>',
                      '<h2>6. 후속 실험의 설계와 판정 기준 · 실행 전 계획</h2><p class="note">아래는 후속 실험을 설계한 당시의 계획이다. 09/23 현재의 실행 범위와 결과는 <a href="#exp57-results">8–11절</a>에서 확인한다.</p>')
    REPORT.write_text(text)
    update_index()
    from sync_html_reports import sync
    sync(INDEX,False)
    if sync(INDEX,True):raise ValueError('Report index sync failed')
    print('Updated 0918 source and post_tabpfn.html; online artifact is not updated.')


if __name__=='__main__':main()
