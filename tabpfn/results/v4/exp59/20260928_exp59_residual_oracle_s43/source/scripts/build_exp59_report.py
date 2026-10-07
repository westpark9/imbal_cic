#!/usr/bin/env python3
"""Build a new 0928 report from audited single-seed residual-oracle results."""
import argparse
import base64
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
from report_typography import apply_typography

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'tabpfn/results/20260928_exp59_residual_oracle_s43'
REPORT_ID='scorer_verifier_target_0928'
REPORT=ROOT/'lablog/html_report'/f'{REPORT_ID}.html'
OLD=ROOT/'lablog/html_report/scorer_verifier_target_0918.html'
INDEX=ROOT/'lablog/html_report/post_tabpfn.html'
DOCS=ROOT/'docs/research/20260928'
ARMS=['designed','matched_random','balanced_random']
ARM_NAMES={'designed':'현재 residual 선택','matched_random':'클래스 비율 일치 무작위','balanced_random':'같은 크기 균형 무작위'}
DATASETS={'cic2018':'CIC-IDS2018','toniot':'ToN-IoT'}


def f(x):return f'{float(x):.4f}'
def n(x):return f'{int(x):,}'
def signed(x):return f'{float(x):+.4f}'
def esc(x):return html.escape(str(x))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    data={}
    for ds in DATASETS:
        p=RESULTS/f'{ds}_s43'
        complete=json.loads((p/'COMPLETE.json').read_text())
        audit=json.loads((p/'RECONSTRUCTION_COMPLETE.json').read_text())
        assert complete['seed']==43 and complete['K']==4
        if audit.get('evaluation_mode')=='fresh_single_seed_bank':
            assert audit['same_global_context_anchor_expert_pool_test_ids']
            assert audit['old_predictions_reused'] is False
        else:
            assert audit['observable_test_assignment_mismatches']==0
            assert all(audit[f'expert{k}_context_ids_exact'] for k in range(1,5))
        d={'path':p,'complete':complete,'audit':audit,
           'manifest':json.loads((p/'source_manifest.json').read_text()),
           'constant':json.loads((p/'constant_reference.json').read_text())}
        for name in ['summary','per_class','region_metrics','region_per_class','oracle_changes','confusion',
                     'probability_metrics','context_compositions','residual_region_composition','train_pool_partition']:
            d[name]=pd.read_csv(p/f'{name}.csv')
        d['global']=d['per_class'].query('model == "global"').set_index('class').sort_values('support',ascending=False)
        d['names']=d['global'].index.tolist()
        d['summary']=d['summary'].set_index('model')
        d['probs']=d['probability_metrics'].set_index(['model','class'])
        d['pc']=d['per_class'].set_index(['model','class'])
        for model,q in d['per_class'].groupby('model'):
            assert int(q.support.sum())==complete['rows']
            np.testing.assert_allclose(q.f1,2*q.TP/(2*q.TP+q.FP+q.FN),atol=1e-12)
            np.testing.assert_allclose(q.f1.mean(),d['summary'].loc[model,'macro_f1'],atol=1e-12)
        for arm in ARMS:
            row=d['summary'].loc[arm+'_residual_oracle']
            np.testing.assert_allclose(row.delta_oracle,row.macro_f1-max(d['summary'].loc['global','macro_f1'],d['constant']['macro_f1']),atol=1e-12)
        data[ds]=d
    return data


def configure_plots():
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
        'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#cbd5e1','text.color':'#203044','svg.fonttype':'path','svg.hashsalt':'exp59-0928'})


def global_plot(data):
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.7),layout='constrained')
    for ax,(ds,d) in zip(axes,data.items()):
        q=d['global'];x=np.arange(len(q))
        ax.barh(x,q.f1,color=['#b85d4d' if z<.5 else '#7b9aaf' for z in q.f1],height=.65)
        for i,v in enumerate(q.f1):ax.text(v+.015,i,f(v),va='center',fontsize=9)
        ax.set(yticks=x,yticklabels=q.index,xlim=(0,1.16),xlabel='Global class F1',title=DATASETS[ds]+' · seed 43')
        ax.invert_yaxis();ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
    return save_figure(fig,'exp59_global_limits','Global의 클래스별 F1. 행 순서는 test 표본 수 내림차순이며, 모든 그림·표는 같은 seed 43 bank를 사용한다.')


def oracle_plot(data):
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5),layout='constrained')
    for ax,(ds,d) in zip(axes,data.items()):
        s=d['summary'];g=s.loc['global','macro_f1'];c=d['constant']['macro_f1']
        vals=[s.loc[a+'_residual_oracle','macro_f1'] for a in ARMS]
        ax.axvline(g,color='#596b7a',linestyle='--',label='Global')
        ax.axvline(c,color='#b16a49',linestyle=':',label='동일 배정 최선 상수')
        ax.scatter(vals,range(3),s=90,color=['#245e8d','#4e8d79','#967eae'],zorder=3)
        for i,v in enumerate(vals):
            ax.text(v+.009,i,f(v)+'  (Δ '+signed(v-max(g,c))+')',va='center',fontsize=9)
        ax.set(yticks=range(3),yticklabels=[ARM_NAMES[a] for a in ARMS],title=DATASETS[ds],xlabel='잔차 소속 oracle Macro-F1',ylim=(-.6,2.6))
        ax.invert_yaxis();ax.set_xlim(max(0,min(vals+[g,c])-.06),min(1.25,max(vals+[g,c])+.23))
        ax.grid(axis='x',alpha=.12);ax.legend(loc='lower right',fontsize=8)
    return save_figure(fig,'exp59_residual_oracle','잔차 소속 oracle과 동일 배정의 최선 상수 대조. Δ는 Global·상수 중 높은 기준 대비 차이이며, 정답을 사용하지 않는 운영 성능을 뜻하지 않는다.')


def probability_cell(d,model,cl):
    pm='global_raw' if model=='global' else model+'_raw'
    p=d['probs'].loc[(pm,cl)];g=d['probs'].loc[('global_raw',cl)]
    q=d['pc'].loc[(model,cl)]
    v=float(p.AP);delta=v-float(g.AP)
    color=f'rgba({"36,94,141" if delta>=0 else "184,93,77"},{min(.35,abs(delta)*.7):.3f})'
    return Markup(f'<span class="prob-cell" style="background:{color}" data-ap="{v:.15g}" '
        f'data-r="{p.R_at_FPR_001:.15g}" data-f1="{q.f1:.15g}" '
        f'data-g-ap="{g.AP:.15g}" data-g-r="{g.R_at_FPR_001:.15g}" data-g-f1="{d["global"].loc[cl,"f1"]:.15g}">{f(v)}</span>')


def expert_tables(ds,d):
    out=[f'<h3>{DATASETS[ds]} — 모든 expert × 모든 클래스</h3>']
    out.append(f'<label>Context 조건 <select class="arm-select" data-dataset="{ds}">'+''.join(f'<option value="{a}">{ARM_NAMES[a]}</option>' for a in ARMS)+'</select></label>')
    for arm in ARMS:
        out.append(f'<div class="arm-panel" data-dataset="{ds}" data-arm="{arm}"'+(' hidden' if arm!='designed' else '')+'>')
        rows=[]
        for cl in d['names']:
            rows.append([cl,n(d['global'].loc[cl,'support']),probability_cell(d,'global',cl),
                *[probability_cell(d,f'{arm}_e{k}',cl) for k in range(1,5)]])
        out.append(table(['클래스','Test 수','Global','e1','e2','e3','e4'],rows,f'exp59-{ds}-{arm}-quality'))
        out.append('</div>')
    focus='infiltration' if ds=='cic2018' else 'scanning'
    out.append('<h3>같은 expert slot의 context 효과</h3>')
    out.append(f'<label>클래스 <select class="class-select" data-dataset="{ds}">'+''.join(f'<option value="{esc(c)}"'+(' selected' if c==focus else '')+f'>{esc(c)}</option>' for c in d['names'])+'</select></label>')
    for cl in d['names']:
        out.append(f'<div class="class-panel" data-dataset="{ds}" data-class="{esc(cl)}"'+(' hidden' if cl!=focus else '')+'>')
        g=d['probs'].loc[('global_raw',cl)]
        out.append(f'<p class="note">Global AP {f(g.AP)} · R@0.1% {f(g.R_at_FPR_001)}. 아래 모든 값은 양성·음성을 포함한 동일 전체 test에서 계산했다.</p>')
        rows=[]
        for k in range(1,5):
            ps=[d['probs'].loc[(f'{a}_e{k}_raw',cl)] for a in ARMS]
            rows.append([f'e{k}',*[f(p.AP) for p in ps],*[f(p.R_at_FPR_001) for p in ps]])
        out.append(table(['Expert','현재 AP','비율 일치 AP','균형 AP','현재 R@0.1%','비율 일치 R','균형 R'],rows,f'exp59-{ds}-{cl}-context'))
        out.append('</div>')
    source=Path(d['manifest']['source']);same=[]
    for k in range(1,5):
        if np.array_equal(np.sort(np.load(source/f'contexts/designed_e{k}.npy')),np.sort(np.load(source/f'contexts/matched_random_e{k}.npy'))):same.append(f'e{k}')
    if same:out.append('<p class="note">'+', '.join(same)+'의 현재 선택과 비율 일치 무작위는 context ID 집합도 같다. 이 slot에서는 서로 다른 행 선택의 효과를 비교한 것으로 세지 않는다.</p>')
    return '\n'.join(out)


def region_tables(ds,d):
    out=[f'<h3>{DATASETS[ds]} — 잔차 소속과 클래스 구성</h3>']
    comp=d['residual_region_composition'].query('split == "test"').sort_values('region')
    rows=[]
    for _,r in comp.iterrows():
        parts=[f'{c} {n(r[c])}' for c in d['names'] if r[c]>0]
        rows.append([f'R{r.region} → e{r.region}',n(r.rows),int(r.classes_present),f'{100*r.dominant_fraction:.1f}%',', '.join(parts)])
    out.append(table(['소속 → 지정 expert','Test 수','클래스 수','최다 클래스 비율','실제 클래스 구성'],rows,f'exp59-{ds}-region-composition'))
    for arm in ARMS:
        out.append('<details'+(' open' if arm=='designed' else '')+f'><summary>{ARM_NAMES[arm]} · 같은 영역의 모든 예측기 비교</summary>')
        q=d['region_metrics'].query('arm == @arm');rows=[]
        for k in range(1,5):
            sub=q[q.region==k].set_index('model')
            if sub.empty:continue
            models=['global','e1','e2','e3','e4','local_majority_constant']
            vals=[sub.loc[m,'present_macro_f1'] for m in models]
            cells=[]
            for m,v in zip(models,vals):
                cell=str(metric(v,max(vals)))
                if m==f'e{k}':cell+='<small class="assigned">지정</small>'
                cells.append(Markup(cell))
            rows.append([f'R{k}',n(sub.iloc[0].rows),int(sub.iloc[0].classes_present),*cells])
        out.append(table(['영역','행 수','클래스 수','Global','e1','e2','e3','e4','영역 최선 상수'],rows,f'exp59-{ds}-{arm}-region'))
        out.append('</details>')
    out.append('<p class="note">영역에 등장하는 클래스들의 F1 평균이다. 동일 행의 예측기끼리 비교한다. 서로 다른 영역 또는 전체 test Macro-F1과 직접 비교하지 않는다. 영역 최선 상수는 그 영역의 최다 클래스로, 앞의 전체 Macro-F1 최적 상수 매핑과 목적함수가 다르다. 단일 클래스 영역도 oracle 전체 집계에는 남기되 높은 점수를 식별력의 증거로 세지 않는다.</p>')
    return '\n'.join(out)


def next_steps(data):
    rows=[]
    for ds,cl in [('cic2018','infiltration'),('toniot','scanning'),('toniot','ransomware')]:
        d=data[ds];g=d['probs'].loc[('global_raw',cl)]
        candidates=[(float(d['probs'].loc[(f'{a}_e{k}_raw',cl),'AP']),a,k)
                    for a in ARMS for k in range(1,5)]
        ap,arm,k=max(candidates,key=lambda x:x[0])
        r=d['probs'].loc[(f'{arm}_e{k}_raw',cl)]
        improvement=('이 조건의 식별력 이득을 실제 라우팅에서 회수할 수 있는지 확인한다.' if ap>g.AP else
                     '혼동 클래스의 음성을 포함하도록 특화 block을 보완하고 같은 slot에서 다시 비교한다.')
        rows.append(f'<li><b>{DATASETS[ds]} {esc(cl)}:</b> Global AP {f(g.AP)} 대비 관찰된 최고 AP는 '
            f'{ARM_NAMES[arm]} e{k}의 {f(ap)}다. 이 조건의 R@0.1%는 {f(r.R_at_FPR_001)} '
            f'(Global {f(g.R_at_FPR_001)}). {improvement}</li>')
    return '<ul>'+''.join(rows)+'</ul><p class="note">관찰된 최고값은 전체 행렬을 요약한 탐색 결과다. 운영 context를 test로 선택했다는 뜻이 아니며, 후속 선택은 D_tune에서 고정하고 다른 seed에서 반복성을 확인한다. 같은 ID 집합인 context 대조는 독립적인 행 선택 효과로 세지 않는다.</p>'


def build(data):
    old=OLD.read_text()
    title=re.search(r'<title>(.*?)</title>',old,re.S).group(1)
    h1=re.search(r'<h1>(.*?)</h1>',old,re.S).group(1)
    css=re.search(r'<style>(.*?)</style>',old,re.S).group(1)
    css+='''
    .hero-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:24px 0}.card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:20px}.card h3{margin:0 0 8px}.value{font-size:26px;font-weight:700;color:var(--acc)}.card p{font-size:13px;margin:6px 0}.definition{background:#edf2f7;border-left:4px solid #245e8d;padding:14px 20px}.equation{font-size:17px;font-weight:700;line-height:1.9}.pill{display:inline-block;padding:2px 9px;background:#e8eff6;border-radius:12px;font-size:12px;margin-right:6px}nav a{margin-right:16px}select{font:inherit;background:white;border:1px solid #bbcbd8;border-radius:5px;padding:6px 10px;margin:6px 10px}label{font-size:13px}.prob-cell{display:block;padding:5px;border-radius:3px;min-width:54px}.assigned{display:block;color:#245e8d;font-weight:700}td{font-variant-numeric:tabular-nums}.small{font-size:12px;color:var(--muted)}[hidden]{display:none!important}footer{border-top:1px solid var(--line);margin-top:44px;padding-top:20px}a{color:var(--acc)}@media(max-width:720px){.hero-grid{grid-template-columns:1fr}.equation{font-size:14px}}
    '''
    parts=[f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{css}</style></head><body><main>',
      f'<header><p class="eyebrow">0928 연구 보고서 · EXP59 · CIC2018 / ToN-IoT · 단일 seed 43</p><h1>{h1}</h1><p class="lede">Global의 보완 대상부터, 정답 residual 소속에 따른 oracle, 모든 expert의 식별력과 세 context 조건의 효과까지 같은 bank에서 확인한다.</p><span class="pill">K=4</span><span class="pill">단일 seed bank 재계산</span><span class="pill">전체 residual 상태 보존</span></header>',
      '<nav class="note"><a href="#global">Global의 한계</a><a href="#oracle">새 oracle</a><a href="#quality">Expert·context 식별력</a><a href="#regions">소속별 분류</a><a href="#next">개선 방향</a><a href="#protocol">실험·검증 근거</a></nav>',
      '<div class="hero-grid">']
    for ds,d in data.items():
        q=d['summary'].loc['designed_residual_oracle'];g=d['summary'].loc['global','macro_f1'];c=d['constant']['macro_f1']
        parts.append(f'<article class="card"><h3>{DATASETS[ds]}</h3><div class="value">ΔO {signed(q.delta_oracle)}</div><p>현재 context의 잔차 소속 oracle {f(q.macro_f1)}</p><p>Global {f(g)} · 동일 배정 최선 상수 {f(c)}</p><p>Test {n(d["complete"]["rows"])}행 · seed 43</p></article>')
    parts+=['</div>','<p class="note">ΔO는 정답을 사용한 잔차 배정에서 Global과 상수 예측 중 높은 기준을 넘는 추가 기여다. 실제 운영 성능이나 임의 선택 정책의 상한이 아니다. 전체 test 식별력과 함께 읽으며, 단일 seed·기존 개발 holdout 범위의 결과다.</p>',
       '<h2 id="global">1. Global은 어디에서 보완이 필요한가?</h2>',global_plot(data)]
    for ds,d in data.items():
        parts.append(f'<h3>{DATASETS[ds]} — 오분류 방향과 식별력</h3>');rows=[]
        cm=d['confusion'].query('model == "global"')
        for cl,r in d['global'].iterrows():
            off=cm[(cm.actual==cl)&(cm.predicted!=cl)].sort_values('rows',ascending=False)
            top=off.iloc[0];p=d['probs'].loc[('global_raw',cl)]
            direction=f'{top.predicted} ({n(top.rows)}행)' if top.rows else '없음'
            rows.append([cl,n(r.support),f(r.precision),f(r.recall),f(r.f1),f(p.AP),f(p.R_at_FPR_001),direction])
        parts.append(table(['클래스','Test 수','Precision','Recall','F1','AP','R@0.1%','가장 많은 오분류 목적지'],rows,f'exp59-{ds}-global'))
        weak=d['global'].sort_values('f1').head(3)
        parts.append('<p class="note">보완 우선 관찰 대상: '+', '.join(f'{esc(c)} (F1 {f(r.f1)})' for c,r in weak.iterrows())+'. Context 효과는 아래에서 같은 평가 샘플과 같은 예산으로 비교한다.</p>')
    parts+=['<h2 id="oracle">2. 새 oracle: 잔차 소속에 지정된 expert의 원래 답</h2>',
      '<div class="definition"><p><b>①</b> D_expert에서 학습한 전체 residual cluster를 고정한다. <b>②</b> Test 정답으로 동일 residual 표현을 계산한다. <b>③</b> 가장 가까운 cluster의 expert 하나에 배정한다. <b>④</b> 그 expert의 원래 클래스 예측을 그대로 채점한다.</p><p class="equation">s(x,y) = 학습 시 정규화한 [z(x), pG(x), onehot(y) − pG(x), log(1 + clipped lossG)]<br>kO = argminₖ ‖s(x,y) − μₖ‖² &nbsp; → &nbsp; ŷO = expertₖO(x)</p><p>다른 expert가 맞혔는지에 따른 재선택과 global 오답 복구는 없다. 정규화·손실 가중치·clipping·cluster 중심은 test에서 다시 맞추지 않았다.</p></div>',
      '<p>Expert가 클래스 하나만 반복해도 높아지는 효과는 <b>동일 배정의 최선 상수 예측</b>으로 검사한다. 영역별 상수 클래스의 모든 조합을 열거하여 전체 Macro-F1을 공동 최적화했다.</p>',
      '<p class="equation">ΔO = Macro-F1(oracle) − max{Macro-F1(global), Macro-F1(동일 배정 최선 상수)}</p>',
      '<p class="note">클래스별 상수 expert만으로 oracle이 1.0이면 상수 대조도 1.0이므로 추가 기여는 0이다. ΔO&gt;0은 이 특정 상수 설명과 Global을 넘는 조건부 이득이며, 정답 배정의 영향을 모두 제거한 일반화 성능을 의미하지 않는다.</p>',oracle_plot(data)]
    rows=[]
    for ds,d in data.items():
        for a in ARMS:
            r=d['summary'].loc[a+'_residual_oracle']
            rows.append([DATASETS[ds],ARM_NAMES[a],f(d['summary'].loc['global','macro_f1']),f(d['constant']['macro_f1']),f(r.macro_f1),signed(r.delta_oracle),f(r.accuracy)])
    parts.append(table(['데이터','Context','Global F1','최선 상수 F1','잔차 oracle F1','ΔO','Oracle accuracy'],rows,'exp59-oracle-summary'))
    parts.append('<details><summary>Oracle의 클래스별 F1과 Global 정오 변화</summary>')
    for ds,d in data.items():
        rows=[]
        for cl in d['names']:
            vals=[d['pc'].loc[(a+'_residual_oracle',cl),'f1'] for a in ARMS]
            change=d['oracle_changes'].query('arm == "designed"').set_index('class').loc[cl]
            rows.append([cl,f(d['global'].loc[cl,'f1']),*[f(v) for v in vals],n(change.fixed),n(change.harmed)])
        parts.append(f'<h3>{DATASETS[ds]}</h3>'+table(['클래스','Global','현재','비율 일치','균형','현재: 오답→정답','현재: 정답→오답'],rows,f'exp59-{ds}-oracle-classes'))
    parts+=['</details>','<h2 id="quality">3. Expert가 구분하는가? Context를 바꾸면 달라지는가?</h2>',
      '<p>모든 expert의 동일 전체 test 예측을 사용한다. AP·저오탐 recall은 양성과 음성을 모두 포함한 클래스별 확률 식별력이다. F1은 원래 다중 클래스 argmax 예측의 결과다. 한 클래스에 여러 expert가 기여할 수 있으므로 전체 행렬을 제시한다.</p>',
      '<label>전체 행렬 지표 <select id="quality-metric"><option value="ap">AP</option><option value="r">Recall @ FPR ≤ 0.1%</option><option value="f1">원래 예측 F1</option></select></label>',
      '<p class="note">파란 배경은 같은 클래스의 Global보다 높음, 주황은 낮음. R@0.1%는 test ROC의 기술적 식별력 지표이며 validation에서 고정한 운영 임계값의 성능과 구분한다.</p>']
    for ds,d in data.items():parts.append(expert_tables(ds,d))
    parts+=['<details><summary>Context 구성과 고정 조건</summary>',
      '<p>같은 dataset·seed 안에서 Global, anchor, K=4, 전체 residual 소속, expert별 context 크기를 공유한다. 비율 일치 무작위는 특화 block의 클래스별 행 수까지 같고, 균형 무작위는 총 행 수가 같다. 세 조건 모두 D_expert에서 특화 block을 고른다.</p>']
    for ds,d in data.items():
        rows=[]
        for _,r in d['context_compositions'].iterrows():rows.append([ARM_NAMES[r.arm],f'e{r.expert}',n(r.block_rows),*[n(r[c]) for c in d['names']]])
        parts.append(f'<h3>{DATASETS[ds]} · 특화 block</h3>'+table(['조건','Expert','행 수',*d['names']],rows,f'exp59-{ds}-context-composition'))
    parts+=['</details>','<h2 id="regions">4. 같은 residual 소속 안에서 직접 비교</h2>',
      '<p>위 oracle에서 고정한 test 소속 R1–R4를 그대로 사용한다. 각 집합에서 Global·모든 expert·상수를 실제 재채점했다. Context 조건을 바꿔도 평가 집합을 다시 나누지 않는다. 정답 residual을 사용하는 소속 조건부 진단이며, 전체 test 식별력과 함께 판단한다.</p>']
    for ds,d in data.items():parts.append(region_tables(ds,d))
    parts+=['<h2 id="next">5. 확인된 기여와 다음 개선</h2>']
    for ds,d in data.items():
        r=d['summary'].loc['designed_residual_oracle'];controls=[d['summary'].loc[a+'_residual_oracle','macro_f1'] for a in ARMS[1:]]
        text=(f'현재 context는 Global·상수 기준 대비 ΔO {signed(r.delta_oracle)}의 조건부 추가 기여를 보였다.' if r.delta_oracle>0 else
              f'현재 context의 ΔO는 {signed(r.delta_oracle)}이다. 아래 context 대조와 혼동 클래스의 음성 구성을 기준으로 특화 block을 보완한다.')
        comparison=('현재 선택의 oracle이 두 무작위 대조보다 높다. 이 차이의 반복성은 후속 seed에서 확인한다.' if r.macro_f1>max(controls) else
                    '같은 예산의 무작위 대조 중 현재 선택 이상인 조건이 있다. 현재 행 선택 자체의 이득과 context 구성의 이득을 구분하고, 유리한 구성을 다음 모델의 대조로 유지한다.')
        parts.append(f'<h3>{DATASETS[ds]}</h3><p>{text} {comparison}</p>')
    parts+=[next_steps(data),
      '<p class="note">이번 preliminary는 seed 43 한 번의 동일 bank 비교다. 새 K=4 scorer/verifier를 학습하지 않았으므로 0918의 다른 bank S-only/S+V 수치를 옮겨 붙이지 않는다.</p>',
      '<h2 id="protocol">6. 데이터 분할·재사용·검증 근거</h2>',
      table(['분할','비율','역할'],[['Train → D_global','약 50%','Global context·공통 anchor·특징 표현'],['Train → D_expert','약 25%','Residual 채굴·expert 특화 block'],['Train → D_route','약 25%','Scorer/verifier용으로 분리; 이번 EXP59에서는 추가 학습하지 않음'],['Val → D_tune / D_cal','약 40% / 60%','선택 / 임계값 보정'],['Test','기존 소속 유지·cap 없음','두 평가에 동일하게 사용']],'exp59-splits'),
      '<p>내부 분할은 클래스×시나리오별 시간순이다. 서로 다른 모델 라벨을 가진 동일 벡터의 행을 제거한 기존 벤치마크이며, 원래 train/val/test 소속은 보존했다. 동일 라벨 중복은 남아 있고 새로운 독립 holdout은 아니다. Expert context는 D_expert의 특화 block과 D_global에서 뽑은 공통 anchor의 합이다.</p>']
    rows=[]
    for ds,d in data.items():
        a=d['audit'];rows.append([DATASETS[ds],43,n(d['complete']['rows']),'동일','새로 계산·저장','동일 run',n(a['full_residual_vs_observable_changed_rows'])])
    parts.append(table(['데이터','Seed','Test 수','기존 pool·Test ID','전체 residual 중심','Expert 예측 출처','전체 residual로 배정 변경'],rows,'exp59-reconstruction-audit'))
    parts+=['<p>기존 EXP57에는 전체 residual 상태가 저장되지 않았고, 같은 seed의 복원에서도 Global 확률이 달라져 과거 예측과 연결하지 않았다. 기존 Global/anchor/expert pool/Test ID·정답을 유지하되 두 bank를 새로 계산했다. 같은 실행에서 만든 전체 residual 중심과 expert 예측을 사용하고, 모든 F1을 재계산하여 저장 확률 argmax와 대조했다. 반복 bootstrap·추가 임계값 보정·scorer/verifier 학습은 이번 범위에 포함하지 않았다.</p>',
      '<details><summary>재현 경로와 검증 파일</summary><p>실험: <code>tabpfn/scripts/exp59_fresh_worker.py</code> · 평가: <code>tabpfn/scripts/exp59_residual_membership_oracle.py</code><br>결과: <code>tabpfn/results/20260928_exp59_residual_oracle_s43/{cic2018,toniot}_s43</code><br>각 결과의 <code>source_manifest.json</code>에 이번 평가 입력의 SHA-256, <code>reconstruction_audit.json</code>에 기존 ID 보존·새 실행 여부, <code>constant_reference.json</code>에 상수 최적화 근거를 저장했다. 전체 residual 중심·정규화·test 배정도 새로 보존했다.</p><p>평가 검증: 오답 복구 금지, 최선 상수와 독립 전수열거 일치, 클래스별 상수 expert의 추가 기여 0, 혼합 영역의 실제 분류 이득을 자동 테스트했다.</p></details>',
      '<footer><p class="note">0918과 같은 제목의 새 0928 탭 · EXP59 · seed 43 · 작성 2026-09-28. 이전 0918 본문은 보존했다.</p></footer>',
      '''<script>
      document.getElementById('quality-metric').addEventListener('change',function(){const m=this.value;document.querySelectorAll('.prob-cell').forEach(c=>{const v=Number(c.dataset[m]),g=Number(c.dataset['g'+m[0].toUpperCase()+m.slice(1)]),d=v-g;c.textContent=v.toFixed(4);c.style.background='rgba('+(d>=0?'36,94,141':'184,93,77')+','+Math.min(.35,Math.abs(d)*.7)+')';});});
      document.querySelectorAll('.arm-select').forEach(s=>s.addEventListener('change',()=>document.querySelectorAll('.arm-panel').forEach(p=>{if(p.dataset.dataset===s.dataset.dataset)p.hidden=p.dataset.arm!==s.value;})));
      document.querySelectorAll('.class-select').forEach(s=>s.addEventListener('change',()=>document.querySelectorAll('.class-panel').forEach(p=>{if(p.dataset.dataset===s.dataset.dataset)p.hidden=p.dataset.class!==s.value;})));
      </script></main></body></html>''']
    page=apply_typography('\n'.join(parts))
    REPORT.write_text(page)
    return page


def integrate(page,data):
    s=INDEX.read_text();m=re.search(r'const REPORTS = (.*?);\n',s)
    reports=json.loads(m.group(1));old=next(r for r in reports if r['id']=='scorer_verifier_target_0918')
    reports=[r for r in reports if r['id']!=REPORT_ID]
    new={**old,'id':REPORT_ID,'date':'2026-09-28','titleKr':old['titleKr'],
         'titleEn':'Residual-membership oracle and context evaluation · seed 43',
         'verdict':'09/28 · 단일 seed 43 · 전체 residual oracle·context 대조 완료',
         'verdictClass':'ok','b64':base64.b64encode(page.encode()).decode()}
    reports.insert(1,new)
    s=s[:m.start(1)]+json.dumps(reports,ensure_ascii=False)+s[m.end(1):]
    flow='<div class="flow-step" id="exp59-overview"><time>09-28</time><div class="flow-rail-dotcol"><div class="d"></div><div class="l"></div></div><div class="body"><button class="jump" data-jump="'+REPORT_ID+'"><h3>EXP59 — 잔차 소속 oracle과 expert/context 역량</h3></button><p>ToN·CIC2018의 seed 43, K=4 bank. Global의 한계부터 전체 residual 배정 oracle, 상수 효과를 제외한 추가 기여, 모든 expert의 식별력과 세 context 대조를 연결한다.</p></div></div>'
    s=re.sub(r'<!-- EXP59_FLOW_START -->.*?<!-- EXP59_FLOW_END -->\n?','',s,flags=re.S)
    anchor='<h2>연구 흐름 (최신순)</h2>'
    s=s.replace(anchor,anchor+'\n<!-- EXP59_FLOW_START -->'+flow+'<!-- EXP59_FLOW_END -->',1)
    finding='<li><p><strong>(09-28 EXP59) 잔차 소속 oracle은 배정된 expert의 원래 답을 채점한다.</strong> '
    finding+=' · '.join(DATASETS[ds]+' 현재 context ΔO '+signed(d['summary'].loc['designed_residual_oracle','delta_oracle']) for ds,d in data.items())
    finding+='. 같은 정답 residual 배정의 최선 상수를 대조하고, expert 역량은 전체 test 식별력과 동일 소속 내 실제 수치로 확인했다. 단일 seed 43의 preliminary 결과다.</p></li>'
    s=re.sub(r'<!-- EXP59_FINDING_START -->.*?<!-- EXP59_FINDING_END -->\n?','',s,flags=re.S)
    s=s.replace('<ol class="findings">','<ol class="findings">\n<!-- EXP59_FINDING_START -->'+finding+'<!-- EXP59_FINDING_END -->',1)
    count=sum(r.get('group')=='report' for r in reports)
    s=s.replace('8&ndash;9월의 보고서 아홉 편을 최신순으로 묶었습니다.',f'8&ndash;9월의 보고서 {count}편을 최신순으로 묶었습니다.')
    INDEX.write_text(s)


def main():
    p=argparse.ArgumentParser();p.add_argument('--integrate',action='store_true');args=p.parse_args()
    DOCS.mkdir(parents=True,exist_ok=True)
    before=sha(OLD)
    data=load();configure_plots();page=build(data)
    if args.integrate:integrate(page,data)
    assert sha(OLD)==before
    manifest={'date':'2026-09-28','report':str(REPORT.relative_to(ROOT)),'seed':43,'K':4,
        'source_0918_unchanged_sha256':before,'report_sha256':sha(REPORT),
        'datasets':{ds:d['complete'] for ds,d in data.items()},
        'validation':{'completed_datasets':2,'oracle_unit_tests':4,'raw_f1_recomputed':True,'old_tab_preserved':True},
        'publishing':{'local_html':'complete','integrated':args.integrate,'git_push':'not performed',
                      'online_artifact':'not updated; no connected Claude artifact editor'}}
    (DOCS/'exp59_report_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(REPORT)


if __name__=='__main__':main()
