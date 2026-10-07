#!/usr/bin/env python3
"""Build a new 0928 report from audited single-seed residual-oracle results."""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

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
from render_exp59_capability import render as capability_tables

ROOT=repo_root(__file__)
RESULTS=ROOT/'tabpfn/results/v4/exp59/20260928_exp59_residual_oracle_s43'
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


def load(available=False):
    capability_path=RESULTS/'expert_capability/capability.json'
    capability=read_record(capability_path)
    assert capability['extra_inference'] is False and capability['extra_training'] is False
    data={}
    for ds in DATASETS:
        p=RESULTS/f'{ds}_s43'
        if not (p/'COMPLETE.json').exists() and available:
            p=p/'partial'
            if not (p/'COMPLETE.json').exists():continue
        complete=read_record(p/'COMPLETE.json')
        audit=read_record(p/'RECONSTRUCTION_COMPLETE.json')
        assert complete['seed']==43 and complete['K']==4
        if audit.get('evaluation_mode')=='fresh_single_seed_bank':
            assert audit['same_global_context_anchor_expert_pool_test_ids']
            assert audit['old_predictions_reused'] is False
        else:
            assert audit['observable_test_assignment_mismatches']==0
            assert all(audit[f'expert{k}_context_ids_exact'] for k in range(1,5))
        d={'path':p,'complete':complete,'audit':audit,'arms':complete['arms'],
           'manifest':read_record(p/'source_manifest.json'),
           'constant':read_record(p/'constant_reference.json'),
           'capability':capability['datasets'][ds]}
        assert d['capability']['rows']==complete['rows']
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
        for arm in d['arms']:
            row=d['summary'].loc[arm+'_residual_oracle']
            np.testing.assert_allclose(row.delta_oracle,row.macro_f1-max(d['summary'].loc['global','macro_f1'],d['constant']['macro_f1']),atol=1e-12)
        data[ds]=d
    if not data:raise RuntimeError('No completed experiment results to report')
    return data


def configure_plots():
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
        'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#cbd5e1','text.color':'#203044','svg.fonttype':'path','svg.hashsalt':'exp59-0928'})


def global_plot(data):
    fig,axes=plt.subplots(1,len(data),figsize=(12.5 if len(data)>1 else 8,4.7),layout='constrained')
    axes=np.atleast_1d(axes)
    for ax,(ds,d) in zip(axes,data.items()):
        q=d['global'];x=np.arange(len(q))
        ax.barh(x,q.f1,color=['#b85d4d' if z<.5 else '#7b9aaf' for z in q.f1],height=.65)
        for i,v in enumerate(q.f1):ax.text(v+.015,i,f(v),va='center',fontsize=9)
        ax.set(yticks=x,yticklabels=q.index,xlim=(0,1.16),xlabel='Global class F1',title=DATASETS[ds]+' · seed 43')
        ax.invert_yaxis();ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
    return save_figure(fig,'exp59_global_limits','Global의 클래스별 F1. 행 순서는 test 표본 수 내림차순이며, 모든 그림·표는 같은 seed 43 bank를 사용한다.')


def scope_text(data):
    return ' · '.join(DATASETS[ds]+' — '+('세 context 완료' if d['arms']==ARMS else '완료 조건 우선 반영') for ds,d in data.items())


def metric_cell(d,model,cl):
    q=d['pc'].loc[(model,cl)];g=d['pc'].loc[('global',cl)]
    values={k:float(q[k]) for k in ['f1','precision','recall']}
    baseline={k:float(g[k]) for k in values}
    values['ap']=float(d['probs'].loc[(model+'_raw',cl),'AP'])
    baseline['ap']=float(d['probs'].loc[('global_raw',cl),'AP'])
    delta=values['f1']-baseline['f1']
    attrs=' '.join(f'data-{k}="{v:.15g}" data-g-{k}="{baseline[k]:.15g}"' for k,v in values.items())
    color=f'rgba({"36,94,141" if delta>=0 else "184,93,77"},{min(.35,abs(delta)*.7):.3f})'
    return Markup(f'<span class="prob-cell" style="background:{color}" {attrs}>{f(values["f1"])}</span>')


def expert_composition(ds,d):
    source=Path(d['manifest']['source']);design=read_record(source/'design.json')
    anchor=design['anchor_rows'];rows=[]
    for _,r in d['context_compositions'].query('arm == "designed"').iterrows():
        counts=r[d['names']].astype(int).sort_values(ascending=False)
        main=', '.join(f'{esc(cl)} {100*num/r.block_rows:.1f}%' for cl,num in counts.head(3).items() if num)
        rows.append([f'e{int(r.expert)}',n(anchor),n(r.block_rows),n(anchor+r.block_rows),int((counts>0).sum()),Markup(main)])
    out=[f'<h3>{DATASETS[ds]}</h3>',table(['Expert','공통 anchor','특화 block','전체 context','Block 클래스 수','Block의 주요 클래스'],rows,f'exp59-{ds}-composition')]
    out.append('<details><summary>세 context의 클래스별 특화 block 행 수</summary>')
    rows=[]
    for _,r in d['context_compositions'].iterrows():
        if r.arm in d['arms']:rows.append([ARM_NAMES[r.arm],f'e{int(r.expert)}',n(r.block_rows),*[n(r[c]) for c in d['names']]])
    out.append(table(['조건','Expert','Block 수',*d['names']],rows,f'exp59-{ds}-context-composition'))
    out.append('</details>')
    return '\n'.join(out)


def full_test_tables(ds,d):
    out=[f'<h3>{DATASETS[ds]} — 전체 test</h3>',f'<label>Context <select class="arm-select" data-dataset="{ds}">']
    out.append(''.join(f'<option value="{a}">{ARM_NAMES[a]}</option>' for a in d['arms'])+'</select></label>')
    for arm in d['arms']:
        out.append(f'<div class="arm-panel" data-dataset="{ds}" data-arm="{arm}"'+(' hidden' if arm!='designed' else '')+'>')
        rows=[[cl,n(d['global'].loc[cl,'support']),metric_cell(d,'global',cl),*[metric_cell(d,f'{arm}_e{k}',cl) for k in range(1,5)]] for cl in d['names']]
        out.append(table(['클래스','Test 수','Global','e1','e2','e3','e4'],rows,f'exp59-{ds}-{arm}-quality'))
        out.append('</div>')
    return '\n'.join(out)


def region_class_tables(ds,d):
    focus='infiltration' if ds=='cic2018' else 'scanning'
    out=[f'<h3>{DATASETS[ds]} — 한 영역의 같은 샘플을 모든 expert가 분류</h3>',
         f'<label>클래스 <select class="region-class-select" data-dataset="{ds}">'+''.join(f'<option value="{esc(c)}"'+(' selected' if c==focus else '')+f'>{esc(c)}</option>' for c in d['names'])+'</select></label>']
    q=d['region_per_class'].query('arm == "designed"')
    composition=d['residual_region_composition'].query('split == "test"').set_index('region')
    for cl in d['names']:
        rows=[]
        for k in range(1,5):
            sub=q[(q.region==k)&(q['class']==cl)].set_index('model')
            if sub.empty:continue
            support=int(sub.iloc[0].support);models=['global','e1','e2','e3','e4']
            vals=[sub.loc[m,'f1'] for m in models];cells=[]
            for m,v in zip(models,vals):
                cell=str(metric(v,max(vals))) if support else '—'
                if m==f'e{k}':cell+='<small class="assigned">지정 expert</small>'
                cells.append(Markup(cell))
            rows.append([f'R{k}',n(composition.loc[k,'rows']),n(support),*cells,n(sub.loc[f'e{k}','FP'])])
        out.append(f'<div class="region-class-panel" data-dataset="{ds}" data-class="{esc(cl)}"'+(' hidden' if cl!=focus else '')+'>')
        out.append(table(['영역','전체 행','해당 클래스','Global F1','e1 F1','e2 F1','e3 F1','e4 F1','지정 expert FP'],rows,f'exp59-{ds}-{cl}-regions'))
        out.append('</div>')
    out.append('<details><summary>각 residual 소속의 실제 클래스 구성</summary>')
    rows=[]
    for k,r in composition.iterrows():
        desc=', '.join(f'{c} {n(r[c])}' for c in d['names'] if r[c]>0)
        rows.append([f'R{k} → e{k}',n(r.rows),int(r.classes_present),desc])
    out.append(table(['배정','행 수','클래스 수','클래스별 행 수'],rows,f'exp59-{ds}-region-composition')+'</details>')
    return '\n'.join(out)


def constant_diagnostic(data):
    rows=[];constants=0;total=0
    for ds,d in data.items():
        source=Path(d['manifest']['source']);design=read_record(source/'design.json')
        names=design['class_names'];regions=np.load(d['path']/'residual_test_regions.npy')
        for k in range(1,5):
            pred=np.load(source/f'predictions/designed_e{k}_raw.npy');counts=np.bincount(pred,minlength=len(names))
            classes=int((counts>0).sum());constants+=classes==1;total+=1
            local=np.bincount(pred[regions==k-1],minlength=len(names))
            dominant=f'{names[local.argmax()]} {100*local.max()/local.sum():.1f}%' if local.sum() else '해당 행 없음'
            own=d['capability']['experts'][k-1]['residual']
            cl=next(c for c in own['classes'] if c['name']==names[local.argmax()])
            truth=f'{100*cl["expert_metrics"]["support"]/own["rows"]:.1f}%'
            rows.append([DATASETS[ds],f'e{k}',classes,dominant,truth])
    text=f'<details><summary>보조 확인: 출력 집중은 실제 클래스 구성과 일치하는가?</summary><p>전체 test에서 한 클래스만 출력한 expert는 {total}개 중 {constants}개다. <b>이 사실만으로 분류 역량이 충분하다고 판단하지 않는다.</b> Anchor에 모든 클래스가 있어도 각 클래스의 식별력이 보장되지는 않는다. 가장 많이 출력한 클래스의 비율을 같은 담당영역의 실제 비율과 비교한다.</p>'
    text+=table(['데이터','Expert','전체 test 출력 클래스 수','담당영역의 최다 예측','그 클래스의 실제 비율'],rows,'exp59-constant-behavior')
    text+='<p>예를 들어 CIC2018 e3의 ddos 출력 집중은 실제 ddos 비중과 가깝다. 반면 e4는 실제 benign 비중보다 benign을 훨씬 많이 출력한다. ToN e4의 R4에는 ransomware만 있어, 영역 안의 높은 recall만으로 다른 클래스와의 구분 능력을 판단할 수 없다. 4장에서 음성이 포함된 입력 유사군을 함께 평가한다.</p>'
    text+='<p class="note">영역마다 클래스 하나만 출력하도록 만든 가상 비교 기준의 Macro-F1은 '+', '.join(DATASETS[ds]+' '+f(d['constant']['macro_f1']) for ds,d in data.items())+'이다. 이 비교 기준은 실제 학습한 expert가 아니다.</p></details>'
    return text


def context_tables(ds,d):
    focus='infiltration' if ds=='cic2018' else 'scanning'
    out=[f'<h3>{DATASETS[ds]} — 같은 expert의 context만 변경</h3>',
         f'<label>클래스 <select class="class-select" data-dataset="{ds}">'+''.join(f'<option value="{esc(c)}"'+(' selected' if c==focus else '')+f'>{esc(c)}</option>' for c in d['names'])+'</select></label>']
    for cl in d['names']:
        out.append(f'<div class="class-panel" data-dataset="{ds}" data-class="{esc(cl)}"'+(' hidden' if cl!=focus else '')+'>')
        g=d['global'].loc[cl];gap=d['probs'].loc[('global_raw',cl),'AP']
        out.append(f'<p class="note">전체 test의 같은 클래스 · Global F1 {f(g.f1)} / AP {f(gap)}</p>')
        rows=[]
        for k in range(1,5):
            f1=[d['pc'].loc[(f'{a}_e{k}',cl),'f1'] for a in d['arms']]
            ap=[d['probs'].loc[(f'{a}_e{k}_raw',cl),'AP'] for a in d['arms']]
            rows.append([f'e{k}',*[metric(v,max(f1)) for v in f1],*[metric(v,max(ap)) for v in ap]])
        short={'designed':'현재','matched_random':'비율 일치','balanced_random':'균형'}
        out.append(table(['Expert',*[short[a]+' F1' for a in d['arms']],*[short[a]+' AP' for a in d['arms']]],rows,f'exp59-{ds}-{cl}-context'))
        out.append('</div>')
    source=Path(d['manifest']['source']);same=[]
    for k in range(1,5):
        if 'matched_random' in d['arms'] and np.array_equal(np.sort(np.load(source/f'contexts/designed_e{k}.npy')),np.sort(np.load(source/f'contexts/matched_random_e{k}.npy'))):same.append(f'e{k}')
    if same:out.append('<p class="note">'+', '.join(same)+'의 현재 조건과 비율 일치 조건은 같은 ID 집합이다. 서로 다른 표본 선택의 효과로 세지 않는다.</p>')
    return '\n'.join(out)


def findings(data):
    rows=[]
    targets=[('cic2018','infiltration',1),('cic2018','web_attacks',1),('toniot','scanning',1)]
    for ds,cl,k in targets:
        if ds not in data:continue
        d=data[ds]
        rows.append([DATASETS[ds],cl,f'e{k}',f(d['global'].loc[cl,'f1']),
            *[f(d['pc'].loc[(f'{a}_e{k}',cl),'f1']) if a in d['arms'] else '계산 중' for a in ARMS]])
    return table(['데이터','클래스','Expert','Global F1','현재 F1','비율 일치 F1','균형 F1'],rows,'exp59-context-evidence')


def build(data):
    old=OLD.read_text();title=re.search(r'<title>(.*?)</title>',old,re.S).group(1);h1=re.search(r'<h1>(.*?)</h1>',old,re.S).group(1)
    css=re.search(r'<style>(.*?)</style>',old,re.S).group(1)+"""
    .hero-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:24px 0}.card{background:white;border:1px solid var(--line);border-radius:8px;padding:20px}.card h3{margin:0 0 8px}.value{font-size:24px;font-weight:700;color:var(--acc)}.card p{font-size:13px;margin:6px 0}.definition{background:#edf2f7;border-left:4px solid #245e8d;padding:14px 20px;margin:20px 0}.pill{display:inline-block;padding:2px 9px;background:#e8eff6;border-radius:12px;font-size:12px;margin-right:6px}nav{display:flex;flex-wrap:wrap;gap:8px 20px}select{font:inherit;max-width:100%;background:white;border:1px solid #bbcbd8;border-radius:5px;padding:6px 10px;margin:6px 10px}label{font-size:13px}.prob-cell{display:block;padding:5px;border-radius:3px;min-width:54px}.assigned{display:block;color:#245e8d;font-weight:700}td{font-variant-numeric:tabular-nums}[hidden]{display:none!important}footer{border-top:1px solid var(--line);margin-top:44px;padding-top:20px}a{color:var(--acc)}.wrap table{min-width:650px}.gain{color:#245e8d}.loss{color:#aa4b3c}.paired{white-space:nowrap}.count-note{display:block;color:#66788a;font-size:11px;margin-top:4px}.capability-panel{border:1px solid var(--line);border-radius:8px;padding:20px;margin:12px 0 30px;min-width:0}.capability-panel h4{font-size:16px;margin:24px 0 10px;color:#245e8d}.expert-verdict{margin-top:0}.interpretation{background:#edf2f7;padding:12px 16px;font-size:14px}.expert-picker{display:block;margin:18px 0 6px;font-weight:700}.capability-panel th{white-space:normal}.capability-panel .wrap{max-width:100%}@media(max-width:720px){.hero-grid{grid-template-columns:1fr}.capability-panel{padding:12px}.capability-panel table{font-size:12px}}
    """
    parts=[f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{css}</style></head><body><main>',
      f'<header><p class="eyebrow">0928 연구 보고서 · EXP59 · 화면 갱신 2026-09-29 · seed 43</p><h1>{h1}</h1><p class="lede">담당 residual의 오류를 고치는 expert는 있다. 다만 입력이 비슷한 다른 클래스까지 구분하는 능력은 부족한 경우가 있어, 혼동 클래스를 포함한 context와 선택 기준을 함께 개선해야 한다.</p><span class="pill">CIC2018 · ToN-IoT</span><span class="pill">K=4 고정</span><span class="pill">단일 seed preliminary</span></header>',
      f'<p class="note" id="report-scope">{esc(scope_text(data))}. 새 bank의 scorer/verifier는 이번 실험에서 학습하지 않았다.</p>',
      '<script type="application/json" id="exp59-report-coverage">'+json.dumps({'cells':sum(len(d['names'])*5*len(d['arms']) for d in data.values()),'panels':sum(len(d['arms']) for d in data.values()),'capability_experts':sum(len(d['capability']['experts']) for d in data.values()),'capability_class_rows':sum(len(d['names'])*8 for d in data.values()),'contexts':{ds:d['arms'] for ds,d in data.items()}})+'</script>',
      '<nav class="note"><a href="#global">1. Global 성능</a><a href="#oracle">2. 새 oracle</a><a href="#composition">3. Expert 구성</a><a href="#quality">4. Expert 분류 성능</a><a href="#context">5. Context 효과·개선</a></nav>',
      '<h2 id="global">1. Global의 클래스별 성능</h2>',global_plot(data)]
    for ds,d in data.items():
        rows=[[cl,n(r.support),f(r.precision),f(r.recall),f(r.f1)] for cl,r in d['global'].iterrows()]
        parts.append(f'<h3>{DATASETS[ds]} · Macro-F1 {f(d["summary"].loc["global","macro_f1"])}</h3>'+table(['클래스','Test 수','Precision','Recall','F1'],rows,f'exp59-{ds}-global'))
    parts+=['<h2 id="oracle">2. 새 oracle — 지정 expert의 예측을 그대로 채점</h2>',
      '<div class="definition"><p><b>Test 정답으로 residual을 계산한다 → 가장 가까운 학습 residual cluster를 찾는다 → 그 cluster의 담당 expert가 예측한 클래스를 그대로 채점한다.</b></p></div>',
      '<p>다른 expert가 맞혔다는 이유로 바꿔 고르거나 global로 복구하지 않는다. 학습 cluster와 residual 계산 규칙은 고정한다. 배정에 test 정답을 사용하므로 <b>소속을 아는 조건에서의 진단</b>이며 실제 모델 성능이나 모든 선택 방식의 상한은 아니다.</p>']
    rows=[]
    for ds,d in data.items():
        g=d['summary'].loc['global','macro_f1'];o=d['summary'].loc['designed_residual_oracle','macro_f1']
        rows.append([DATASETS[ds],n(d['complete']['rows']),f(g),f(o),signed(o-g)])
    parts.append(table(['데이터','Test 수','Global Macro-F1','현재 context Oracle Macro-F1','Global 대비 차이'],rows,'exp59-oracle-summary'))
    parts.append(constant_diagnostic(data))
    parts+=['<h2 id="composition">3. 각 expert는 어떤 context로 구성됐는가?</h2>',
      '<p>각 expert는 <b>공통 anchor + residual cluster에서 고른 특화 block</b>으로 구성한다. 표의 비율은 특화 block만의 구성이다. Anchor에는 모든 클래스가 들어 있지만, 그것만으로 혼동 클래스를 구별할 수 있다고 보장하지 않는다.</p>',
      '<p class="note">K=4는 기존 context 대조 실험과 맞춘 고정 조건이며 최적 K를 선택한 결과가 아니다. 클래스 수보다 K를 작게 두어도 특정 클래스에 block이 몰리는 현상은 남는다. Expert 번호는 데이터셋 안에서만 비교한다.</p>']
    for ds,d in data.items():parts.append(expert_composition(ds,d))
    parts.append('<p class="interpretation"><b>한 클래스만 있는 block을 어떻게 다룰 것인가?</b> ToN e4의 특화 block은 ransomware 537개뿐이다. 한 클래스 residual cluster 자체는 유지할 수 있지만, expert context는 공통 anchor에만 음성을 맡기지 않고 혼동되는 다른 클래스 사례를 명시적으로 포함하도록 바꾼다. 아래 분석은 현재 bank의 결과이며, 보강 context를 새로 실험한 결과는 아니다.</p>')
    parts+=['<h2 id="quality">4. Expert 분류 역량 — 교정 능력과 혼동 억제</h2>',
      '<p><b>저장된 예측만 재집계했다. 추가 학습·추론은 없다.</b> 현재 residual context의 모든 expert와 모든 클래스를 두 관점에서 평가한다. Expert는 학습·저장된 예측을 그대로 사용하며, expert나 클래스를 test 성능으로 골라 제외하지 않았다.</p>',
      '<div class="hero-grid" id="capability-scope"><div class="card"><h3>A. 담당 오류를 고치는가?</h3><p>기존 oracle과 같은 <b>정답 residual 영역 R1–R4</b>에서 각 담당 expert를 Global과 비교한다. 클래스별 F1과 정답 교정·훼손을 함께 본다.</p><p>정답을 사용한 소속 진단이다. 한 클래스뿐인 영역의 높은 정확도는 음성 구분의 근거가 아니다.</p></div><div class="card"><h3>B. 비슷한 다른 클래스도 구분하는가?</h3><p>이미 저장된 <b>입력 기반 최근접 배정 O1–O4</b>를 사용한다. 정답 없이 feature와 Global 예측확률이 같은 중심에 가장 가까운 test 샘플을 모아, 다른 클래스에 대한 FP와 원인을 확인한다.</p><p>학습 때 고정한 스케일·중심을 사용한다. 새로운 거리 임계값이나 validation 튜닝은 추가하지 않았다.</p></div></div>',
      '<p class="note">Rk와 Ok는 서로 다른 평가 집합이다. 두 집합의 F1 차이는 같은 샘플에서의 성능 하락이나 인과 효과로 읽지 않는다. 각 집합 안에서 Global과 expert를 비교한다. Ok는 하나의 입력 유사성 기준이며, 학습된 scorer/verifier의 배정 결과가 아니다.</p>',
      '<p><b>교정</b>은 Global 오답 → Expert 정답, <b>훼손</b>은 Global 정답 → Expert 오답이다. 총건수는 표본이 많은 클래스의 영향을 받으므로 expert 채택 기준이나 Macro-F1 변화로 바로 바꾸지 않는다. 클래스별 결과와 함께 읽는다.</p>']
    for ds,d in data.items():parts.append(capability_tables(ds,DATASETS[ds],d['capability']))
    parts.append('<p class="note">양성이 없는 클래스는 Recall·F1을 “—”로 표시하고 FP는 남긴다. 해당 클래스 예측이 없으면 Precision도 “—”다. Precision·Recall·F1은 0–1 값이며, 혼동 표의 비율은 %다. 혼동 쌍의 정렬은 현재 결과의 설명용이다. 다음 context의 표본 선택 규칙은 train에서 만들고 validation으로 고정한다.</p>')
    parts+=['<details id="full-test-reference"><summary>보조 자료 · 모든 test를 각 expert에 맡긴 클래스별 성능</summary>',
      '<p>전체 test의 FP를 포함한 스트레스 진단이다. 이 범위에서의 낮은 F1만으로 담당영역에서의 교정 능력이 없다고 판단하지 않는다.</p>',
      '<label>전체 test 지표 <select id="quality-metric"><option value="f1">F1</option><option value="precision">Precision</option><option value="recall">Recall</option><option value="ap">AP — 확률 순위</option></select></label>',
      '<p class="note">기본 화면은 F1이다. AP는 해당 클래스의 확률 순위를 보는 보조 지표로, 원래 클래스 예측의 F1과 구분한다. 파란 배경은 같은 클래스의 Global보다 높고, 주황은 낮다.</p>']
    for ds,d in data.items():parts.append(full_test_tables(ds,d))
    parts.append('<p>ToN e4의 전체 test ransomware recall은 0.9154 → 0.9958이지만 FP도 9,386 → 10,696으로 늘어 F1은 0.1208 → 0.1166이다. 반면 위 O4에서는 F1이 소폭 높아진다. 평가 범위를 명시해야 단일 block의 효과와 영역 밖 오인을 혼동하지 않는다.</p></details>')
    parts.append('<details id="region-reference"><summary>보조 자료 · 한 residual 영역에서 모든 expert를 가로로 비교</summary><p><b>행 R1–R4는 서로 다른 test 부분집합이고, 열 Global·e1–e4는 그 행의 동일한 샘플을 분류하는 모델이다.</b> 클래스 선택은 그 클래스의 F1을 보여주며 음성 샘플을 제거하지 않는다. 예를 들어 CIC2018 infiltration의 R1/e1 0.8893, R3/e3 0.9674, R4/e4 0.0000은 서로 다른 샘플의 결과라 expert 순위로 비교하지 않는다. 같은 R3 행 안의 열을 비교해야 한다.</p>')
    for ds,d in data.items():parts.append(region_class_tables(ds,d))
    parts.append('<p class="note">위 영역별 표는 현재 residual context 기준이다. 표적 클래스가 없는 영역은 F1을 “—”로 표시하고 FP는 남긴다. Test 정답을 사용한 영역 진단이므로 정답 없는 라우팅의 성능으로 해석하지 않는다.</p></details>')
    parts+=['<h2 id="context">5. Context 효과는 관찰된다 — 다음은 구성 기준의 개선</h2>',
      '<p>Global·anchor·K·expert별 context 크기와 평가 행을 고정했다. <b>비율 일치 무작위</b>는 특화 block의 클래스별 행 수를 유지하고 표본 선택을 바꾼다. <b>균형 무작위</b>는 총행 수를 유지하며 클래스 구성을 바꾼다. 서로 다른 expert 번호 사이에는 context 크기도 다르므로 context 효과는 같은 번호 안에서 비교한다.</p>',findings(data),
      '<p>CIC2018 e1의 infiltration은 클래스별 행 수가 같아도 현재 선택보다 무작위 선택의 F1이 높다. ToN e1의 scanning은 세 context 모두 Global을 개선하고, 현재 residual 선택이 가장 높지는 않다. <b>Context를 바꾸면 성능이 달라진다는 근거는 있으나, 현재 residual 선택 규칙이 우월하다는 근거는 일관되지 않다.</b> 단일 seed 관찰이므로 반복 안정성과 원인은 다음 대조에서 확인한다.</p>']
    for ds,d in data.items():parts.append(context_tables(ds,d))
    parts.append('<details><summary>같은 residual 배정에서 세 context의 Oracle Macro-F1</summary>')
    rows=[]
    for ds,d in data.items():rows.append([DATASETS[ds],f(d['summary'].loc['global','macro_f1']),*[f(d['summary'].loc[a+'_residual_oracle','macro_f1']) if a in d['arms'] else '계산 중' for a in ARMS]])
    parts.append(table(['데이터','Global','현재','비율 일치','균형'],rows,'exp59-oracle-contexts')+'</details>')
    parts+=['<h3>모델을 어떻게 개선할 것인가?</h3>',
      '<ol><li><b>한 클래스 위주 block에도 혼동 음성을 명시적으로 포함한다.</b> 우선 기존 context와 총크기를 맞춘 보강 조건을 비교한다. ToN e4는 ransomware와 혼동되는 benign, CIC2018 e3는 infiltration으로 오인되는 benign, ToN e3는 mitm으로 오인되는 benign과 injection·dos로 오인되는 xss를 점검한다. 현재 test 혼동은 개선 가설이며 표본 선택에는 train만 사용하고 validation으로 규칙을 고정한다. 원인 분리를 위해 같은 양성·음성 수를 유지한 무작위 음성 대조도 둔다. 4장의 교정·훼손과 클래스별 F1로 판정하고, 개선되지 않는 expert는 재구성하거나 호출 대상에서 제외한다.</li>',
      '<li><b>개선한 bank에 scorer/verifier를 다시 맞춘다.</b> 이번 K=4 bank에는 두 모듈을 아직 학습하지 않았다. 기존 교정·훼손 표적의 학습을 같은 bank에서 수행하고, 검증 자료로 선택 규칙을 고정한다. 목표 클래스 recall뿐 아니라 다른 클래스 FP와 기존 정답 훼손이 줄어드는지 확인한다.</li>',
      '<li><b>같은 bank에서 구성 요소의 기여를 비교한다.</b> Global, scorer/verifier 모두 제거, scorer만 유지, verifier만 유지, 둘 다 유지 조건을 맞춘다. Scorer를 제거한 경우에도 정답을 쓰지 않는 expert 선택 규칙을 고정한다. 같은 정보 예산의 단일 context 모델·무작위 context bank도 대조하여 Macro-F1·클래스별 F1·교정/훼손·호출 비용을 비교한다.</li></ol>',
      '<p class="note">현재 K=4 bank의 입력 기반 최근접 배정 Macro-F1은 CIC2018 0.7613, ToN 0.7191이다. 정답 residual oracle과는 다르다. 개선의 판정은 oracle 상승이 아니라 정답 없는 실제 모델의 성능으로 한다. 예시 최고값을 보고 test에서 운영 context를 선택하지 않는다.</p>',
      '<footer><p class="note">0918과 같은 제목의 0928 탭 · EXP59 · seed 43 · 2026-09-29 expert별 두 관점 분석 추가. 추가 학습·추론 없이 저장된 예측과 배정만 재집계했다. 기존 0918 본문은 유지했다.</p></footer>',
      """<script>
      document.getElementById('quality-metric').addEventListener('change',function(){const m=this.value;document.querySelectorAll('.prob-cell').forEach(c=>{const v=Number(c.dataset[m]),g=Number(c.dataset['g'+m[0].toUpperCase()+m.slice(1)]),d=v-g;c.textContent=v.toFixed(4);c.style.background='rgba('+(d>=0?'36,94,141':'184,93,77')+','+Math.min(.35,Math.abs(d)*.7)+')';});});
      for(const [selector,panel,key] of [['.arm-select','.arm-panel','arm'],['.class-select','.class-panel','class'],['.region-class-select','.region-class-panel','class'],['.capability-select','.capability-panel','expert']]){
        document.querySelectorAll(selector).forEach(s=>s.addEventListener('change',()=>document.querySelectorAll(panel).forEach(p=>{if(p.dataset.dataset===s.dataset.dataset)p.hidden=p.dataset[key]!==s.value;})));
      }
      </script></main></body></html>"""]
    page=apply_typography('\n'.join(parts));REPORT.write_text(page);return page


def integrate(page,data):
    s=INDEX.read_text();m=re.search(r'const REPORTS = (.*?);\n',s)
    reports=json.loads(m.group(1));old=next(r for r in reports if r['id']=='scorer_verifier_target_0918')
    reports=[r for r in reports if r['id']!=REPORT_ID]
    complete=len(data)==len(DATASETS) and all(d['arms']==ARMS for d in data.values())
    new={**old,'id':REPORT_ID,'date':'2026-09-28','titleKr':old['titleKr'],
         'titleEn':'Residual-membership oracle and context evaluation · seed 43',
         'verdict':('09/29 갱신 · expert별 교정·혼동 억제 분석 · 추가 추론 없음' if complete else '09/28 · 완료 조건 우선 반영 · 나머지 계산 중'),
         'verdictClass':'warn','b64':base64.b64encode(page.encode()).decode()}
    reports.insert(1,new)
    s=s[:m.start(1)]+json.dumps(reports,ensure_ascii=False)+s[m.end(1):]
    flow='<div class="flow-step" id="exp59-overview"><time>09-28</time><div class="flow-rail-dotcol"><div class="d"></div><div class="l"></div></div><div class="body"><button class="jump" data-jump="'+REPORT_ID+'"><h3>EXP59 — Expert별 교정 능력과 혼동 억제</h3></button><p>09-29 추가 추론 없이 재집계. 정답 residual 영역에서의 클래스별 교정·훼손과, 입력 기반 최근접 배정에 포함된 다른 클래스의 FP를 함께 확인한다. 8개 expert 각각의 결과·해석을 수록하고 혼동 클래스 보강 context로 연결한다.</p></div></div>'
    flow=flow.replace('</p></div></div>','<br>'+esc(scope_text(data))+'</p></div></div>')
    s=re.sub(r'<!-- EXP59_FLOW_START -->.*?<!-- EXP59_FLOW_END -->\n?','',s,flags=re.S)
    anchor='<h2>연구 흐름 (최신순)</h2>'
    s=s.replace(anchor,anchor+'\n<!-- EXP59_FLOW_START -->'+flow+'<!-- EXP59_FLOW_END -->',1)
    finding='<li><p><strong>(09-28 EXP59 · 09-29 개편) Context가 성능을 바꾸는 근거와 다음 개선.</strong> '
    finding+=' · '.join(DATASETS[ds]+' Global '+f(d['summary'].loc['global','macro_f1'])+' → 잔차 소속 oracle '+f(d['summary'].loc['designed_residual_oracle','macro_f1']) for ds,d in data.items())
    finding+='. Oracle은 정답 소속 조건의 진단이다. 담당 오류를 고치는 expert라도 입력이 비슷한 benign·다른 공격을 오인할 수 있다. Expert별 교정·훼손과 혼동 쌍을 확인하여, 혼동 음성을 포함한 context와 정답 없는 선택 성능을 함께 개선한다. '+esc(scope_text(data))+'.</p></li>'
    s=re.sub(r'<!-- EXP59_FINDING_START -->.*?<!-- EXP59_FINDING_END -->\n?','',s,flags=re.S)
    s=s.replace('<ol class="findings">','<ol class="findings">\n<!-- EXP59_FINDING_START -->'+finding+'<!-- EXP59_FINDING_END -->',1)
    count=sum(r.get('group')=='report' for r in reports)
    s=re.sub(r'(<p class="asof">2026-08-18 &ndash; )\d{4}-\d{2}-\d{2}', r'\g<1>2026-09-29', s, count=1)
    s=s.replace('8&ndash;9월의 보고서 아홉 편을 최신순으로 묶었습니다.',f'8&ndash;9월의 보고서 {count}편을 최신순으로 묶었습니다.')
    INDEX.write_text(s)


def main():
    p=argparse.ArgumentParser();p.add_argument('--integrate',action='store_true');p.add_argument('--available',action='store_true');args=p.parse_args()
    DOCS.mkdir(parents=True,exist_ok=True)
    before=sha(OLD)
    data=load(available=args.available);configure_plots();page=build(data)
    if args.integrate:integrate(page,data)
    assert sha(OLD)==before
    manifest={'date':'2026-09-28','revised':'2026-09-29','report':str(REPORT.relative_to(ROOT)),'seed':43,'K':4,
        'source_0918_unchanged_sha256':before,'report_sha256':sha(REPORT),
        'datasets':{ds:d['complete'] for ds,d in data.items()},
        'scope':scope_text(data),
        'expert_capability':{'artifact':str((RESULTS/'expert_capability/capability.json').relative_to(ROOT)),
            'sha256':sha(RESULTS/'expert_capability/capability.json'),'additional_inference':False,
            'views':['label-assisted residual membership','saved input-only nearest centroid membership'],
            'experts':sum(len(d['capability']['experts']) for d in data.values())},
        'validation':{'completed_datasets':sum(d['arms']==ARMS for d in data.values()),'included_contexts':{ds:d['arms'] for ds,d in data.items()},'oracle_unit_tests':4,'raw_f1_recomputed':True,'old_tab_preserved':True},
        'publishing':{'local_html':'complete','integrated':args.integrate,'git_push':'not performed',
                      'online_artifact':'not updated; no connected Claude artifact editor'}}
    (DOCS/'exp59_report_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(REPORT)


if __name__=='__main__':main()
