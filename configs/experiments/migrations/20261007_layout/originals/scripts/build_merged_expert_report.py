#!/usr/bin/env python3
"""Merge the three post-audit reports and publish verified EXP62 local results."""
import base64
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from build_exp56_report_section import Markup, table
from render_exp63_capability import load_sweep, render as render_sweep, CSS as SWEEP_CSS, JS as SWEEP_JS
from report_order import support_order, permute, apply_rules

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / 'lablog/html_report'
DOC = ROOT / 'docs/research/20260930'
RUN = ROOT / 'tabpfn/results/20260930_exp62_sv_current_bank_s43'
DIAG = ROOT / 'tabpfn/results/20260928_exp59_residual_oracle_s43'
ID = 'scorer_verifier_target_0928'
OLD_IDS = ['clean_revalidation_0916', 'scorer_verifier_target_0918']
TITLE = '정제 데이터에서의 Expert 역량과 S/V 검증'
NAMES = {'cic2018': 'CIC2018', 'toniot': 'ToN-IoT'}
ARMS = {'global': 'Global', 'fixed_region': '입력 기반 고정 배정', 's1v0': 'S + V'}
CLASSES = {'benign':'Benign','bot':'Bot','brute_force':'Brute force','ddos':'DDoS','dos':'DoS',
           'infiltration':'Infiltration','web_attacks':'Web attack','backdoor':'Backdoor',
           'injection':'Injection','mitm':'MitM','password':'Password','ransomware':'Ransomware',
           'scanning':'Scanning','xss':'XSS'}

def read(path): return json.loads(Path(path).read_text())
def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def f(v): return f'{float(v):.4f}'
def n(v): return f'{int(v):,}'
def pct(v): return f'{100*float(v):.2f}%'
def delta(v): return f'{float(v):+.4f}'
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def records(df): return json.loads(df.to_json(orient='records',double_precision=15))

def archive():
    dest=DOC/'archive/pre_merge_reports';dest.mkdir(parents=True,exist_ok=True)
    for name in [*OLD_IDS, ID, 'post_tabpfn']:
        src=HTML/(name+'.html');target=dest/src.name
        if not target.exists():shutil.copy2(src,target)
    return dest

def verify_metrics(ds, summaries, classes, names):
    """Recompute the report metrics from full-size labels and saved predictions."""
    p=RUN/ds;cache=p/'cache';C=len(names);y=np.load(cache/'eval_y.npy');g=np.load(cache/'eval_p0.npy',mmap_mode='r').argmax(1)
    predictions={'global':g,'fixed_region':np.load(p/'policy/fixed_region_final.npy'),'s1v0':np.load(p/'policy/s1v0_final.npy')}
    checks=[]
    for arm,pred in predictions.items():
        assert len(y)==len(pred)
        cm=np.bincount(y.astype('int64')*C+pred,minlength=C*C).reshape(C,C)
        support=cm.sum(1);predicted=cm.sum(0);tp=cm.diagonal()
        precision=np.divide(tp,predicted,out=np.zeros(C),where=predicted>0)
        recall=np.divide(tp,support,out=np.zeros(C),where=support>0)
        f1=np.divide(2*tp,support+predicted,out=np.zeros(C),where=support+predicted>0)
        s=summaries[arm]
        np.testing.assert_allclose([f1.mean(),tp.sum()/len(y)],[s['macro_f1'],s['accuracy']],rtol=0,atol=1e-12)
        assert int(((g!=y)&(pred==y)).sum())==s['helpful']
        assert int(((g==y)&(pred!=y)).sum())==s['harmful']
        for i,cl in enumerate(names):
            row=classes[arm][i];assert row['class']==cl and row['support']==int(support[i])
            np.testing.assert_allclose([precision[i],recall[i],f1[i]],[row['precision'],row['recall'],row['system_f1']],rtol=0,atol=1e-12)
        checks.append(dict(dataset=ds,arm=arm,rows=len(y),classes=C,metrics_and_corrections_match=True))
    from nfv3_v3_exp51_scorer_target import apply_thresholds
    with np.load(p/'policy/eval_s1v0_scores.npz') as z:
        final,called,accepted=apply_thresholds(g,z['candidate'],z['pre'],z['post'],read(p/'policy/selected_thresholds.json'))
    np.testing.assert_array_equal(final,predictions['s1v0'])
    assert int(called.sum())==summaries['s1v0']['proposed'] and int(accepted.sum())==summaries['s1v0']['accepted']
    return checks

def load():
    import sys
    sys.path.insert(0,str(ROOT/'tabpfn/scripts'))
    data={'seed':43,'K':4,'datasets':{},'capability_sweep':load_sweep(),
          'class_names':CLASSES,'arms':ARMS,'current_source':'EXP62 · 09/30 · 동일 context 재학습',
          'diagnostic_source':'EXP59 · 09/28 · 저장된 expert 진단'}
    checks=[]
    for ds in NAMES:
        p=RUN/ds;assert (p/'COMPLETE.json').exists()
        meta=read(p/'cache/COMPLETE.json');names=meta['class_names']
        frame=pd.read_csv(p/'summary.csv');c=pd.read_csv(p/'class_metrics.csv')
        summaries={r['arm']:r for r in records(frame[frame.split=='full_test'])}
        classes={a:records(c[(c.split=='full_test')&(c.arm==a)]) for a in ARMS}
        checks+=verify_metrics(ds,summaries,classes,names)
        order=support_order([r['support'] for r in classes['global']])   # report order rule: test count desc
        names=permute(names,order);classes={a:permute(v,order) for a,v in classes.items()}
        grid=pd.read_csv(p/'policy/threshold_grid.csv');candidates=grid[~grid.reason.fillna('').str.startswith('selected:')]
        selected=read(p/'policy/selected_thresholds.json')
        ties=candidates[candidates.feasible & ((candidates.delta_macro_f1-selected['delta_macro_f1']).abs()<1e-12)]
        old=pd.read_csv(DIAG/f'{ds}_s43/summary.csv').set_index('model')
        compositions=pd.read_csv(DIAG/f'{ds}_s43/fresh_bank/context_compositions.csv').query('arm == "designed"')
        design=read(DIAG/f'{ds}_s43/fresh_bank/design.json')
        hist=pd.read_csv(ROOT/f'tabpfn/results/exp56_label_free_20260922/{ds}/1a_summary.csv')
        data['datasets'][ds]=dict(title=NAMES[ds],class_order=names,summaries=summaries,classes=classes,
            oracle=float(old.loc['designed_residual_oracle','macro_f1']),diagnostic_global=float(old.loc['global','macro_f1']),
            compositions=records(compositions),anchor_rows=design['anchor_rows'],
            historical=records(hist[hist.policy.isin(['global','scorer_verifier'])]),
            thresholds=selected,
            grid=dict(candidates=len(candidates),feasible=int(candidates.feasible.sum()),
                      positive_macro=int((candidates.delta_macro_f1>0).sum()),
                      equal_gain_min_weighted_proposal=float(ties.proposal_rate.min()) if len(ties) else None),
            completed_kst=datetime.fromtimestamp(read(p/'progress.json')['updated_epoch'],timezone(timedelta(hours=9))).strftime('%H:%M:%S'),
            validation_confirmation=records(frame[frame.split=='cal_confirm'])[0])
    write(DOC/'exp62_numeric_validation.json',dict(checks=checks,threshold_application_verified=True))
    write(DOC/'merged_expert_report_data.json',data)
    return data

CSS='''
:root{--ink:#203044;--muted:#5e7082;--line:#dce4eb;--blue:#255e88;--paper:#fff;--bg:#f6f8fa;--gain:#236c68;--loss:#a24b40}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:18px}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.8 "Noto Sans CJK KR","Pretendard",sans-serif}main{max-width:1180px;margin:auto;padding:42px 32px 70px}header{border-bottom:1px solid var(--line);padding-bottom:24px}.eyebrow{font-size:12px;letter-spacing:.04em;color:var(--blue);margin:0}h1{font-size:32px;line-height:1.45;margin:10px 0 14px}h2{font-size:24px;line-height:1.5;margin:44px 0 16px}h3{font-size:18px;margin:24px 0 10px}h4{font-size:16px;margin:20px 0 8px}p{margin:10px 0}p,li,h1,h2,h3,h4{word-break:keep-all;overflow-wrap:anywhere}.lede{font-size:17px;max-width:960px}.note,small{font-size:12px;color:var(--muted)}.pill{display:inline-block;margin:6px 8px 0 0;padding:2px 10px;border:1px solid var(--line);border-radius:20px;font-size:12px;color:var(--muted);background:white}nav{display:flex;flex-wrap:wrap;gap:8px 20px;margin:22px 0}a,.link-button{color:var(--blue)}.link-button{font:inherit;border:0;background:none;text-decoration:underline;padding:0;cursor:pointer}a:focus-visible,button:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #6a9bbe;outline-offset:3px}.hero-grid,.definitions{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:22px 0}.card{background:white;border:1px solid var(--line);border-radius:9px;padding:22px;min-width:0}.card h3{margin:0 0 8px}.value{font-size:31px;line-height:1.3;font-weight:700;color:var(--blue)}.card .sub{font-size:13px;color:var(--muted)}.card p{font-size:14px}.callout,.interpretation{padding:15px 18px;background:#edf3f7;border-left:3px solid #83a4bb;margin:18px 0}.source-label{font-size:12px;color:var(--muted);margin:0 0 14px}.wrap{max-width:100%;overflow-x:auto;border:1px solid var(--line);border-radius:6px;margin:14px 0 20px;background:white}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}th{background:#edf1f5;color:#44576b;font-size:12px;white-space:nowrap}tr:last-child td{border-bottom:0}.wrap table{min-width:640px}.current-table td{white-space:nowrap}.gain{color:var(--gain)}.loss{color:var(--loss)}.paired{white-space:nowrap}.count-note{display:block;color:var(--muted);font-size:11px;margin-top:4px}details{border-top:1px solid var(--line);margin:20px 0;padding-top:14px}summary{cursor:pointer;font-weight:700;color:var(--blue)}.detail-body{padding:6px 0 14px}.metric-picker,.expert-picker{display:block;font-size:13px;font-weight:700;margin:18px 0 8px}select{font:inherit;padding:6px 10px;background:white;border:1px solid #b7c7d4;border-radius:5px;max-width:100%;margin:0 8px}.capability-panel{border:1px solid var(--line);border-radius:8px;padding:18px;margin:14px 0;min-width:0}.capability-panel th{white-space:normal}.capability-panel h4{color:var(--blue)}.expert-verdict{margin-top:0}.definitions .card{padding:18px}.definitions .card p{font-size:14px}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}.flow .step{border:1px solid var(--line);background:white;border-radius:6px;padding:14px;min-width:0}.step .number{display:block;font-size:21px;font-weight:700;color:var(--blue)}.step .label{font-size:12px;color:var(--muted)}.plan{padding-left:22px}.plan li{margin:12px 0}.history{margin-top:38px}footer{border-top:1px solid var(--line);margin-top:34px;padding-top:16px;font-size:12px;color:var(--muted)}code{font-size:11px;overflow-wrap:anywhere}[hidden]{display:none!important}.live-caption{margin-top:6px;font-size:12px;color:var(--muted)}
@media(max-width:760px){main{padding:26px 16px 50px}h1{font-size:26px}h2{font-size:21px}.lede{font-size:15px}.hero-grid,.definitions{grid-template-columns:1fr}.card{padding:17px}.capability-panel{padding:12px}.flow{grid-template-columns:1fr 1fr}th,td{padding:9px 10px}table{font-size:12px}.wrap table{min-width:620px}nav{font-size:13px}.value{font-size:28px}}
'''

def class_tables(data):
    out=['<details id="class-detail" open><summary>Global의 클래스별 성능과 S/V 적용 후 변화</summary><div class="detail-body">',
         '<label class="metric-picker">표시 지표 <select id="current-metric"><option value="system_f1">F1</option><option value="precision">Precision</option><option value="recall">Recall</option></select></label>']
    for ds,d in data['datasets'].items():
        rows=[]
        for i,cl in enumerate(d['class_order']):
            cells=[CLASSES[cl],n(d['classes']['global'][i]['support'])]
            for arm in ARMS:
                r=d['classes'][arm][i]
                cells.append(Markup('<span class="current-metric" '+ ' '.join(f'data-{key}="{r[key]:.15g}"' for key in ['system_f1','precision','recall'])+'>'+f(r['system_f1'])+'</span>'))
            change=d['classes']['s1v0'][i]['delta_f1']
            cells.append(Markup(f'<span class="{"gain" if change>=0 else "loss"}">{delta(change)}</span>'))
            rows.append(cells)
        out.append('<h3>'+d['title']+'</h3>'+table(['클래스','Test 수',*ARMS.values(),'ΔF1 · S+V − Global'],rows,f'current-{ds}-classes'))
    out+=['<p class="note">마지막 열은 선택한 표시 지표와 무관하게 F1 변화다. Global의 CIC2018 Infiltration·Web attack, ToN의 Scanning·MitM·Ransomware가 주요 개선 대상이다.</p></div></details>']
    return ''.join(out)

def build(data):
    d=data['datasets'];cic=d['cic2018'];ton=d['toniot']
    out=[f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title><style>{CSS}{SWEEP_CSS}</style></head><body><main>',
        '<header><p class="eyebrow">09/17 · 09/18 · 09/28 통합 보고서 · 2026-09-30 갱신</p>',f'<h1>{TITLE}</h1>',
        '<p class="lede">Expert의 교정 능력을 실제 분류 이득으로 연결할 수 있는가? CIC2018은 Global을 유지했고, ToN은 Scanning을 개선했다. 다음은 CIC2018의 유효한 채택 후보 확보와 ToN의 호출 비용·Benign 오탐 개선이다.</p>',
        '<span class="pill">모순 행 제거 데이터</span><span class="pill">CIC2018 · ToN-IoT</span><span class="pill">seed 43 · 3장 K=2·4·6·8</span><span class="pill">단일 seed preliminary</span></header>',
        '<nav aria-label="보고서 목차"><a href="#current">1. K=4 성능</a><a href="#oracle">2. 정정한 oracle</a><a href="#capability">3. Expert 분류 역량 · K 비교</a><a href="#sv">4. K=4 S/V와 다음 개선</a></nav>',
        '<section id="current"><h2>1. K=4 기준 성능과 개선 대상</h2><p class="source-label">09/30 EXP62 · K=4 기준 결과 · K=2·4·6·8 후속 비교와 expert 직접 평가는 3장에 제시</p><div class="hero-grid">']
    for ds,label,text in [('cic2018','Global 유지 · 호출 0건','현재 후보·임계값 조합에서는 유효한 개입을 선택하지 못했다. Infiltration·Web attack을 개선할 후보와 채택 기준이 우선이다.'),('toniot','Global 대비 +0.0821 · 호출률 100%','Scanning F1 0.0394 → 0.8587. 표본당 expert 1개를 호출하며, MitM·Ransomware와 Benign 오탐은 추가 개선 대상이다.')]:
        a=d[ds]['summaries']['s1v0'];out.append(f'<article class="card"><h3>{d[ds]["title"]}</h3><div class="value">{f(a["macro_f1"])} <small>Macro-F1</small></div><p class="sub">{label}</p><p>{text}</p><p class="live-caption">{d[ds]["completed_kst"]} 완료 · test {n(a["rows"])}행</p></article>')
    out+=['</div>']
    rows=[]
    for ds,a in d.items():
        for arm,label in ARMS.items():
            s=a['summaries'][arm];rows.append([a['title'],label,f(s['macro_f1']),pct(s['accuracy']),delta(s['delta_macro_f1']),n(s['helpful']),n(s['harmful'])])
    out.append(table(['데이터','방법','Macro-F1','Accuracy','ΔMacro-F1','교정','훼손'],rows,'current-summary'))
    out.append('<p class="note">고정 배정은 입력 표현·Global 확률로 가장 가까운 cluster의 expert를 선택한다. S+V는 S가 고른 후보를 V가 채택할 때만 Global 출력을 바꾼다. 교정은 Global 오답 → 정답, 훼손은 Global 정답 → 오답이다.</p>')
    out.append('<p class="callout"><b>SOTA와의 현재 위치.</b> 같은 C0 100k 비교군 최고는 CIC2018 DistPFN <b>0.7839</b>, ToN XGBoost <b>0.6983</b>다. 전체 train XGBoost는 각각 <b>0.7795 / 0.7162</b>다. 제안 모델은 expert·route에 추가 학습 데이터를 사용하므로 동일 학습 예산의 우위로 해석하지 않는다. <button class="link-button" data-report-jump="sota_pfn_comparison">0908 SOTA 클래스별 성능·비용 보기</button></p>')
    out.append(class_tables(data));out.append('</section>')

    out+=['<section id="oracle"><h2>2. 정정한 oracle: 담당 expert의 예측을 그대로 평가</h2>',
        '<p class="source-label">2장은 09/28에 저장한 K=4 expert의 진단(EXP59)이다. 3장은 09/30 EXP62·EXP63의 fitted bank를 평가한다. 아래 oracle 수치를 최신 정책·직접 평가와 같은 fitted bank의 결과로 혼합하지 않는다.</p>',
        '<div class="callout"><b>Test 정답을 포함해 residual을 계산한다 → 가장 가까운 학습 residual cluster를 찾는다 → 그 expert가 예측한 클래스를 그대로 채점한다.</b></div>',
        '<p>기존의 “여러 expert 중 맞힌 예측 선택, 없으면 Global 유지”를 바꿨다. 이 값은 <b>정답을 사용한 residual 배정의 조건부 진단</b>이다. 실제 추론 성능이나 가능한 모든 정책의 절대 상한은 아니다.</p>']
    out.append(table(['진단 데이터','Global','Residual 배정 oracle','변화'],[[a['title'],f(a['diagnostic_global']),f(a['oracle']),delta(a['oracle']-a['diagnostic_global'])] for a in d.values()],'diagnostic-oracle'))
    out.append('<p class="note">클래스 하나에 expert 하나를 대응시키지 않는다. 한 클래스도 여러 residual cluster에 분포할 수 있다. Expert 자체의 분류 역량은 다음 장에서 배정 표본을 바꾸지 않는 직접 평가로 확인한다.</p></section>')
    out.append(render_sweep(data['capability_sweep'],CLASSES))

    out+=['<section id="sv"><h2>4. K=4 S/V는 교정 이득을 보존하는가?</h2><p class="source-label">09/30 완료 · EXP62 · K=4 기준 결과 · Direct decision-gain scorer + NLL quantile verifier · 정책과 임계값은 validation에서 선택</p>']
    rows=[]
    for ds,a in d.items():
        s=a['summaries']['s1v0'];rows.append([a['title'],n(s['proposed']),pct(s['proposed']/s['rows']),n(s['accepted']),pct(s['accepted']/s['rows']),n(s['changed']),n(s['helpful']),n(s['harmful'])])
    out.append(table(['데이터','정책상 호출','호출률','V 채택','채택 / 전체','출력 변경','교정','훼손'],rows,'sv-activation'))
    out.append('<p class="note">호출·채택은 저장된 예측에 정책을 적용해 집계한 논리적 건수다. 준비 단계에서는 모든 expert의 예측을 계산했으므로 이 실행시간을 희소 호출의 온라인 추론 비용으로 사용하지 않는다. 채택해도 Global과 같은 클래스를 예측할 수 있어 채택 수와 출력 변경 수는 다르다.</p>')
    out+=['<div class="hero-grid"><article class="card"><h3>CIC2018 · 유효한 개입 후보 확보</h3>',
        f'<p>Validation에서 탐색한 <b>{cic["grid"]["candidates"]}개 개입 정책 모두</b> Macro-F1이 감소하고 Benign 오탐·tail F1 조건을 위반했다. 최종 정책은 Global 유지다.</p>',
        '<p>현재 후보·S/V 점수가 교정과 훼손을 구분하는지 validation에서 먼저 점검하고, Infiltration·Web attack의 혼동을 줄이는 expert와 채택 기준을 개선한다.</p></article>',
        '<article class="card"><h3>ToN · 성능 개선, 호출과 오탐 개선 필요</h3><p>S가 expert를 고르고 V가 일부를 채택한다. <b>호출률은 100%</b>이므로 호출을 생략하는 기능은 아직 없다. 개선은 Scanning 중심이며 Benign 정답 1,101건을 훼손했다.</p><p>Validation 성능이 같은 후보 중 호출이 적은 정책도 있지만, 현재 선택기는 동점에서 먼저 본 정책을 유지한다. 다음에는 호출 비용을 동점 선택 기준으로 반영하고 Benign·tail 성능을 확인한다.</p></article></div>',
        '<details><summary>S/V의 클래스별 교정·훼손과 validation 확인 결과</summary>']
    for ds,a in d.items():
        rows=[[CLASSES[r['class']],n(r['support']),f(r['global_f1']),f(r['system_f1']),n(r['helpful']),n(r['harmful'])] for r in a['classes']['s1v0']]
        out.append('<h3>'+a['title']+'</h3>'+table(['클래스','Test 수','Global F1','S+V F1','교정','훼손'],rows,f'sv-{ds}-changes'))
    out.append(f'<p>ToN은 현재 정책과 같은 validation Macro-F1을 내면서 가중 호출률이 <b>{pct(ton["grid"]["equal_gain_min_weighted_proposal"])}</b>인 후보도 있었다. 이는 validation의 후보 비교이며, 해당 정책의 test 호출률·성능을 뜻하지 않는다.</p>')
    out.append('<p>ToN은 임계값 선택 구간에서 tail 평균 F1을 유지했지만, 별도 validation 확인 구간에서는 변화가 −0.0501이었다. Test의 MitM·Ransomware 평균 F1 변화도 −0.0006이다. 전체 Macro-F1 개선과 별개로 tail 성능의 구간별 유지가 다음 선택 기준에 필요하다.</p></details>')
    out+=['<h3>다음 개선과 실험 순서</h3><ol class="plan">',
        '<li><b>CIC2018:</b> validation의 후보별 교정·훼손을 확인한다. 교정 가능한 후보가 있는데 선택·채택이 놓치면 S/V를, 후보 자체의 손해가 크면 해당 expert의 context와 사용 대상을 개선한다.</li>',
        '<li><b>ToN:</b> 검증 성능이 같은 정책 사이에서 낮은 호출률을 우선하고, Benign 훼손과 tail 성능을 확인 구간에서도 점검한다. Test 결과로 임계값을 조정하지 않는다.</li>',
        '<li><b>최종 후보 이후:</b> 같은 bank에서 S/V 제거 비교를 수행해 각 요소의 기여를 검증하고, 학습 데이터 예산과 실제 추론 비용을 맞춰 SOTA와 비교한다.</li></ol></section>']

    rows=[]
    for ds,a in d.items():
        hist={r['policy']:r for r in a['historical']};g=hist['global'];s=hist['scorer_verifier']
        rows.append([a['title'],f'seed 42 · K={8 if ds=="cic2018" else 7}',f(g['macro_f1']),f(s['macro_f1']),n(s['calls']),n(s['accepted'])])
    out+=['<details class="history" id="history"><summary>과거 근거: 09/17 요소별 진단과 이전 S/V 결과</summary>',
        '<p>요소별 실험은 Global·expert·선택·채택 중 어디에 병목이 있는지 구분한 근거로 남긴다. 정제 이후에도 이전 seed 42 구성에서는 S/V를 실행했으며, 아래 수치를 현재 seed 43·K=4 결과와 구분한다.</p>',
        table(['데이터','이전 구성','Global','S+V','호출','채택'],rows,'historical-sv'),
        '<p><b>BoT-IoT의 이전 관찰:</b> 정제 후 Global Macro-F1은 0.3460이었고, 자연 비율 context 재실행에서는 약 0.204였다. 당시 context 변경만으로 DDoS·DoS 구분을 회복하지 못했다는 진단을 보존한다. 두 조건만으로 feature의 구분 정보가 없다고 확정하지 않는다.</p>',
        '<p class="note">09/17·09/18의 조건별 표와 09/28의 A/B 진단은 원본 보관본에 남겼다. 현재 3장은 같은 표본의 expert 직접 평가와 K 비교를 사용한다.</p></details>',
        '<footer><p>데이터 품질과 정제 기준: <button class="link-button" data-report-jump="dataset_quality_0911">0911 데이터 품질 감사</button> · 비교 모델 성능·비용: <button class="link-button" data-report-jump="sota_pfn_comparison">0908 SOTA</button></p>',
        '<details><summary>결과 출처와 원본 보관 위치</summary><p>K=4 S/V: <code>tabpfn/results/20260930_exp62_sv_current_bank_s43/</code><br>3장 K 비교·expert 직접 평가: <code>tabpfn/results/20260930_exp63_k_sweep_s43/</code><br>Oracle·과거 A/B: <code>tabpfn/results/20260928_exp59_residual_oracle_s43/</code><br>이전 S/V: <code>tabpfn/results/exp56_label_free_20260922/</code><br>통합 전 HTML: <code>docs/research/20260930/archive/pre_merge_reports/</code><br>3장 변경 전 HTML: <code>docs/research/20260930/archive/pre_exp63_capability/</code></p></details>',
        '<p>단일 seed 및 기존에 관찰한 test에 대한 예비 결과다. 논문 최종 주장에는 반복 seed와 독립 검증을 추가한다.</p></footer>']
    out.append('<script type="application/json" id="merged-data">'+json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</',r'<\/')+'</script>')
    out.append('<script>'+SWEEP_JS+'</script>')
    out.append('''<script>
document.getElementById('current-metric').addEventListener('change',function(){
  for(const el of document.querySelectorAll('.current-metric'))el.textContent=Number(el.getAttribute('data-'+this.value)).toFixed(4);
});
for(const b of document.querySelectorAll('[data-report-jump]'))b.addEventListener('click',function(){
  if(window.parent!==window)window.parent.postMessage({type:'lablog:navigate',reportId:this.dataset.reportJump},'*');
  else location.href='post_tabpfn.html#'+this.dataset.reportJump;
});
</script></main></body></html>''')
    return '\n'.join(out)

def integrate(source,data):
    index=HTML/'post_tabpfn.html';text=index.read_text();m=re.search(r'const REPORTS = (.*?);\n',text)
    reports=json.loads(m.group(1));reports=[r for r in reports if r['id'] not in OLD_IDS]
    entry=next(r for r in reports if r['id']==ID)
    entry.update(date='2026-09-30',titleKr=TITLE,titleEn='Residual oracle · Expert capability · Scorer & verifier',
                 verdict='3장 갱신 · K=2·4·6·8 expert 직접 평가 완료',verdictClass='warn',
                 b64=base64.b64encode(source.encode()).decode())
    prefix=text[:m.start()];tail=text[m.end():]
    prefix=prefix.replace('보고서 11편을 최신순으로 묶었습니다.','보고서 9편을 최신순으로 묶었습니다.')
    # Replace only the overview's current summary; older chronological entries remain historical.
    start=prefix.index('<p class="lede">',prefix.index('<h1>TabPFN 도입 이후 연구 기록</h1>'))
    end=prefix.index('<h2>연구 흐름 (최신순)</h2>',start)
    summary='''<p class="lede">모순 행 제거 후 SOTA 비교와 seed 43·K=2·4·6·8 검증을 완료했다. <strong>Expert 직접 평가에서 ToN Scanning의 이득은 test·validation 확인 구간에 함께 나타났지만, K 증가만으로 다른 취약 클래스 문제가 해결되지는 않았다.</strong> <button class="jump" data-jump="scorer_verifier_target_0928">Expert 역량과 S/V 검증</button> 3장에 K 비교와 클래스별 근거를 반영했다. 1·4장은 K=4 기준 결과다.</p>
<div class="status-grid">
<div class="status-cell warn"><p class="label">09/30 · CIC2018 K 비교 완료</p><p class="value">모든 K에서 Global / S+V <strong>0.7824</strong>, 호출·채택 0건. Expert 직접 평가에서 Infiltration은 전체 test F1 개선이 없고, Web attack의 test 개선도 validation 확인 구간과 함께 유지되지 않았다.</p></div>
<div class="status-cell good"><p class="label">09/30 · ToN K 비교 완료</p><p class="value">K=2·4·6·8의 S+V Macro-F1은 <strong>0.7636 / 0.7617 / 0.7677 / 0.7463</strong>. 모두 호출률 100%. Scanning의 분류 이득은 확인됐지만 MitM·Ransomware와 다른 클래스의 훼손은 추가 개선 대상이다.</p></div>
<!-- EXP61_OVERVIEW --><div class="status-cell"><p class="label">09/30 · 정제 후 SOTA 완료</p><p class="value">같은 C0 100k 비교군 최고: CIC2018 DistPFN <strong>0.7839</strong>, ToN XGBoost <strong>0.6983</strong>. 전체 train XGBoost는 0.7795 / 0.7162. <button class="jump" data-jump="sota_pfn_comparison">0908 클래스별 성능·비용</button>.</p></div>
<div class="status-cell"><p class="label">Expert 역량 · 같은 표본의 직접 평가</p><p class="value">모든 expert의 동일 test·validation 확인 구간을 비교한다. 클래스별 P/R/F1·FP와 context 구성을 연결했다. 과거 A/B는 배정 정책 진단으로 해석을 정정하고, 정답을 사용하는 residual oracle과 구분한다.</p></div>
</div>

'''
    prefix=prefix[:start]+summary+prefix[end:]
    prefix=re.sub(r'<!-- EXP59_FLOW_START -->.*?<!-- EXP59_FLOW_END -->','',prefix,flags=re.S)
    prefix=re.sub(r'<!-- MERGED_EXPERT_FLOW_START -->.*?<!-- MERGED_EXPERT_FLOW_END -->','',prefix,flags=re.S)
    for old in OLD_IDS:
        prefix=re.sub(r'(?m)^\s*<div class="flow-step">[^\n]*data-jump="'+old+r'"[^\n]*\n','',prefix)
    flow='''<!-- MERGED_EXPERT_FLOW_START --><div class="flow-step" id="merged-expert-overview"><time>09-30<br>09/17–28 통합</time><div class="flow-rail-dotcol"><div class="d"></div><div class="l"></div></div><div class="body"><button class="jump" data-jump="scorer_verifier_target_0928"><h3>정제 데이터에서의 Expert 역량과 S/V 검증</h3></button><p>K=4 기준 성능 → 정정한 oracle → K=2·4·6·8 expert 직접 평가 → S/V와 다음 개선. 3장은 데이터·K·평가 구간·지표를 바꿔 클래스 성능과 context 구성을 확인할 수 있다.</p></div></div><!-- MERGED_EXPERT_FLOW_END -->'''
    prefix=prefix.replace('<div class="flow-rail">','<div class="flow-rail">\n'+flow,1)
    prefix=prefix.replace('현재 진행은 09-16 보고서에 기록한다.','현재 결과는 정제 데이터의 Expert 역량·S/V 통합 보고서에 기록한다.')
    aliases=json.dumps({old:ID for old in OLD_IDS},ensure_ascii=False)
    if 'const REPORT_ALIASES' not in tail:tail='const REPORT_ALIASES = '+aliases+';\n'+tail
    tail=tail.replace('function activate(id) {\n  const r = REPORTS.find(x => x.id === id);',
                      'function activate(id) {\n  id = REPORT_ALIASES[id] || id;\n  const r = REPORTS.find(x => x.id === id) || REPORTS[0];\n  id = r.id;')
    tail=tail.replace("const startId = (location.hash || '').replace('#', '') || REPORTS[0].id;",
                      "const requestedId = (location.hash || '').replace('#', '') || REPORTS[0].id;\nconst startId = REPORT_ALIASES[requestedId] || requestedId;")
    if "e.data.type === 'lablog:navigate'" not in tail:
        tail=tail.replace('</script>','''window.addEventListener('message', e => {
  if (e.source !== viewer.contentWindow || !e.data || e.data.type !== 'lablog:navigate') return;
  const id = REPORT_ALIASES[e.data.reportId] || e.data.reportId;
  if (REPORTS.some(r => r.id === id)) activate(id);
});
window.addEventListener('hashchange', () => {
  const requested = location.hash.slice(1), id = REPORT_ALIASES[requested] || requested;
  if (REPORTS.some(r => r.id === id) && !document.querySelector('.nav-item.active[data-id="'+id+'"]')) activate(id);
});
</script>''',1)
    index.write_text(prefix+'const REPORTS = '+json.dumps(reports,ensure_ascii=False)+';\n'+tail)
    for old in OLD_IDS:
        target='post_tabpfn.html#'+ID
        (HTML/(old+'.html')).write_text(f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="0; url={target}"><title>{TITLE}</title></head><body><p>이 보고서는 통합되었습니다. <a href="{target}">{TITLE} 열기</a></p></body></html>\n')

def main():
    dest=archive();data=load();source=apply_rules(build(data));(HTML/(ID+'.html')).write_text(source);integrate(source,data)
    write(DOC/'merged_expert_publication.json',dict(local_updated=True,archive=str(dest.relative_to(ROOT)),
        canonical_report=ID,merged_report_ids=[*OLD_IDS,ID],aliases_preserved=True,
        html=str((HTML/'post_tabpfn.html').relative_to(ROOT)),git_pushed=False,online_artifact_updated=False,
        online_blocker='No connected Claude artifact editing tool is available',numeric_validation=['exp62_numeric_validation.json','exp63_capability_numeric_validation.json'],
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [HTML/(ID+'.html'),HTML/'post_tabpfn.html']}))
    print(json.dumps(dict(report=ID,bytes=len(source.encode()),datasets=list(data['datasets']),archive=str(dest)),ensure_ascii=False))

if __name__=='__main__':main()
