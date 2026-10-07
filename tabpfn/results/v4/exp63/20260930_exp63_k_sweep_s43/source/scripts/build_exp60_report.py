#!/usr/bin/env python3
"""Local EXP60 report from completed, frozen predictions; no new inference."""
import base64
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from analyze_exp59_expert_capability import evaluate, confusion, metrics
from build_exp56_report_section import table
from report_typography import apply_typography

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'tabpfn/results/20260929_exp60_counterexamples_s43'
BASE=ROOT/'tabpfn/results/20260928_exp59_residual_oracle_s43'
REPORT_ID='counterexample_context_0929'
REPORT=ROOT/'lablog/html_report'/f'{REPORT_ID}.html'
INDEX=REPORT.parent/'post_tabpfn.html'
DOCS=ROOT/'docs/research/20260929'
ARMS=['designed','counter_random','counter_near']
ARM_NAMES={'designed':'기존 context','counter_random':'무작위 상대사례','counter_near':'가까운 상대사례'}
DS={'cic2018':'CIC2018','toniot':'ToN-IoT'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def records(df):return json.loads(df.to_json(orient='records',double_precision=15))
def n(v):return f'{int(v):,}'


def load():
    payload={'arms':ARMS,'arm_names':ARM_NAMES,'datasets':{},'sources':{},'seed':43,'K':4}
    assert json.loads((RESULTS/'status.json').read_text())['state']=='complete'
    for ds,title in DS.items():
        out=RESULTS/f'{ds}_s43';base=BASE/f'{ds}_s43';source=base/'fresh_bank'
        assert (out/'COMPLETE.json').exists()
        design=json.loads((source/'design.json').read_text());names=design['class_names'];C=len(names)
        identity=np.load(source/'evaluation_identity.npz');y=identity['y']
        g=np.load(source/'predictions/global_raw.npy')
        membership={'residual':np.load(base/'residual_test_regions.npy'),
                    'observable':np.load(source/'fixed_regions.npz')['test']}
        pc=pd.read_csv(out/'class_metrics.csv').set_index(['model','scope','class_name']).sort_index()
        summary=pd.read_csv(out/'summary.csv').set_index('model')
        composition=pd.read_csv(out/'context_compositions.csv')
        old_composition=pd.read_csv(base/'context_compositions.csv').query("arm=='designed'").set_index('expert')
        d={'title':title,'names':names,'rows':len(y),'anchor_rows':design['anchor_rows'],
           'summary':{model:json.loads(row.to_json()) for model,row in summary.iterrows()},
           'full_classes':records(pc.reset_index().query("scope=='full_test'")),
           'experts':{},'composition':records(composition),'old_composition':records(old_composition.reset_index())}
        for k in range(1,5):
            e={scope:{} for scope in membership}
            audit=json.loads((out/f'contexts/e{k}_selection.json').read_text())
            assert audit['per_class_counts_equal'] and audit['all_pairs_cross_class']
            a=composition[(composition.arm=='counter_random') & (composition.expert==k)].iloc[0]
            b=composition[(composition.arm=='counter_near') & (composition.expert==k)].iloc[0]
            np.testing.assert_array_equal(a[names].to_numpy(),b[names].to_numpy())
            assert int(a.block_rows)==int(old_composition.loc[k,'block_rows'])
            e['audit']=audit
            for arm in ARMS:
                p=(source if arm=='designed' else out)/f'predictions/{arm}_e{k}_raw.npy'
                pred=np.load(p)
                assert pred.shape==y.shape
                payload['sources'][str(p.relative_to(ROOT))]=sha(p)
                for scope,m in membership.items():
                    value=evaluate(y,g,pred,names,m==k-1)
                    assert sum(c['fixed'] for c in value['classes'])==value['fixed']
                    assert sum(c['harmed'] for c in value['classes'])==value['harmed']
                    for c in value['classes']:
                        ref=pc.loc[(f'{arm}_e{k}',scope,c['name'])]
                        for field in ['TP','FP','FN','support']:
                            assert c['expert_metrics'][field]==int(ref[field])
                        np.testing.assert_allclose(c['expert_metrics']['f1'],ref.f1,atol=1e-12)
                    value.pop('confusion_pairs')
                    for c in value['classes']:c.pop('fp_sources')
                    e[scope][arm]=value
            d['experts'][str(k)]=e
        for model,row in summary.iterrows():
            q=pc.loc[(model,'full_test')]
            np.testing.assert_allclose(q.f1.mean(),row.macro_f1,atol=1e-12)
            assert int(q.support.sum())==len(y)
            np.testing.assert_allclose((q.TP.sum()/len(y)),row.accuracy,atol=1e-12)
        for p in [out/'summary.csv',out/'class_metrics.csv',out/'CONTEXTS_COMPLETE.json',source/'evaluation_identity.npz',base/'residual_test_regions.npy',source/'fixed_regions.npz']:
            payload['sources'][str(p.relative_to(ROOT))]=sha(p)
        payload['datasets'][ds]=d
    (DOCS/'exp60_report_data.json').write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False)+'\n')
    return payload


CSS='''
:root{--ink:#203044;--muted:#617386;--accent:#245e8d;--line:#dce4ec;--paper:#fff;--pale:#edf3f8;--loss:#a94e3d;--gain:#236a56}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f6f8fa;color:var(--ink);line-height:1.75;font-size:15px}
main{max-width:1180px;margin:auto;padding:48px 36px 70px}h1{word-break:keep-all;font-size:34px;letter-spacing:-1px;line-height:1.3;margin:8px 0 18px}h2{font-size:25px;margin:56px 0 18px;scroll-margin-top:20px}h3{font-size:18px;margin:24px 0 12px}h4{margin:22px 0 10px}p{margin:10px 0}.eyebrow{font-size:12px;font-weight:700;letter-spacing:1px;color:var(--accent)}.lede{font-size:18px;max-width:950px}.note,figcaption{font-size:13px;color:var(--muted)}.pill{display:inline-block;font-size:12px;padding:4px 11px;border-radius:14px;background:var(--pale);margin:4px 6px 4px 0}
nav{display:flex;flex-wrap:wrap;gap:8px 22px;margin:24px 0}a{color:var(--accent)}.cards{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:24px 0}.card,.viewer{background:white;border:1px solid var(--line);border-radius:10px;padding:22px;min-width:0}.card h3{margin-top:0}.big{font-size:27px;font-weight:700;color:var(--accent)}.definition{border-left:4px solid var(--accent);background:var(--pale);padding:14px 20px;margin:18px 0}.wrap{max-width:100%;overflow-x:auto;margin:12px 0 22px;border:1px solid var(--line);border-radius:6px}table{border-collapse:collapse;width:100%;min-width:620px;background:white;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}th{background:#e8eff5;color:#264762;font-weight:700}th:first-child,td:first-child{text-align:left}tbody tr:nth-child(even){background:#f8fafc}.total{font-weight:700;background:#e8eff5!important}caption{text-align:left}label{display:inline-block;font-size:13px;font-weight:700;margin:4px 14px 4px 0}select{font:inherit;display:inline-block;max-width:100%;padding:7px 9px;border:1px solid #bccbd8;border-radius:5px;background:white;margin-left:7px}.controls{display:flex;flex-wrap:wrap;gap:4px 12px}.gain{color:var(--gain)}.loss{color:var(--loss)}.mini{font-size:12px;color:var(--muted)}.reading{background:#edf3f8;padding:15px 18px;margin:18px 0}.barrow{display:grid;grid-template-columns:135px 1fr 65px;gap:9px;align-items:center;margin:9px 0;font-size:13px}.bartrack{background:#eef2f6;height:13px;border-radius:4px;overflow:hidden}.bar{height:100%;background:#245e8d}.context-row{display:flex;gap:10px;align-items:stretch;margin:10px 0}.context-label{min-width:130px;font-size:14px;font-weight:700;align-self:center}.block{border:1px solid var(--line);background:white;border-radius:5px;padding:12px;text-align:center;flex:1}.block.core{background:#e7f0f7;flex:3}.block.counter{background:#d7e7f2}.block strong{display:block}details{margin:16px 0}summary{cursor:pointer;color:var(--accent);font-weight:700}footer{border-top:1px solid var(--line);margin-top:48px;padding-top:20px;color:var(--muted);font-size:12px}code{overflow-wrap:anywhere}.nowrap{white-space:nowrap}[hidden]{display:none!important}
@media(max-width:720px){main{padding:24px 16px 45px}h1{font-size:27px}h2{font-size:22px}.lede{font-size:16px}.cards{grid-template-columns:1fr}.card,.viewer{padding:15px}label{display:block;margin-right:0}.context-row{flex-wrap:wrap}.context-label{width:100%}.block{padding:9px}.barrow{grid-template-columns:116px 1fr 54px;font-size:12px}select{margin-left:3px}th,td{padding:8px}nav{gap:5px 14px}}
'''


def composition(data):
    parts=[]
    for ds,d in data['datasets'].items():
        rows=[]
        for k,e in d['experts'].items():
            a=e['audit'];rows.append([f'e{k}',n(d['anchor_rows']),n(a['block_rows']),n(a['shared_core_rows']),n(a['counterpart_rows']),
                f"{a['random_mean_distance']:.3f} / {a['near_mean_distance']:.3f}"])
        parts += [f'<h3>{d["title"]}</h3>',table(['Expert','공통 anchor','기존 block','유지 core','상대사례','평균 거리 무작위 / 가까운'],rows,f'{ds}-contexts')]
        parts.append('<details><summary>특화 block의 클래스별 개수와 상대사례의 Global 정답 비율</summary>')
        counts=[]
        old={int(x['expert']):x for x in d['old_composition']}
        new={int(x['expert']):x for x in d['composition'] if x['arm']=='counter_near'}
        for k in range(1,5):
            counts.append([f'e{k} 기존',*[n(old[k][c]) for c in d['names']]])
            counts.append([f'e{k} 신규 두 조건',*[n(new[k][c]) for c in d['names']]])
        parts.append(table(['특화 block',*d['names']],counts,f'{ds}-class-composition'))
        rows=[]
        for k,e in d['experts'].items():
            a=e['audit'];den=a['counterpart_rows']
            rows.append([f'e{k}',n(den),f"{a['random_counter_global_correct']/den:.1%}",f"{a['near_counter_global_correct']/den:.1%}"])
        parts += [table(['Expert','상대사례 수','무작위 중 Global 정답','가까운 중 Global 정답'],rows),
            '<p class="note">거리 기준은 고정된 입력 표현과 Global 예측 확률이다. 가까운 사례에도 Global 오류가 포함된다. 이 비율은 후보의 성질을 설명하며 정답 사례만 선택한 실험은 아니다.</p></details>']
    return '\n'.join(parts)


INTERPRETATIONS={
'cic2018':{
'1':'가까운 조건은 입력 유사군에서 DDoS 교정을 늘리지만 기존보다 훼손도 늘린다. 담당영역에서는 교정 6,246 → 4,671, 훼손 50 → 2,567로 악화된다. e1에는 상대사례를 일괄 20% 배정하기보다 기존 교정 사례 보존과 상대 클래스별 배분을 먼저 점검한다.',
'2':'Bot 중심의 입력 유사군에서 세 조건 모두 교정·훼손 0건이다. 이 유사군의 비-Bot 표본은 4개뿐이므로 상대사례 효과나 음성 구분 능력을 확인하기에는 부족하다. 추가 평가는 Bot과 비슷한 음성이 충분한 범위에서 구성한다.',
'3':'입력 유사군의 Infiltration FP는 기존 21,535, 무작위 20,545, 가까운 21,447이다. 가까운 사례 1,937개 중 1,932개가 benign이어도 오탐은 크게 줄지 않았다. 담당영역 교정도 9,598 → 8,041로 감소한다. 특화 양성 보존과, Global이 구분 가능한 가까운 음성의 효과를 분리해 확인한다.',
'4':'가까운 조건도 담당영역의 benign 복구 편향을 바꾸지 못했다. 입력 유사군의 Web attack TP는 기존 14, 무작위 15, 가까운 13으로 작다. Web 사례를 보존하면서 benign과의 경계 사례를 보강하는 배분을 별도로 검토한다.'},
'toniot':{
'1':'가까운 조건에서 입력 유사군의 교정 5,225 → 5,445, 훼손 1,528 → 1,315로 함께 개선된다. 기존 scanning 교정 능력을 유지하면서 주변 클래스 오인을 줄인 사례다. 전체 입력 기반 배정의 scanning 점수는 다른 expert의 영향도 받으므로 이 expert 자체의 결과와 구분한다.',
'2':'담당영역의 훼손이 75,492 → 63,943으로 줄지만 교정 40,940보다 여전히 많다. 입력 유사군 훼손은 무작위 22,797, 가까운 23,141로 무작위가 조금 적다. 상대 클래스 보강 효과와 근접 검색의 추가 효과를 분리하고, benign을 공격으로 바꾸는 사례를 계속 억제해야 한다.',
'3':'가까운 조건에서 입력 유사군의 교정 8,262 → 14,651, 훼손 55,466 → 41,037로 개선된다. DoS·Injection·XSS 구분의 개선이 크지만 benign→MitM 오인은 남는다. 담당영역 교정은 3,274 → 3,078로 줄어 두 범위의 이득을 함께 검증해야 한다.',
'4':'입력 유사군에서 교정 14 → 46, 훼손 8 → 9이며 ransomware FP가 1,449개 줄어든다. 다만 정답 residual 영역은 ransomware 단일 클래스이고 입력 유사군에서도 낮은 precision이 남는다. 높은 recall과 음성 구분 능력을 분리해 판단한다.'}}


def build(data):
    data['interpretations']=INTERPRETATIONS
    parts=[f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>상대사례 context와 Expert 역량 · 0929</title><style>{CSS}</style></head><body><main>',
    '<header><p class="eyebrow">0929 연구 보고서 · EXP60 · 완료 결과</p><h1>상대사례 context와 Expert 역량</h1><p class="lede">ToN에서는 가까운 상대사례가 입력 기반 배정 성능을 높였다. CIC2018에서는 같은 규칙의 이득이 유지되지 않았다. 교정 대상을 보존하면서 어떤 상대 클래스를 보강할지 expert별로 조절해야 한다.</p><span class="pill">CIC2018 · ToN-IoT</span><span class="pill">seed 43 · K=4</span><span class="pill">S/V 미적용</span><span class="pill">16개 신규 expert 완료</span></header>',
    '<nav><a href="#composition">1. Context 비교</a><a href="#overall">2. 전체 성능</a><a href="#capability">3. Expert 역량</a><a href="#detail">4. 클래스별 근거</a><a href="#next">5. 해석과 개선</a></nav>',
    '<div class="cards"><article class="card"><h3>ToN-IoT · 입력 기반 고정 배정</h3><div class="big">0.7191 → 0.7530</div><p>기존 → 가까운 상대사례 Macro-F1, 무작위는 0.7300</p><p class="note">Global은 0.6796. 다만 정확도는 Global 90.13%보다 낮은 89.40%로, 남은 정답 훼손을 S/V 단계에서 확인해야 한다.</p></article><article class="card"><h3>CIC2018 · 입력 기반 고정 배정</h3><div class="big">0.7613 → 0.7579</div><p>기존 → 가까운 상대사례 Macro-F1, 무작위는 0.7623</p><p class="note">세 context 모두 Global 0.7824보다 낮다. 가까운 사례 선택만으로 Infiltration 오탐을 해결하지 못했다.</p></article></div>',
    '<h2 id="composition">1. 무엇을 바꿨는가</h2><p>공통 anchor와 Global을 고정하고 특화 block의 일부를 다른 클래스 사례로 교체했다. 기존 block에도 Global 정답과 오류가 모두 들어 있었다. 이번 변화는 상대 클래스를 의도적으로 짝지어 보강한 것이다.</p>',
    '<div class="context-row"><span class="context-label">기존</span><div class="block"><strong>공통 anchor</strong>그대로 유지</div><div class="block core"><strong>Residual 특화 block</strong>기존 선택 전체</div></div>',
    '<div class="context-row"><span class="context-label">신규 두 조건</span><div class="block"><strong>공통 anchor</strong>그대로 유지</div><div class="block core"><strong>기존 block의 80%</strong>두 조건의 공통 core</div><div class="block counter"><strong>상대사례 20%</strong>무작위 또는 가까운</div></div>',
    '<p>상대사례 후보는 학습 Expert 풀 전체에서 찾으며 query와 라벨이 다른 행으로 제한한다. 가까운 조건은 고정된 입력 표현·Global 확률 거리로 검색한다. 무작위 조건은 같은 query·상대 클래스 조합을 유지해 추출한다. 두 신규 조건은 <b>anchor, core, 총 context 크기, 클래스별 개수</b>가 같다.</p>',
    '<div class="definition"><b>근접 검색의 효과는 무작위와 가까운 조건 사이에서 비교한다.</b> 기존 대비 차이에는 원래 block 20%를 교체한 효과도 포함된다. 20%는 이번 preliminary의 고정 비율이며 최적값으로 선택한 것이 아니다.</div>',composition(data),
    '<h2 id="overall">2. 전체 test에서의 변화</h2><div class="definition"><b>입력 기반 고정 배정:</b> 입력 표현과 Global 확률로 가장 가까운 기존 centroid의 expert를 선택하고, 그 expert의 raw 예측을 그대로 사용한다. S/V가 선택·거절한 결과가 아니다.<br><b>Residual 소속 oracle:</b> test 정답을 포함한 residual로 가까운 군집을 찾고 해당 expert의 예측을 그대로 채점한다. 소속을 아는 진단이며 실제 모델 성능이나 모든 라우팅의 상한이 아니다.</div>',
    '<div class="controls"><label>전체 평가<select id="overall-scope"><option value="input_nearest">입력 기반 고정 배정</option><option value="residual_oracle">정답 residual 소속 oracle</option></select></label><label>클래스 지표<select id="overall-metric"><option value="f1">F1</option><option value="precision">Precision</option><option value="recall">Recall</option></select></label></div><div id="overall-content"></div>',
    '<h2 id="capability">3. Expert별 교정과 훼손</h2><p>A는 정답 residual로 정한 담당영역, B는 입력 정보만으로 가까운 expert에 배정한 유사군이다. B에는 담당영역 안팎의 행이 함께 포함된다. 각 범위 안에서 같은 행의 Global과 expert를 비교한다.</p><label>관점<select id="summary-scope"><option value="observable">B · 입력 유사군</option><option value="residual">A · 담당 residual 영역</option></select></label><div id="capability-content"></div><p class="note">교정: Global 오답 → expert 정답. 훼손: Global 정답 → expert 오답. 교정−훼손은 정확도 변화의 건수이며 Macro-F1의 변화와 동일하지 않다.</p>',
    '<h2 id="detail">4. 어떤 클래스에서 달라졌는가</h2><div class="viewer"><div class="controls"><label>데이터<select id="detail-dataset"><option value="cic2018">CIC2018</option><option value="toniot">ToN-IoT</option></select></label><label>Expert<select id="detail-expert"><option>1</option><option>2</option><option selected>3</option><option>4</option></select></label><label>관점<select id="detail-scope"><option value="observable">B · 입력 유사군</option><option value="residual">A · 담당 residual 영역</option></select></label><label>지표<select id="detail-metric"><option value="f1">F1</option><option value="precision">Precision</option><option value="recall">Recall</option></select></label></div><div id="detail-content"></div></div>',
    '<h2 id="next">5. 이번 결과의 해석과 다음 개선</h2>',
    '<h3>ToN: 유효한 context 후보, 남은 훼손의 억제</h3><p>가까운 조건의 이득은 특히 e3의 입력 유사군에서 뚜렷하다. 반면 e2는 가까운 조건이 무작위보다 일관되게 낫지 않다. 모든 expert에 같은 규칙을 강제하기보다 validation에서 expert별 context를 비교한 뒤, 새 bank의 route 예측으로 S/V를 학습해야 한다. Macro-F1 상승과 함께 Global 정답 훼손이 줄어드는지도 확인한다.</p>',
    '<h3>CIC2018: 특화 사례 보존과 상대사례 선택의 분리</h3><p>e3의 가까운 상대사례는 대부분 benign이지만 Infiltration FP가 기존 대비 88개만 줄었고, 담당영역 교정은 1,557개 감소했다. 단순히 가까운 다른 라벨을 넣는 것만으로 충분하지 않다. 핵심 tail 양성을 보존한 조건, Global이 맞히는 가까운 상대사례 조건, 혼동 클래스별 배분 조건을 구분해 validation에서 확인하는 것이 다음 개선이다.</p>',
    '<h3>가까움과 유용한 경계 사례는 같은 기준이 아니다</h3><p>가까운 상대사례의 Global 정답 비율은 CIC e3 77.3%, ToN e3 40.0%이고 무작위는 각각 99.5%, 90.9%다. 근접 검색은 어려운 사례를 많이 가져오지만, 그 사례가 expert의 경계를 개선하는지는 별도 검증 대상이다. 이번 실험은 이 차이를 보여주며 거리만으로 채택을 결정하지 않는다.</p>',
    '<p class="note">단일 seed·K=4의 preliminary이며 기존에 관찰한 development holdout을 사용했다. 현재 결과로 최종 모델이나 SOTA 우위를 주장하지 않는다. 다음 단계는 validation에서 context와 gate 설정을 고정한 S/V 비교다.</p>',
    '<details><summary>결과 파일과 계산 검증</summary><p><code>tabpfn/results/20260929_exp60_counterexamples_s43/</code></p><p>기존·신규 raw 예측에서 A/B의 TP·FP·FN과 교정·훼손을 재집계해 CSV와 대조했다. 각 상세표의 클래스별 교정·훼손 합은 위 expert 합계와 일치한다. 보고서 작성 과정의 추가 추론은 없다.</p></details>',
    '<footer>2026-09-29 · 로컬 HTML 보고서 · Global·anchor·평가 행·기존 centroid 고정 · 전체 실험 완료 16:57 KST</footer>',
    '<script id="exp60-data" type="application/json">'+json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</',r'<\/')+'</script>',
    '<script>'+JS+'</script></main></body></html>']
    page=apply_typography('\n'.join(parts));REPORT.write_text(page)
    return page


JS=r'''
const D=JSON.parse(document.getElementById('exp60-data').textContent),arms=D.arms,AN=D.arm_names;
const N=x=>Number(x).toLocaleString('en-US'),F=x=>Number(x).toFixed(4),P=x=>(100*x).toFixed(2)+'%';
const $=id=>document.getElementById(id),E=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
function T(head,rows,kind=''){return '<div class="wrap"><table data-kind="'+kind+'"><thead><tr>'+head.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>'}
function V(m,key){if((key==='precision'&&!(m.TP+m.FP))||((key==='recall'||key==='f1')&&!m.support))return '—';return F(m[key])}
function overall(){const scope=$('overall-scope').value,key=$('overall-metric').value;let html='';
 for(const [ds,d] of Object.entries(D.datasets)){
  const models=['global',...arms.map(a=>a+'_'+scope)],labels=['Global',...arms.map(a=>AN[a])];
  html+='<h3>'+d.title+' · '+N(d.rows)+'행</h3><div class="card">';
  models.forEach((m,i)=>html+='<div class="barrow"><span>'+labels[i]+'</span><div class="bartrack"><div class="bar" style="width:'+(d.summary[m].macro_f1*100)+'%;background:'+['#8794a2','#668fac','#8bb4ce','#245e8d'][i]+'"></div></div><b>'+F(d.summary[m].macro_f1)+'</b></div>');
  html+='</div>'+T(['조건','Macro-F1','정확도','교정','훼손'],models.map((m,i)=>{const q=d.summary[m];return [labels[i],F(q.macro_f1),P(q.accuracy),N(q.fixed),N(q.harmed)]}),ds+'-overall');
  const rows=d.names.map(c=>[E(c),N(d.full_classes.find(x=>x.model==='global'&&x.class_name===c).support),...models.map(m=>V(d.full_classes.find(x=>x.model===m&&x.class_name===c),key))]);
  html+='<details><summary>전체 배정의 클래스별 '+key.toUpperCase()+'</summary>'+T(['클래스','Test 수',...labels],rows,ds+'-overall-classes')+'</details>';
 }$('overall-content').innerHTML=html;
}
function capability(){const scope=$('summary-scope').value;let html='';
 for(const [ds,d] of Object.entries(D.datasets)){let rows=[];
  for(const [k,e] of Object.entries(d.experts)){rows.push(['e'+k,N(e[scope].designed.rows),...arms.map(a=>N(e[scope][a].fixed)+' / '+N(e[scope][a].harmed))]);}
  html+='<h3>'+d.title+'</h3>'+T(['Expert','평가 행','기존 교정 / 훼손','무작위 교정 / 훼손','가까운 교정 / 훼손'],rows,ds+'-expert-summary');
 }$('capability-content').innerHTML=html;
}
function detail(){const ds=$('detail-dataset').value,k=$('detail-expert').value,scope=$('detail-scope').value,key=$('detail-metric').value;
 const d=D.datasets[ds],e=d.experts[k][scope],base=e.designed,labels=arms.map(a=>AN[a]);
 let html='<h3>'+d.title+' · e'+k+' · '+(scope==='residual'?'A 담당 residual 영역':'B 입력 유사군')+'</h3>';
 html+='<p>동일한 '+N(base.rows)+'행 평가 · Global 정답 '+N(base.global_correct)+'행</p>';
 html+=T(['조건','교정','훼손','Expert 정답'],arms.map(a=>[AN[a],N(e[a].fixed),N(e[a].harmed),N(e[a].expert_correct)]),'detail-totals');
 html+='<div class="reading">'+D.interpretations[ds][k]+'</div><h4>클래스별 '+key.toUpperCase()+'</h4>';
 let rows=d.names.map((c,i)=>[E(c),N(base.classes[i].expert_metrics.support),V(base.classes[i].global_metrics,key),...arms.map(a=>V(e[a].classes[i].expert_metrics,key))]);
 html+=T(['클래스','양성 수','Global',...labels],rows,'detail-metrics');
 html+='<h4>클래스별 교정·훼손의 합계 확인</h4><p class="note">각 행은 해당 실제 클래스에서 발생한 교정·훼손이다. FP는 그 열의 클래스로 잘못 예측된 다른 클래스 표본의 합이다.</p>';
 rows=d.names.map((c,i)=>[E(c),...arms.map(a=>{const m=e[a].classes[i].expert_metrics;return N(m.TP)+' / '+N(m.FP)}),...arms.map(a=>{const q=e[a].classes[i];return N(q.fixed)+' / '+N(q.harmed)})]);
 rows.push(['합계','—','—','—',...arms.map(a=>N(e[a].fixed)+' / '+N(e[a].harmed))]);
 html+=T(['클래스','기존 TP / FP','무작위 TP / FP','가까운 TP / FP','기존 교정 / 훼손','무작위 교정 / 훼손','가까운 교정 / 훼손'],rows,'detail-counts');
 let pairs=[];for(let a=0;a<d.names.length;a++)for(let b=0;b<d.names.length;b++)if(a!==b){const values=arms.map(arm=>e[arm].expert_confusion[a][b]),g=base.global_confusion[a][b];if(g||values.some(x=>x))pairs.push({a,b,g,values,rank:Math.max(...values)})}
 pairs.sort((a,b)=>b.rank-a.rank||a.a-b.a||a.b-b.b);
 html+='<h4>주요 혼동 쌍</h4><p class="note">세 context 중 최대 오인 수가 큰 10개 쌍. 교정·훼손 전체 합계는 위의 모든 클래스 표에서 확인한다.</p>';
 html+=T(['실제 → 예측','Global',...labels],pairs.slice(0,10).map(p=>[E(d.names[p.a]+' → '+d.names[p.b]),N(p.g),...p.values.map(N)]),'detail-confusions');
 html+='<p class="note">양성 0개인 클래스의 F1·Recall과 예측 0개인 클래스의 Precision은 —로 표시한다. A/B의 평가 집합은 달라 서로의 F1을 직접 빼서 해석하지 않는다.</p>';
 $('detail-content').innerHTML=html;
}
for(const id of ['overall-scope','overall-metric'])$(id).addEventListener('change',overall);
$('summary-scope').addEventListener('change',capability);
for(const id of ['detail-dataset','detail-expert','detail-scope','detail-metric'])$(id).addEventListener('change',detail);
overall();capability();detail();
'''


def integrate(page):
    source=INDEX.read_text();m=re.search(r'const REPORTS = (.*?);\n',source)
    reports=json.loads(m.group(1));reports=[r for r in reports if r['id']!=REPORT_ID]
    reports.insert(1,dict(id=REPORT_ID,date='2026-09-29',titleKr='상대사례 context와 Expert 역량',
       titleEn='Random vs nearby counterexamples · seed 43',verdict='두 데이터 완료 · ToN 개선, CIC 경계 보강 재설계',verdictClass='warn',
       group='report',type='iframe',b64=base64.b64encode(page.encode()).decode()))
    source=source[:m.start(1)]+json.dumps(reports,ensure_ascii=False)+source[m.end(1):]
    flow='<div class="flow-step"><time>09-29</time><div class="flow-rail-dotcol"><div class="d"></div><div class="l"></div></div><div class="body"><button class="jump" data-jump="'+REPORT_ID+'"><h3>EXP60 · 무작위와 가까운 상대사례 context</h3></button><p>같은 크기·클래스별 개수의 상대사례 두 조건, 16개 expert 완료. 입력 기반 배정 Macro-F1은 기존→가까운 조건에서 ToN 0.7191→0.7530, CIC2018 0.7613→0.7579. Expert별 교정·훼손과 클래스별 FP를 연결해 다음 context와 S/V 개선을 정리했다.</p></div></div>'
    finding='<li><p><strong>(09-29 EXP60) 가까운 상대사례의 효과는 데이터와 expert별로 다르다.</strong> ToN은 입력 기반 배정의 Macro-F1이 개선됐지만 CIC2018은 개선되지 않았다. ToN e3의 교정·혼동 억제는 유효한 후보이며 CIC은 특화 양성 보존과 상대 음성 선택을 분리해 점검한다. S/V는 아직 적용하지 않았다.</p></li>'
    nxt='<li><strong>상대사례 context 이후 S/V 검증</strong> · EXP60 두 데이터 완료. Validation에서 expert별 context 후보를 비교하고 새 bank의 route 예측으로 gate를 학습한다. CIC은 기존 교정 사례 보존·상대 클래스 배분을 먼저 확인한다.</li>'
    for tag,anchor,content in [('FLOW','<h2>연구 흐름 (최신순)</h2>',flow),('FINDING','<ol class="findings">',finding),('NEXT','<ol class="next-list">',nxt)]:
        source=re.sub(r'<!-- EXP60_'+tag+r'_START -->.*?<!-- EXP60_'+tag+r'_END -->\n?','',source,flags=re.S)
        assert anchor in source
        source=source.replace(anchor,anchor+'\n<!-- EXP60_'+tag+'_START -->'+content+'<!-- EXP60_'+tag+'_END -->',1)
    count=sum(r.get('group')=='report' for r in reports)
    source=re.sub(r'8&ndash;9월의 보고서 (?:\d+|아홉)편?을 최신순으로 묶었습니다\.',f'8&ndash;9월의 보고서 {count}편을 최신순으로 묶었습니다.',source)
    INDEX.write_text(source)


if __name__=='__main__':
    preserved={p:sha(p) for p in [REPORT.parent/'scorer_verifier_target_0918.html',REPORT.parent/'scorer_verifier_target_0928.html']}
    data=load();page=build(data);integrate(page)
    for p,digest in preserved.items():assert sha(p)==digest
    manifest=dict(report=str(REPORT.relative_to(ROOT)),seed=43,K=4,completed_new_experts=16,
        report_sha256=sha(REPORT),extra_inference_for_report=False,
        validation={'raw_prediction_counts_match_csv':True,'class_corrections_sum_to_expert_totals':True,'old_reports_unchanged':True},
        publishing={'local':'complete','integrated':True,'online':'not requested; user explicitly requested local post HTML','git_push':'not performed'})
    (DOCS/'exp60_report_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(REPORT)
