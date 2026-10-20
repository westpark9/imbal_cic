#!/usr/bin/env python3
"""Build the EXP71 HTML and paper-writing notes from validated saved results."""
from pathlib import Path
import argparse
import base64
import hashlib
import html
import json
import re
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(ROOT/'scripts/common'))
from exp71_bounded_updates import aggregate, CASES, STAGES
from report_order import apply_rules  # table rules (class order, numeric right-alignment style)

REPORT_ID = 'bounded_updates_1013'
TITLE = '새 공격 반영: 제한된 과거 사례와 갱신 비용'
NAMES = {'benign':'Benign','brute_force':'Brute force','dos':'DoS','ddos':'DDoS',
         'web_attacks':'Web attack','infiltration':'Infiltration','bot':'Bot',
         'scanning':'Scanning','injection':'Injection','password':'Password',
         'xss':'XSS','ransomware':'Ransomware','backdoor':'Backdoor','mitm':'MitM'}


def interpretation(data):
    """Describe observed class-level differences without inventing their cause."""
    paragraphs=[]
    for s in data['summary']:
        ds,case=s['dataset'],s['case']
        title=('CIC2018' if ds=='cic2018' else 'ToN')+' · '+('균등' if case=='uniform' else '공격 내 자연')
        final={m:max((r for r in data['results'] if (r['dataset'],r['case'],r['method'])==(ds,case,m)),key=lambda r:r['stage']) for m in ['xgb','tabpfn']}
        x,p=s['xgb'],s['tabpfn']
        by_class={m:{c['name']:c for c in final[m]['classes']} for m in final}
        differences=sorted((c for c in by_class['xgb'] if c!='benign'),
            key=lambda c:abs(by_class['tabpfn'][c]['f1']-by_class['xgb'][c]['f1']),reverse=True)[:2]
        difference_text='; '.join(f"{NAMES[c]} XGB {100*by_class['xgb'][c]['f1']:.2f}% / TabPFN {100*by_class['tabpfn'][c]['f1']:.2f}%" for c in differences)
        paragraphs.append(f"{title}: 후속 갱신 합계는 XGB {x['update_sum']:.2f}초, TabPFN {p['update_sum']:.2f}초다. "
            f"TabPFN−XGB 최종 Macro-F1 차이는 {100*(p['final_macro_f1']-x['final_macro_f1']):+.2f}%p, "
            f"정상 오탐률 차이는 {100*(p['final_benign_fpr']-x['final_benign_fpr']):+.2f}%p다. "
            f"전체 {p['final_delivered_rows']:,}개를 공급했고 최종 갱신에는 {p['final_context_rows']:,}개를 사용했다. "
            '공격별 F1의 절대 차이가 가장 큰 두 클래스는 '+difference_text+'다.')
        counts=data['datasets'][ds]['fixed_test_class_counts']
        rare=sorted((c for c in counts if c!='benign'),key=counts.get)[:2]
        # Follow the same descending evaluation-support order as the tables.
        rare.reverse()
        parts=[]
        for c in rare:
            details=[]
            for m in ['xgb','tabpfn']:
                r=final[m];v=next(v for v in r['classes'] if v['name']==c)
                i=r['seen_classes'].index(c);b=r['seen_classes'].index('benign');cm=r['confusion_matrix']
                details.append(f"{'XGB' if m=='xgb' else 'TabPFN'} F1 {100*v['f1']:.2f}% "
                    f"(P {100*v['precision']:.2f}% / R {100*v['recall']:.2f}%, "
                    f"공격→Benign {cm[i][b]:,}건 / Benign→공격 {cm[b][i]:,}건)")
            parts.append(f"{NAMES[c]}는 test {counts[c]:,}개, 최초 공급 {data['datasets'][ds]['allocation'][case][c]:,}개, "
                f"최종 context 잔존 {final['tabpfn']['fit_class_counts'][c]:,}개다. "+'; '.join(details)+'.')
        paragraphs.append(f"{title}의 평가 표본이 가장 적은 두 공격을 따로 보면, "+' '.join(parts)+
            ' 이 혼동 건수는 최종 예측에서 직접 확인한 관찰이며, 사례 수 부족이나 context 구성 중 무엇이 원인인지는 이 비교만으로 분리할 수 없다.')
    return paragraphs


def build_data(run, history_mode='bounded'):
    aggregate(run, require_complete=True)
    validation=json.loads((run/'validation.json').read_text())
    if validation.get('status')!='passed' or validation.get('summary_sha256')!=hashlib.sha256((run/'summary.csv').read_bytes()).hexdigest():
        raise ValueError('Validate the completed predictions before publishing the report')
    request=json.loads((run/'request.json').read_text())
    if request['history_mode']!=history_mode:raise ValueError('Unexpected history policy')
    source=Path(request['stream_root'])
    data=json.loads((run/'report_data.json').read_text())
    data.update(run=str(run.relative_to(ROOT)), request=request,
                prepared=json.loads((source/'prepared.json').read_text()),validation=validation,datasets={},summary=[])
    for ds in STAGES:
        meta=json.loads((source/'data'/ds/'manifest.json').read_text())
        data['datasets'][ds]=meta
        for case in CASES:
            pair={}
            for method in ['xgb','tabpfn']:
                rows=sorted([r for r in data['results'] if (r['dataset'],r['case'],r['method'])==(ds,case,method)],key=lambda r:r['stage'])
                if len(rows)!=len(STAGES[ds]):raise ValueError('Missing stage')
                if any(r['pfn_cold_fit'] for r in rows[1:]):raise ValueError('Resumed cold fit must be handled separately before cost comparison')
                if history_mode=='bounded' and any(r['history_rows']>request['history_cap'] for r in rows):raise ValueError('History cap exceeded')
                if history_mode=='cumulative' and any(r['context_rows']!=r['delivered_rows'] for r in rows):raise ValueError('Missing cumulative history')
                last=rows[-1]
                pair[method]=dict(initial_update_seconds=rows[0]['update_seconds'],
                    update_sum=sum(r['update_seconds'] for r in rows[1:]),
                    fit_sum=sum(r['fit_seconds'] for r in rows[1:]),
                    prediction_sum=sum(r['predict_seconds'] for r in rows),
                    final_macro_f1=last['macro_f1'],final_benign_fpr=last['benign_false_alarm_rate'],
                    final_context_rows=last['context_rows'],final_delivered_rows=last['delivered_rows'],
                    old_corrected_sum=sum(r['old_corrected'] for r in rows[1:]),
                    old_damaged_sum=sum(r['old_damaged'] for r in rows[1:]),
                    final_without_overlap_f1=last['without_context_vector_overlap']['macro_f1'])
            data['summary'].append(dict(dataset=ds,case=case,**pair,
                pfn_xgb_update_ratio=pair['tabpfn']['update_sum']/pair['xgb']['update_sum']))
    data['interpretation']=interpretation(data)
    return data


PAGE = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>새 공격 반영: 제한된 과거 사례와 갱신 비용</title>
<style>
:root{--bg:#f5f6f4;--paper:#fff;--ink:#172634;--muted:#566773;--line:#d9e0e2;--blue:#185c90;--teal:#168078;--soft:#eef4f7;--warn:#805b16}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.75 "Noto Sans CJK KR","Noto Sans KR",system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:42px 30px 80px}h1{font-size:32px;line-height:1.4;margin:10px 0 18px;letter-spacing:-.03em}h2{font-size:23px;margin:42px 0 14px}h3{font-size:17px;margin:24px 0 10px}p{margin:10px 0}p,li,h1,h2{word-break:keep-all;overflow-wrap:anywhere}.kicker{color:var(--blue);font-size:12px;letter-spacing:.06em}.lead{font-size:18px;max-width:980px}.note,small{color:var(--muted);font-size:12px}.pill{display:inline-block;border:1px solid var(--line);border-radius:16px;padding:2px 11px;margin:3px 5px 3px 0;font-size:12px;background:var(--paper)}nav{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0;border-bottom:1px solid var(--line);padding-bottom:20px}a{color:var(--blue)}.callout{background:var(--soft);border-left:3px solid var(--blue);padding:15px 20px;border-radius:2px;margin:18px 0}.table-wrap{overflow:auto;background:var(--paper);border:1px solid var(--line);border-radius:8px;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}th{background:var(--soft);color:#314c60;font-weight:600}th:first-child,td:first-child{text-align:left}td.left,th.left{text-align:left}tbody tr:last-child td{border-bottom:0}tr.new td:first-child{border-left:3px solid var(--teal)}.control{display:flex;gap:18px;flex-wrap:wrap;background:var(--paper);padding:18px;border:1px solid var(--line);border-radius:8px;position:sticky;top:0;z-index:1}label{display:flex;align-items:center;gap:9px;font-size:13px}select{padding:7px 10px;background:var(--paper);border:1px solid #a9b7c1;border-radius:5px;color:var(--ink);font:inherit}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0}.card{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:16px}.card strong{display:block;font-size:21px;line-height:1.5;font-variant-numeric:tabular-nums}.card span{font-size:12px;color:var(--muted)}.chart{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:12px;margin:14px 0}.chart svg{display:block;width:100%;max-height:280px}details{margin:14px 0}summary{cursor:pointer;color:var(--blue)}code{font-size:12px;overflow-wrap:anywhere}footer{border-top:1px solid var(--line);margin-top:36px;padding-top:18px}.gain{color:var(--teal)}.loss{color:#9a4141}@media(max-width:650px){main{padding:26px 15px 50px}h1{font-size:26px}.lead{font-size:16px}.cards{grid-template-columns:1fr}.control{position:static;gap:12px}table{font-size:12px}th,td{padding:8px 10px}}
</style></head><body><main>
<header><p class="kicker">2026-10-13 · EXP71 · v7 · 단일 모델 갱신 시나리오</p><h1>새 공격 반영:<br>제한된 과거 사례와 갱신 비용</h1>
<p class="lead">새로운 공격 사례를 반영할 때, <strong>과거 사례를 최대 4,096개만 유지하는 TabPFN context 갱신</strong>은 같은 사례로 다시 학습하는 XGBoost와 비교해 어떤 비용과 탐지 품질을 보이는가?</p>
<span class="pill">CIC2018 · ToN</span><span class="pill">Seed 43</span><span class="pill">균등 / 공격 내 자연 분포</span><span class="pill">두 모델 동일 학습·평가 IDs</span><span class="pill">Expert·S/V 제외</span>
<nav><a href="#finding">결과 요약</a><a href="#protocol">사례 공급·보존</a><a href="#stage">단계별 비용</a><a href="#classes">클래스별 성능</a><a href="#retention">기존 탐지 유지</a><a href="#interpretation">해석과 다음 판단</a></nav></header>
<section id="finding"><h2>1. 갱신 비용과 탐지 품질을 함께 판단</h2><div id="verdict" class="callout"></div>
<p>아래 시간은 <strong>초기 구축을 제외한 후속 갱신 시간의 합</strong>이다. 과거 선택·입력 구성, 모델 생성, 학습 또는 전처리·context 캐시 구성을 포함한다. TabPFN은 가중치를 재학습하지 않지만 새 context의 전처리·캐시를 다시 구성하므로 갱신 비용이 0은 아니다. 전체 test 추론 시간은 합산하지 않는다. 성능은 최종 도입 단계의 동일 test에서 측정했다.</p>
<div class="table-wrap"><table id="summary"><thead><tr><th>데이터 · 배정</th><th>XGB 갱신(초)</th><th>TabPFN 갱신(초)</th><th>TabPFN / XGB</th><th>XGB F1(%)</th><th>TabPFN F1(%)</th><th>XGB 정상 오탐(%)</th><th>TabPFN 정상 오탐(%)</th></tr></thead><tbody></tbody></table></div>
<h3>초기 구축과 전체 test 추론은 별도 비용</h3>
<div class="table-wrap"><table id="other-costs"><thead><tr><th>데이터 · 배정</th><th>XGB 초기(초)</th><th>TabPFN 초기(초)</th><th>XGB 추론 합계(초)</th><th>TabPFN 추론 합계(초)</th></tr></thead><tbody></tbody></table></div>
<p class="note">추론 합계는 각 단계의 도입된 클래스 전체 test를 반복 평가한 시간이다. 단일 운영 batch나 같은 수의 신규 트래픽을 뜻하지 않는다. 초기 비용은 모델 생성·첫 fit/cache를 포함하며, Python 프로세스 시작·데이터 사전 준비·배포 비용은 제외한다.</p>
<p class="note">비용은 각 단계 1회 측정한 경과 시간이다. 4개 조건은 서로 다른 데이터·구성이므로 통계적 반복 표본으로 해석하지 않는다. 단일 seed·기존 개발 holdout 결과이며, 낮은 비용만으로 유용한 갱신이라고 결론내리지 않는다.</p></section>
<section id="protocol"><h2>2. 무엇을 언제 추가하고, 무엇을 남겼는가</h2>
<p><strong>Case 1 — 공격 균등:</strong> 각 공격 1,024개를 첫 도입 단계에 공급한다. CIC2018 Web attack은 가용 train 450개로 제한한다. <strong>Case 2 — 공격 내 자연 분포:</strong> 같은 총 공격 예산을 정제 train의 공격별 빈도에 비례해 배분한다. Test 빈도는 사용하지 않는다.</p>
<p>두 case 모두 최종 공급량의 75%는 Benign이다. CIC2018은 초기 포함 매 단계 3,342개, ToN은 4,608개를 공급한다. Benign IDs와 공급 시점은 두 case가 같다. 최종 공급량은 CIC2018 22,280개, ToN 36,864개다. <strong>공격 내 자연 분포이며 전체 트래픽의 자연 분포 재현은 아니다.</strong></p>
<p>각 갱신은 직전 학습 집합에서 최대 4,096개를 남기고 그 단계의 신규 사례를 모두 더한다. 따라서 context 전체는 4,096개를 넘을 수 있다. 버린 사례는 나중에 다시 가져오지 않는다. 과거 집합도 Benign 목표 75%와 case별 공격 비율을 적용하되, 클래스 가용량을 넘지 않고 이미 도입한 클래스는 최소 1개를 유지한다.</p>
<p>원본에서 확인한 클래스 최초 출현 순서를 유지했다. CIC2018: Benign·Brute force·DoS → DDoS → Web attack → Infiltration → Bot. ToN: Benign·Scanning·DoS → Injection·DDoS → Password → XSS → Ransomware·Backdoor → MitM. 실제 timestamp 재생이나 환경 변화 실험은 아니다.</p>
<div id="allocations"></div>
<p class="note">배분 표의 정렬은 데이터셋별 고정 test 표본 수 내림차순이다. 배정 수는 실험 전체에 공급되는 서로 다른 행 수이며 최종 context의 잔존 수와 다르다. 최소 1개 보존으로 자연 비율이 바뀐 경우는 아래 단계별 표에 표시한다.</p></section>
<section id="stage"><h2>3. 단계별 갱신 비용</h2><div class="control"><label>데이터 <select id="dataset"><option value="cic2018">CIC2018</option><option value="toniot">ToN</option></select></label><label>배정 <select id="case"><option value="uniform">Case 1 · 공격 균등</option><option value="natural">Case 2 · 공격 내 자연 분포</option></select></label><label>단계 <select id="stage-select"></select></label></div>
<div class="cards" id="cards"></div><div class="chart" id="chart"></div>
<div class="table-wrap"><table id="stages"><thead><tr><th>단계 · 새 공격</th><th>신규 행</th><th>보존 행</th><th>Context 행</th><th>XGB 갱신(초)</th><th>TabPFN 갱신(초)</th><th>XGB 신규 공격 F1(%)</th><th>TabPFN 신규 공격 F1(%)</th></tr></thead><tbody></tbody></table></div>
<p class="note">신규 공격 F1은 그 단계에 새로 도입된 공격의 클래스별 F1 평균이다. Precision의 FP는 Benign·기존 공격을 포함한 전체 평가 집합에서 계산한다. 초기 단계는 신규 공격 지표를 정의하지 않는다. 누적 비용 그래프는 초기 구축을 제외한다.</p>
<details><summary>선택한 단계의 비용 분해·추론·최소 보존 보정</summary><div class="table-wrap"><table id="cost-detail"><thead><tr><th>항목</th><th>XGB</th><th>TabPFN</th></tr></thead><tbody></tbody></table></div><p id="memory-note" class="note"></p><p class="note">첫 추론 지연은 최대 32,768행인 첫 batch의 경과 시간이며 단일 행 latency가 아니다. 전체 추론 시간은 test 배치와 결과 기록을 포함한다. 운영 배포·모델 직렬화는 이 실험에서 측정하지 않았다.</p></details></section>
<section id="classes"><h2>4. 선택한 단계의 클래스별 성능과 context</h2><p>동일한 평가 표본에서 Precision·Recall·F1을 비교한다. 클래스는 <strong>해당 단계 test support 내림차순</strong>으로 정렬한다. 왼쪽 표시선은 이번에 새로 도입한 공격이다.</p>
<div class="table-wrap"><table id="class-table"><thead><tr><th>클래스</th><th>Test 행</th><th>이번 유입</th><th>Context 잔존</th><th>XGB P(%)</th><th>XGB R(%)</th><th>XGB F1(%)</th><th>TabPFN P(%)</th><th>TabPFN R(%)</th><th>TabPFN F1(%)</th><th>ΔF1(%p)</th></tr></thead><tbody></tbody></table></div>
<details><summary>Context와 같은 feature vector가 있는 test 행을 제외한 보조 평가</summary><p id="overlap-note"></p></details></section>
<section id="retention"><h2>5. 기존 공격·정상 탐지가 유지됐는가</h2><p>갱신 전후 <strong>동일한 기존 클래스의 test 행</strong>만 비교한다. 신규 클래스 정답 행은 제외하지만 기존 행을 신규 공격으로 오분류한 경우는 오류로 남긴다. 클래스가 추가되면서 평균의 분모가 바뀌는 효과와 구별한다.</p>
<div class="table-wrap"><table id="retention-table"><thead><tr><th>갱신 단계</th><th>XGB 기존 F1 전(%)</th><th>XGB 후(%)</th><th>TabPFN 기존 F1 전(%)</th><th>TabPFN 후(%)</th><th>XGB 교정</th><th>XGB 훼손</th><th>TabPFN 교정</th><th>TabPFN 훼손</th></tr></thead><tbody></tbody></table></div><p class="note">교정은 이전 오답→현재 정답, 훼손은 이전 정답→현재 오답이다. 같은 행이 여러 단계에서 반복 평가되므로 단계별 건수를 고유 표본 수로 합산하지 않는다.</p></section>
<section id="interpretation"><h2>6. 이 결과로 판단할 것과 다음 비교</h2><div id="interpretation-text"></div>
<p>이 실험의 비용 차이는 같은 단계·같은 학습 사례에서의 모델 갱신 방식 비교다. 두 case 간에는 공격별 확보 사례 수, 단계별 유입량과 실제 context 크기가 달라진다. 따라서 균등/자연 배정의 성능 차이를 분포 하나의 인과 효과로 단정하지 않는다.</p>
<p>자연 배정에서 확보 라벨이 아주 적은 공격은 그 수와 오탐을 함께 확인해야 한다. 개선 판단은 새 공격의 탐지 품질, 기존 클래스 유지와 정상 오탐을 기준으로 한다. 다음 누적 전체 조건에서는 <strong>이번과 동일한 공급 stream IDs</strong>를 사용하고 과거 사례를 버리지 않아, 보존 상한이 비용·탐지 유지에 미친 영향을 비교한다. 해당 조건은 아직 실행하지 않았다.</p>
<p>현재는 하나의 seed와 반복 관찰한 개발 holdout을 사용했다. 가까운 갱신 시간의 차이는 반복 측정으로 확인해야 하며, TabPFN 전체 시스템이나 expert·S/V의 기여로 일반화하지 않는다. 후속 판단은 작은 라벨·context 조건에서 유용한 탐지를 유지하는 갱신 범위를 찾는 데 둔다.</p></section>
<footer><p><strong>실행 근거</strong> <code id="run-path"></code></p><p id="source-links"></p><p class="note">TabPFN: v3_20260417_multiclass, ensemble 4, frozen backbone, fit_with_cache, CPU cache. XGB: 300 rounds, depth 8, lr 0.05, CUDA hist. RTX 4090, GPU worker 순차 실행. 준비·실험 코드와 checkpoint·입력 hash를 실행 폴더에 보존했다.</p></footer>
</main><script id="report-data" type="application/json">__DATA__</script><script>
'use strict';
const DATA=JSON.parse(document.getElementById('report-data').textContent);
const DN={cic2018:'CIC2018',toniot:'ToN'}, CN={uniform:'균등',natural:'공격 내 자연'};
const N={benign:'Benign',brute_force:'Brute force',dos:'DoS',ddos:'DDoS',web_attacks:'Web attack',infiltration:'Infiltration',bot:'Bot',scanning:'Scanning',injection:'Injection',password:'Password',xss:'XSS',ransomware:'Ransomware',backdoor:'Backdoor',mitm:'MitM'};
const el=id=>document.getElementById(id), num=x=>Number(x).toLocaleString('en-US'), dec=x=>x==null?'—':Number(x).toFixed(2), pct=x=>x==null?'—':dec(x*100), rate=x=>x==null?'—':Number(x*100).toFixed(4);
const cell=(v,cls='')=>`<td class="${cls}">${v}</td>`, row=c=>`<tr>${c.map(x=>cell(x)).join('')}</tr>`;
const selected=()=>({ds:el('dataset').value,c:el('case').value,s:Number(el('stage-select').value)});
const records=(ds,c,m)=>DATA.results.filter(r=>r.dataset===ds&&r.case===c&&r.method===m).sort((a,b)=>a.stage-b.stage);
function setup(){
 el('summary').tBodies[0].innerHTML=DATA.summary.map(s=>row([DN[s.dataset]+' · '+CN[s.case],dec(s.xgb.update_sum),dec(s.tabpfn.update_sum),dec(s.pfn_xgb_update_ratio)+'×',pct(s.xgb.final_macro_f1),pct(s.tabpfn.final_macro_f1),rate(s.xgb.final_benign_fpr),rate(s.tabpfn.final_benign_fpr)])).join('');
 el('other-costs').tBodies[0].innerHTML=DATA.summary.map(s=>row([DN[s.dataset]+' · '+CN[s.case],dec(s.xgb.initial_update_seconds),dec(s.tabpfn.initial_update_seconds),dec(s.xgb.prediction_sum),dec(s.tabpfn.prediction_sum)])).join('');
 let faster=DATA.summary.filter(s=>s.pfn_xgb_update_ratio<1);
 const cu=DATA.summary.find(s=>s.dataset==='cic2018'&&s.case==='uniform');
 const tu=DATA.summary.find(s=>s.dataset==='toniot'&&s.case==='uniform'),tn=DATA.summary.find(s=>s.dataset==='toniot'&&s.case==='natural');
 el('verdict').textContent=`4개 조건 중 ${faster.length}개에서 TabPFN의 후속 갱신 시간 합계가 더 짧았다. 최종 Macro-F1의 TabPFN−XGB 차이는 CIC2018 균등에서 ${(100*(cu.tabpfn.final_macro_f1-cu.xgb.final_macro_f1)).toFixed(3)}%p, ToN 균등·자연에서 각각 ${dec(100*(tu.tabpfn.final_macro_f1-tu.xgb.final_macro_f1))}%p·${dec(100*(tn.tabpfn.final_macro_f1-tn.xgb.final_macro_f1))}%p였다. 비용 절감과 탐지 품질이 함께 유지되는지 확인하려면, 다음에는 같은 사례를 누적 보존한 결과와 비교해야 한다.`;
 el('allocations').innerHTML=Object.entries(DATA.datasets).map(([ds,d])=>`<h3>${DN[ds]} · 전체 공급 배분</h3><div class="table-wrap"><table><thead><tr><th>클래스</th><th>정제 train 가용</th><th>고정 test</th><th>균등 공급</th><th>자연 공급</th></tr></thead><tbody>${Object.keys(d.fixed_test_class_counts).sort((a,b)=>d.fixed_test_class_counts[b]-d.fixed_test_class_counts[a]).map(c=>row([N[c],num(d.available_train[c]),num(d.fixed_test_class_counts[c]),num(d.allocation.uniform[c]),num(d.allocation.natural[c])])).join('')}</tbody></table></div>`).join('');
 for(const text of DATA.interpretation){const p=document.createElement('p');p.textContent=text;el('interpretation-text').appendChild(p);}
 el('run-path').textContent=DATA.run;
 const sources=[['단계별 CSV',DATA.run+'/summary.csv'],['클래스별 CSV',DATA.run+'/class_metrics.csv'],['전체 수치 JSON',DATA.run+'/report_data.json'],['실험 설정','docs/research/20261013/exp71_protocol.md'],['연구 해석 기록','docs/research/20261013/exp71_results.md']];
 el('source-links').innerHTML=sources.map(([label,path])=>location.protocol==='data:'?`${label}: <code>${path}</code>`:`<a href="../../${path}">${label}</a>`).join(location.protocol==='data:'?'<br>':' · ');
 resetStages();
}
function resetStages(){let {ds,c}=selected();const rr=records(ds,c,'xgb');el('stage-select').innerHTML=rr.map(r=>`<option value="${r.stage}">${r.stage===0?'초기':r.stage+' · '+r.new_classes.map(x=>N[x]).join(' / ')}</option>`).join('');el('stage-select').value=rr.at(-1).stage;render();}
function plot(xs,ps){
 const w=940,h=220,left=65,right=32,top=20,bottom=40;
 let xc=[0],pc=[0];for(let i=1;i<xs.length;i++){xc.push(xc.at(-1)+xs[i].update_seconds);pc.push(pc.at(-1)+ps[i].update_seconds);}
 const ymax=Math.max(...xc,...pc)*1.18||1, xx=i=>left+i*(w-left-right)/(xs.length-1), yy=v=>h-bottom-v*(h-top-bottom)/ymax;
 let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="초기 제외 누적 갱신 시간"><text x="${left}" y="13" font-size="12" fill="#566773">누적 갱신 시간(초) · 초기 제외</text>`;
 for(let i=0;i<=4;i++){const v=ymax*i/4;svg+=`<line x1="${left}" x2="${w-right}" y1="${yy(v)}" y2="${yy(v)}" stroke="#e0e6e9"/><text x="${left-9}" y="${yy(v)+4}" text-anchor="end" font-size="11">${v.toFixed(1)}</text>`;}
 for(const [v,color,name] of [[xc,'#185c90','XGB'],[pc,'#168078','TabPFN']]){svg+=`<polyline fill="none" stroke="${color}" stroke-width="2.5" points="${v.map((n,i)=>xx(i)+','+yy(n)).join(' ')}"/>`;v.forEach((n,i)=>svg+=`<circle cx="${xx(i)}" cy="${yy(n)}" r="3.5" fill="${color}"><title>${name} 단계 ${i}: ${n.toFixed(3)}초</title></circle>`);}
 xs.forEach((r,i)=>svg+=`<text x="${xx(i)}" y="${h-15}" text-anchor="middle" font-size="12">${i===0?'초기':i+'단계'}</text>`);
 svg+=`<text x="${w-220}" y="15" fill="#185c90" font-size="12">● XGB</text><text x="${w-145}" y="15" fill="#168078" font-size="12">● TabPFN</text></svg>`;el('chart').innerHTML=svg;
}
function render(){
 const {ds,c,s}=selected(),xs=records(ds,c,'xgb'),ps=records(ds,c,'tabpfn'),x=xs[s],p=ps[s];if(!x||!p)throw new Error('Missing paired stage');
 el('cards').innerHTML=`<div class="card"><span>이번 입력 = 과거 보존 + 신규 유입</span><strong>${num(p.history_rows)} + ${num(p.incoming_rows)} = ${num(p.context_rows)}</strong></div><div class="card"><span>갱신 시간 · XGB / TabPFN</span><strong>${dec(x.update_seconds)} / ${dec(p.update_seconds)}초</strong></div><div class="card"><span>이번 단계 전체 test 표본 수</span><strong>${num(p.rows)}</strong></div>`;
 el('stages').tBodies[0].innerHTML=xs.map((r,i)=>row([i===0?'초기':i+' · '+r.new_classes.map(n=>N[n]).join(' / '),num(r.incoming_rows),num(r.history_rows),num(r.context_rows),dec(r.update_seconds),dec(ps[i].update_seconds),pct(r.new_class_macro_f1),pct(ps[i].new_class_macro_f1)])).join('');
 plot(xs,ps);
 const fields=[['선택·입력 구성(초)','select_seconds'],['모델 생성(초)','construct_seconds'],['학습 / 전처리·캐시(초)','fit_seconds'],['갱신 합계(초)','update_seconds'],['전체 test 추론(초)','predict_seconds'],['첫 batch 추론(초)','first_batch_seconds']];
 el('cost-detail').tBodies[0].innerHTML=fields.map(([title,k])=>row([title,dec(x[k]),dec(p[k])])).join('');
 el('memory-note').textContent='최소 1개 보존으로 배분이 증가한 클래스: '+(p.memory.floor_adjusted_classes.map(n=>N[n]).join(', ')||'없음')+'. 과거 보존 수는 직전 학습 집합의 가용량으로 제한된다.';
 const xm=Object.fromEntries(x.classes.map(v=>[v.name,v]));
 el('class-table').tBodies[0].innerHTML=[...p.classes].sort((a,b)=>b.support-a.support).map(q=>{const a=xm[q.name],delta=(q.f1-a.f1)*100;return `<tr class="${p.new_classes.includes(q.name)?'new':''}">${[N[q.name],num(q.support),num(p.incoming_class_counts[q.name]),num(p.fit_class_counts[q.name]),pct(a.precision),pct(a.recall),pct(a.f1),pct(q.precision),pct(q.recall),pct(q.f1)].map(v=>cell(v)).join('')}${cell((delta>0?'+':'')+dec(delta),delta>0?'gain':delta<0?'loss':'')}</tr>`;}).join('');
 el('overlap-note').textContent=`Context와 동일 feature인 test ${num(p.exact_feature_overlap_test_rows)}개를 제외한 ${num(p.without_context_vector_overlap.rows)}개에서 Macro-F1은 XGB ${pct(x.without_context_vector_overlap.macro_f1)}%, TabPFN ${pct(p.without_context_vector_overlap.macro_f1)}%다. 주 평가의 전체 test를 대체하지 않는 보조 결과다.`;
 el('retention-table').tBodies[0].innerHTML=xs.slice(1).map(r=>{const b=ps[r.stage];return row([r.stage+' · '+r.new_classes.map(n=>N[n]).join(' / '),pct(r.retention_before_macro_f1),pct(r.retention_after_macro_f1),pct(b.retention_before_macro_f1),pct(b.retention_after_macro_f1),num(r.old_corrected),num(r.old_damaged),num(b.old_corrected),num(b.old_damaged)]);}).join('');
}
el('dataset').addEventListener('change',resetStages);el('case').addEventListener('change',resetStages);el('stage-select').addEventListener('change',render);setup();
</script></body></html>'''


def notes(data):
    lines=['# EXP71 · 제한된 과거 사례의 새 공격 반영 결과','',
           '2026-10-13. Seed 43, CIC2018·ToN, 균등/공격 내 자연 분포, 단일 TabPFN·XGB. 두 모델은 각 단계에서 동일한 학습·평가 IDs를 사용했다. 초기 구축을 제외한 갱신 시간을 중심으로 비교하며, 추론은 별도 기록한다. Expert/S/V 및 누적 전체 조건은 실행하지 않았다.','',
           '공급 라벨 전체를 모델에 계속 누적하지 않고 직전 학습 집합에서 최대 4,096행을 보존한 뒤 신규 사례를 더했다. 균등과 자연 배정은 최종 공격 총량과 Benign 공급을 공유한다. 자연 배정은 정제 train의 공격끼리의 비율이며, 적은 클래스의 보존에 최소 1행 조건을 적용했다. 이 조건의 클래스별 보정은 원본 결과에 남겼다.','']
    for s in data['summary']:
        name=('CIC2018' if s['dataset']=='cic2018' else 'ToN')+' / '+('균등' if s['case']=='uniform' else '공격 내 자연')
        x,p=s['xgb'],s['tabpfn']
        lines.append(f"{name}에서 후속 갱신 시간 합계는 XGB {x['update_sum']:.3f}초, TabPFN {p['update_sum']:.3f}초로, TabPFN/XGB 비율은 {s['pfn_xgb_update_ratio']:.3f}다. 최종 Macro-F1은 {100*x['final_macro_f1']:.2f}%와 {100*p['final_macro_f1']:.2f}%, Benign 오탐률은 {100*x['final_benign_fpr']:.4f}%와 {100*p['final_benign_fpr']:.4f}%다. 전체 {p['final_delivered_rows']:,}행을 공급했고 마지막 갱신에는 {p['final_context_rows']:,}행을 사용했다. 비용과 탐지 품질을 함께 판단해야 하며 단일 수치로 우위를 확정하지 않는다.")
        lines.append('')
    lines.extend(['## 소표본 공격의 관찰',''])
    for paragraph in data['interpretation'][1::2]:lines.extend([paragraph,''])
    lines += ['관찰의 해석은 같은 단계의 같은 사례에서 두 갱신 방식이 보인 비용·예측 차이로 한정한다. 균등/자연 case 간에는 공격별 확보 라벨, 유입량과 실제 context 크기도 달라진다. 이를 분포만의 인과 효과로 설명하지 않는다. 특정 공격의 재현율 상승은 정상 및 다른 공격에서 유입된 FP와 함께 확인한다. 기존 클래스 유지 수치는 신규 정답 행을 제외한 고정 과거 표본군에서 계산하며 새 공격으로의 오분류는 오류에 포함한다.','',
              '갱신 시간은 메모리에서 입력 선택·모델 구성·fit/cache가 완료되기까지의 경과 시간이며 운영 배포·모델 직렬화를 측정하지 않았다. 첫 batch와 전체 test 추론 비용은 별도 지표다. 단일 seed·기존 개발 holdout·단계별 1회 timing이라는 범위에서 해석한다.','',
              '후속 판단은 같은 공급 IDs를 누적 전체 조건으로 재생하여 과거 보존 상한이 탐지 유지와 비용에 미친 영향을 비교하는 것이다. 이번 결과만으로 bounded 방식의 효과나 TabPFN 전체의 비용 우위를 일반화하지 않는다. 누적 전체 실행은 사용자의 결과 검토 후 진행한다.','',
              '## 출처','',f"- 실행: `{data['run']}`",f"- 단계별 수치: `{data['run']}/summary.csv`",f"- 클래스별 수치: `{data['run']}/class_metrics.csv`",f"- 확률·예측·행 ID·혼동행렬: `{data['run']}/jobs/`",'- 설계: `docs/research/20261013/exp71_protocol.md`',f'- 보고서: `lablog/html_report/{REPORT_ID}.html`','']
    return '\n'.join(lines)


def integrate(page, data):
    text=page.read_text()
    match=re.search(r'const REPORTS = (.*?);\n',text)
    reports=json.loads(match.group(1))
    title='새 공격 도입: TabPFN의 갱신 비용과 탐지 성능' if 'cumulative' in data else TITLE
    entry=dict(id=REPORT_ID,date='2026-10-13',titleKr=title,
               titleEn='Attack arrival · Update / inference cost · Detection quality' if 'cumulative' in data else 'Bounded history · Uniform / natural attack allocation',
               verdict='단일 모델 · 갱신 비용·클래스별 성능',verdictClass='neutral',group='report',type='iframe',b64='')
    reports=[r for r in reports if r['id']!=REPORT_ID];reports.insert(1,entry)
    text=text[:match.start(1)]+json.dumps(reports,ensure_ascii=False)+text[match.end(1):]
    text=text.replace('8&ndash;9월의 보고서 9편', '8&ndash;10월의 보고서 10편')
    text=re.sub(r'<p class="asof">.*?</p>', '<p class="asof">2026-08-18 &ndash; 2026-10-13 &middot; TabPFN의 context·expert 연구와 데이터 품질 감사, 새 공격 반영의 갱신 비용 검증.</p>',text,count=1)
    lead='<p class="lede">새 공격이 추가될 때 제한된 과거 사례로 갱신하는 단일 TabPFN과 XGBoost를 비교했다. <strong>균등·공격 내 자연 배정에서 초기 구축과 후속 갱신, 클래스별 성능과 기존 탐지 유지를 구분한다.</strong> <button class="jump" data-jump="'+REPORT_ID+'">10/13 갱신 비용·탐지 품질 결과</button>. 기존 expert·S/V 결과는 09/30 탭에 보존했다.</p>'
    text=re.sub(r'<p class="lede">.*?</p>',lambda _:lead,text,count=1)
    block='<!-- EXP71_FLOW_START --><div class="flow-step"><time>10-13</time><div class="flow-rail-dotcol"><div class="d"></div><div class="l"></div></div><div class="body"><button class="jump" data-jump="'+REPORT_ID+'"><h3>'+TITLE+'</h3></button><p>과거 최대 4,096개 + 신규 정상·공격 사례. CIC2018·ToN, 균등/공격 내 자연 분포, seed 43. 두 모델의 갱신 비용·클래스별 성능·기존 탐지 유지를 함께 확인한다.</p></div></div><!-- EXP71_FLOW_END -->'
    text=re.sub(r'<!-- EXP71_FLOW_START -->.*?<!-- EXP71_FLOW_END -->','',text,flags=re.S)
    text=text.replace('<div class="flow-rail">','<div class="flow-rail">\n'+block,1)
    faster=sum(s['pfn_xgb_update_ratio']<1 for s in data['summary'])
    card='<!-- EXP71_CARD_START --><div class="status-cell"><p class="label">10/13 · 제한된 과거 갱신 완료</p><p class="value">4개 구성 중 '+str(faster)+'개에서 TabPFN 후속 갱신 시간 합계가 더 짧았다. 같은 사례의 클래스별 성능·정상 오탐·기존 탐지 유지까지 확인한다. <button class="jump" data-jump="'+REPORT_ID+'">새 결과 보기</button>.</p></div><!-- EXP71_CARD_END -->'
    text=re.sub(r'<!-- EXP71_CARD_START -->.*?<!-- EXP71_CARD_END -->','',text,flags=re.S)
    text=text.replace('<div class="status-grid">','<div class="status-grid">\n'+card,1)
    if 'cumulative' in data:
        text=text.replace(TITLE,title)
        text=text.replace('새 공격이 추가될 때 제한된 과거 사례로 갱신하는 단일 TabPFN과 XGBoost를 비교했다.',
            '원본 클래스의 출현 순서대로 새 공격을 도입하고, 공급 배정과 과거 보존 방식을 바꿔 단일 TabPFN·XGBoost의 갱신 비용·추론 비용·탐지 성능을 비교했다.')
        text=text.replace('과거 최대 4,096개 + 신규 정상·공격 사례.', '과거 최대 4,096개 보존과 누적 전체를 같은 신규 공급 사례로 비교.')
        text=text.replace('10/13 · 제한된 과거 갱신 완료','10/13 · 제한·누적 비교 완료')
        start=text.index('<!-- EXP71_CARD_START -->');end=text.index('<!-- EXP71_CARD_END -->',start)
        newcard='<div class="status-cell"><p class="label">10/13 · 새 공격 도입 시나리오</p><p class="value">제한 보존의 일부 조건에서 갱신 비용 이점을 확인했다. 누적 전체·대량 추론·ToN tail에서는 같은 이점이 유지되지 않았다. 시나리오 → 비용 → 성능 → TabPFN 채택 판단 순서로 정리했다. <button class="jump" data-jump="'+REPORT_ID+'">결과와 판단 보기</button>.</p></div>'
        text=text[:start]+'<!-- EXP71_CARD_START -->'+newcard+text[end:]
    page.write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--cumulative',type=Path)
    args=parser.parse_args();run=args.run.resolve();data=build_data(run)
    page=PAGE;snapshot_run=run
    if args.cumulative:
        from exp71_comparison_report import comparison_text
        from exp71_scenario_report import make_page, notes as comparison_notes
        cumulative=args.cumulative.resolve();data['cumulative']=build_data(cumulative,'cumulative')
        verified=json.loads((cumulative/'comparison_validation.json').read_text())
        if verified['status']!='passed' or Path(verified['reference']).resolve()!=run:raise ValueError('Unverified reference run')
        for name,path in [('reference_summary_sha256',run),('current_summary_sha256',cumulative)]:
            if verified[name]!=hashlib.sha256((path/'summary.csv').read_bytes()).hexdigest():raise ValueError('Compared results changed')
        data['policy_comparison']=verified;data['comparison_text']=comparison_text(data)
        page=make_page(data,page);snapshot_run=cumulative
    report=ROOT/'lablog/html_report'/f'{REPORT_ID}.html'
    report.write_text(apply_rules(page.replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/'))))
    record=ROOT/'docs/research/20261013'/('exp71_cumulative_results.md' if args.cumulative else 'exp71_results.md')
    record.write_text(comparison_notes(data) if args.cumulative else notes(data))
    integrate(ROOT/'lablog/html_report/post_tabpfn.html',data)
    (snapshot_run/'report_snapshot.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Built',report.relative_to(ROOT),'and',record.relative_to(ROOT))


if __name__=='__main__':main()
