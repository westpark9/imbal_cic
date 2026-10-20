"""Present EXP71 by scenario, cost, detection quality, and adoption evidence."""
import re
from exp71_comparison_report import environment_note


TITLE = '새 공격 도입: TabPFN의 갱신 비용과 탐지 성능'

BODY = r'''
<header><p class="kicker">2026-10-13 실험 · 10-14 구성 개편 · EXP71 · Seed 43</p>
<h1>새 공격의 라벨 사례가 확보되면,<br>TabPFN의 context 갱신이 유리한가?</h1>
<p class="lead">원본 데이터에서 확인한 <strong>공격 클래스의 최초 출현 순서</strong>에 따라 라벨 사례를 추가했다. 같은 사례로 갱신하는 XGBoost와 TabPFN을 <strong>갱신 비용 → 추론 비용 → 탐지 성능</strong> 순서로 비교한다.</p>
<nav><a href="#scenario">1. 시나리오와 변화 조건</a><a href="#cost">2. 갱신·추론 비용</a><a href="#performance">3. 탐지 성능</a><a href="#conclusion">4. 채택에 대한 판단</a></nav></header>

<section id="scenario"><h2>1. 새 공격을 어떤 순서와 구성으로 추가했는가</h2>
<p>초기에는 Benign과 두 공격 클래스로 시작한다. 다음 단계마다 <strong>새 공격의 라벨 사례와 Benign 사례를 함께 공급</strong>하고, 그때까지 도입된 클래스의 고정 test 행에서 두 모델을 평가한다. 새 클래스의 라벨 사례는 해당 클래스의 평가 전에 제공한다.</p>
<div id="timelines"></div>
<p class="note">초기 구간과 일부 인접 클래스를 묶되 최초 출현의 상대적 순서를 유지했다. 원본 timestamp에 따른 트래픽 재생이나 미관측 공격의 라벨 없는 탐지를 평가한 것은 아니다.</p>
<div class="model-grid"><article class="model-card xgb"><h3>XGBoost</h3><p>이번에 사용할 과거·신규 사례를 모아<br><strong>분류기를 새로 학습</strong></p></article><article class="model-card pfn"><h3>TabPFN</h3><p>동일한 사례로 context를 바꾸고<br><strong>전처리·추론 캐시를 재구성</strong><br><small>사전학습 가중치는 고정</small></p></article></div>
<h3>두 축을 바꿨다: 공격별 공급량 × 과거 사례 보존</h3>
<div class="table-wrap"><table id="design-table"><thead><tr><th>변화 조건</th><th class="left">설정 1</th><th class="left">설정 2</th></tr></thead><tbody>
<tr><td>공격별 공급량</td><td class="left"><strong>공격 균등</strong><br>각 공격 1,024개. CIC2018 Web attack은 가용량 450개.</td><td class="left"><strong>공격 내 자연 비율</strong><br>같은 총 공격 예산을 정제 train의 공격별 빈도에 비례 배분.</td></tr>
<tr><td>과거 사례 보존</td><td class="left"><strong>제한 보존</strong><br>직전 학습 집합에서 최대 4,096개를 남긴 뒤 신규 사례를 모두 추가.</td><td class="left"><strong>누적 전체</strong><br>도입 시점까지 공급된 모든 과거·신규 사례를 사용.</td></tr>
</tbody></table></div>
<p>Benign은 전체 공급량의 75%로 정하고 매 단계에 나눠 넣었다. CIC2018은 매 단계 3,342개, ToN은 4,608개다. 최종 공급량은 각각 <strong>22,280개 / 36,864개</strong>이며, 제한 보존의 실제 입력량은 이보다 작다. 두 데이터 × 두 공급 배정 × 두 보존 방식 = <strong>8개 구성</strong>을 두 모델에서 비교했다.</p>
<p class="note">공격 내 자연 비율은 전체 네트워크 트래픽의 자연 분포를 뜻하지 않는다. 두 모델은 각 구성에서 동일한 학습·test IDs를 사용한다. 보존 방식끼리는 신규 공급 IDs와 test를 고정했으며, 균등/자연 배정끼리는 공격별 공급 수와 단계별 입력 크기가 함께 달라진다.</p>
<details><summary>클래스별 공급량·test 표본 수와 최초 출현일</summary><div id="allocations"></div><p class="note">표는 해당 데이터셋의 전체 test 표본 수 내림차순. 공급량은 실험 전체에 제공한 서로 다른 행 수이며 마지막 context 잔존 수와 구별한다. 최초 출현일은 원본 기록의 UTC 기준이다.</p></details></section>

<div class="control" id="explore-controls" aria-label="비용과 성능 상세 조건">
<label>데이터 <select id="dataset"><option value="cic2018">CIC2018</option><option value="toniot">ToN</option></select></label>
<label>공급 <select id="case"><option value="uniform">공격 균등</option><option value="natural">공격 내 자연 비율</option></select></label>
<label>보존 <select id="policy"><option value="bounded">제한 보존</option><option value="cumulative">누적 전체</option></select></label>
<label>상세 단계 <select id="stage-select"></select></label></div>

<section id="cost"><h2>2. 갱신이 빠른가, 추론까지 효율적인가</h2>
<p><strong>갱신</strong>은 새 라벨 사례를 반영해 다음 예측을 준비하는 비용이고, <strong>추론</strong>은 준비된 모델로 test 행을 예측하는 비용이다. TabPFN이 가중치를 재학습하지 않는다는 사실이 두 비용 모두 낮다는 뜻은 아니다.</p>
<div class="callout" id="cost-verdict"></div>
<div class="legend" aria-label="모델 범례"><span class="xgb-label">■ ┄ XGBoost · 주황 / 점선</span><span class="pfn-label">● ━ TabPFN · 파랑 / 실선</span></div>
<p id="selection-caption" class="selection-caption"></p>
<div class="chart-grid"><div><h3>후속 갱신 시간의 누적</h3><div id="chart" class="chart"></div></div><div><h3>선택한 단계의 test 추론</h3><div id="inference-chart" class="chart"></div></div></div>
<p class="note" id="inference-caption"></p>
<h3>8개 구성의 비용 비교</h3><p>갱신 열은 <strong>초기 구축을 제외한 후속 단계의 합</strong>, 추론 열은 <strong>최종 단계의 전체 test 1회</strong>다. 두 측정 범위가 다르므로 이 표의 열을 더해 운영 총비용으로 부르지 않는다.</p>
<div class="table-wrap"><table id="cost-summary"><thead><tr><th>데이터 · 공급 · 보존</th><th>최종 입력</th><th class="xgb-label">XGB 갱신(초)</th><th class="pfn-label">TabPFN 갱신(초)</th><th class="xgb-label">XGB 추론(초)</th><th class="pfn-label">TabPFN 추론(초)</th><th>최종 test 행</th></tr></thead><tbody></tbody></table></div>
<details><summary>선택한 구성의 초기 구축·단계별 비용과 입력량</summary><div class="table-wrap"><table id="stages"><thead><tr><th>단계 · 새 공격</th><th>신규 사례</th><th>과거 보존</th><th>전체 입력</th><th>XGB 갱신(초)</th><th>TabPFN 갱신(초)</th><th>XGB 추론(초)</th><th>TabPFN 추론(초)</th></tr></thead><tbody></tbody></table></div><div class="table-wrap"><table id="cost-detail"><thead><tr><th>선택 단계의 항목</th><th>XGB</th><th>TabPFN</th></tr></thead><tbody></tbody></table></div><p class="note">갱신 = 입력 선택·구성 + 모델 생성 + 학습 또는 전처리·캐시 구성. 초기 행은 첫 모델 구축 비용을 포함한다. 프로세스 시작·데이터 사전 준비·운영 배포·직렬화는 미측정이다. 추론은 배치 예측과 결과 기록을 포함하며, 첫 batch는 최대 32,768행이므로 단일 행 지연 시간이 아니다.</p></details>
<details><summary>측정 환경과 재개 이력</summary><p id="environment-note"></p><p class="note">RTX 4090, GPU 모델 worker 순차 실행. 단계별 1회 timing·단일 seed이므로 근접한 차이는 반복 측정이 필요하다.</p></details></section>

<section id="performance"><h2>3. 새 공격을 잘 탐지하고 기존 탐지를 유지했는가</h2>
<div id="performance-verdict" class="callout"></div>
<p>아래 요약은 모든 공격이 도입된 <strong>최종 단계의 같은 test</strong>를 비교한다. 전체 Macro-F1과 Benign 오탐률을 함께 보고, 소수 공격은 클래스별 Precision·Recall·F1로 확인한다.</p>
<div class="table-wrap"><table id="performance-summary"><thead><tr><th>데이터 · 공급 · 보존</th><th class="xgb-label">XGB F1(%)</th><th class="pfn-label">TabPFN F1(%)</th><th>TabPFN−XGB(%p)</th><th>XGB 정상 오탐(%)</th><th>TabPFN 정상 오탐(%)</th></tr></thead><tbody></tbody></table></div>
<div class="chart-grid"><div><h3>선택한 구성의 신규 공격 F1</h3><div id="new-attack-chart" class="chart"></div></div><div class="finding-box"><h3>성능 차이를 읽는 기준</h3><p>각 단계의 신규 공격만 평균하되, Precision의 FP는 정상·기존 공격을 포함한 그 단계 전체 test에서 계산한다.</p><p>단계마다 평가 공격이 바뀌므로 선의 상승·하락을 같은 공격의 학습 곡선으로 해석하지 않는다. 기존 탐지 유지 여부는 아래의 동일 기존 test 행 비교로 확인한다.</p></div></div>
<h3 id="classes">선택한 단계의 클래스별 성능</h3><p id="class-caption"></p>
<div class="table-wrap"><table id="class-table"><thead><tr><th>클래스</th><th>Test 행</th><th>이번 유입</th><th>Context 잔존</th><th>XGB P(%)</th><th>XGB R(%)</th><th>XGB F1(%)</th><th>TabPFN P(%)</th><th>TabPFN R(%)</th><th>TabPFN F1(%)</th><th>ΔF1(%p)</th></tr></thead><tbody></tbody></table></div>
<p class="note">해당 단계 test 표본 수 내림차순. 왼쪽 표시는 이번에 새로 도입한 공격이며, 색은 모델을 의미하지 않는다.</p>
<details><summary>같은 기존 test 행에서 갱신 전후 교정·훼손</summary><p>새 공격의 정답 행은 제외하고 갱신 전부터 있던 같은 표본군만 비교한다. 기존 행을 새 공격으로 잘못 예측한 경우도 오류에 포함한다.</p><div class="table-wrap"><table id="retention-table"><thead><tr><th>갱신 단계</th><th>XGB 기존 F1 전(%)</th><th>XGB 후(%)</th><th>TabPFN 기존 F1 전(%)</th><th>TabPFN 후(%)</th><th>XGB 교정</th><th>XGB 훼손</th><th>TabPFN 교정</th><th>TabPFN 훼손</th></tr></thead><tbody></tbody></table></div></details>
<details><summary>동일 모델에서 제한 보존 → 누적 전체의 클래스별 변화</summary><label>모델 <select id="history-method"><option value="tabpfn">TabPFN</option><option value="xgb">XGBoost</option></select></label><p>위에서 선택한 데이터·공급 배정·단계의 동일 test 행을 비교한다.</p><div class="table-wrap"><table id="history-classes"><thead><tr><th>클래스</th><th>Test 행</th><th>제한 사례</th><th>누적 사례</th><th>제한 P(%)</th><th>제한 R(%)</th><th>제한 F1(%)</th><th>누적 P(%)</th><th>누적 R(%)</th><th>누적 F1(%)</th></tr></thead><tbody></tbody></table></div><p id="history-corrections"></p></details>
<details><summary>Context와 동일 feature인 test 행을 제외한 보조 평가</summary><p id="overlap-note"></p></details>
</section>

<section id="conclusion"><h2>4. 이 결과가 TabPFN을 채택할 근거가 되는가</h2>
<div class="callout"><strong>일부 조건에서 사용할 근거는 있다. 현재 결과로 IDS 전반의 비용·tail 탐지 우위를 주장하기에는 부족하다.</strong><p>작은 과거 집합을 유지할 때의 갱신 비용과 CIC2018 일부 조건의 탐지 품질은 후속 검증의 근거다. 전체 누적과 대량 추론에서는 같은 이점이 유지되지 않았다.</p></div>
<div id="adoption-evidence"></div>
<h3>다음 검증은 ‘얼마나 높은 F1인가’보다 ‘어디에 쓰면 유리한가’</h3>
<p><strong>갱신 한 번 뒤에 실제로 처리할 트래픽 규모와 필요한 새 공격 탐지 품질을 먼저 정한다.</strong> 같은 라벨 공급과 출현 순서에서 context/과거 사례 예산을 바꾸고, 갱신·추론 시간, 새 공격 P/R/F1, 기존 공격 유지, 정상 오탐을 함께 측정한다. 전체 누적 XGBoost와 제한 보존 XGBoost를 함께 기준으로 두어, TabPFN의 이점이 유지되는 범위를 찾는다.</p>
<p>자연 배정의 소수 공격은 실제 확보 사례 수와 Precision·Recall을 함께 보고, 라벨 공급 확대와 과거 보존 확대를 구별해야 한다. 이번 비교는 신규 MitM 사례 수를 늘리지 않았으므로 추가 라벨의 효과를 검증한 실험으로 해석하지 않는다.</p>
<p class="note">새 실험을 이번 편집에서 실행하지 않았다. 10만 행 확대와 위 후속 검증은 미실행 설계다. 이번 결과는 단일 TabPFN·XGBoost 비교이며 제안 모델의 expert·S/V 기여를 검증하지 않는다.</p>
<details><summary>상세 수치 해석·실행 출처</summary><div id="detailed-findings"></div><p id="source-links"></p><p class="note">TabPFN v3_20260417_multiclass, ensemble 4, frozen backbone, fit_with_cache, CPU cache. XGB 300 rounds, depth 8, learning rate 0.05, CUDA hist. 같은 개발 holdout을 반복 관찰한 결과이며 최종 논문의 확정 성과와 구별한다.</p></details></section>
<footer><p>10/13 실험 · 단일 seed 43 · 같은 구성의 두 모델에 동일 학습·평가 IDs · 로컬 HTML과 통합 탭 동기화</p></footer>
</main><script id="report-data" type="application/json">__DATA__</script><script>
'use strict';
const DATA=JSON.parse(document.getElementById('report-data').textContent);
const DN={cic2018:'CIC2018',toniot:'ToN'},CN={uniform:'공격 균등',natural:'공격 내 자연'},PN={bounded:'제한 보존',cumulative:'누적 전체'};
const N={benign:'Benign',brute_force:'Brute force',dos:'DoS',ddos:'DDoS',web_attacks:'Web attack',infiltration:'Infiltration',bot:'Bot',scanning:'Scanning',injection:'Injection',password:'Password',xss:'XSS',ransomware:'Ransomware',backdoor:'Backdoor',mitm:'MitM'};
const COLORS={xgb:'#D55E00',tabpfn:'#0072B2'};
const el=id=>document.getElementById(id),num=x=>Number(x).toLocaleString('en-US'),dec=x=>x==null?'—':Number(x).toFixed(2),pct=x=>x==null?'—':dec(x*100),rate=x=>Number(x*100).toFixed(4);
const cell=(v,cls='')=>`<td class="${cls}">${v}</td>`,row=c=>`<tr>${c.map(x=>cell(x)).join('')}</tr>`;
const policyData=p=>p==='bounded'?DATA:DATA.cumulative;
const records=(ds,c,m,p=el('policy').value)=>policyData(p).results.filter(r=>r.dataset===ds&&r.case===c&&r.method===m).sort((a,b)=>a.stage-b.stage);
const selected=()=>({ds:el('dataset').value,c:el('case').value,p:el('policy').value,s:Number(el('stage-select').value)});
const summaryRows=()=>['bounded','cumulative'].flatMap(policy=>policyData(policy).summary.map(s=>({...s,policy})));
const label=s=>`${DN[s.dataset]} · ${CN[s.case]} · ${PN[s.policy]}`;
function marker(model,x,y){return model==='xgb'?`<rect x="${x-4}" y="${y-4}" width="8" height="8" fill="${COLORS[model]}"/>`:`<circle cx="${x}" cy="${y}" r="4.3" fill="${COLORS[model]}"/>`;}
function lineChart(id,series,labels,title,percent=false){
 const w=620,h=255,left=55,right=25,top=32,bottom=40;
 const values=series.flatMap(s=>s.values).filter(v=>v!=null),max=percent?100:Math.max(...values)*1.16||1;
 const xx=i=>left+i*(w-left-right)/Math.max(labels.length-1,1),yy=v=>h-bottom-v*(h-top-bottom)/max;
 let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${title}"><title>${title}</title><text x="${left}" y="17" font-size="13" fill="#566773">${percent?'F1 (%)':'초기 제외 누적 시간 (초)'}</text><text x="410" y="17" font-size="13" fill="${COLORS.xgb}">■ XGB</text><text x="505" y="17" font-size="13" fill="${COLORS.tabpfn}">● TabPFN</text>`;
 for(let i=0;i<=4;i++){let v=max*i/4;svg+=`<line x1="${left}" x2="${w-right}" y1="${yy(v)}" y2="${yy(v)}" stroke="#dfe5e9"/><text x="${left-9}" y="${yy(v)+4}" text-anchor="end" font-size="12">${v.toFixed(percent?0:1)}</text>`;}
 for(const s of series){svg+=`<polyline data-model="${s.model}" fill="none" stroke="${COLORS[s.model]}" stroke-width="3" ${s.model==='xgb'?'stroke-dasharray="8 5"':''} points="${s.values.map((v,i)=>v==null?'':xx(i)+','+yy(v)).filter(Boolean).join(' ')}"/>`;s.values.forEach((v,i)=>{if(v!=null)svg+=`<g><title>${s.model==='xgb'?'XGBoost':'TabPFN'} · ${labels[i]}: ${dec(v)}${percent?'%':'초'}</title>${marker(s.model,xx(i),yy(v))}</g>`;});}
 labels.forEach((t,i)=>svg+=`<text x="${xx(i)}" y="${h-14}" text-anchor="middle" font-size="12">${t}</text>`);
 el(id).innerHTML=svg+'</svg>';
}
function inferenceChart(x,p){
 const w=620,h=255,left=100,right=95,max=Math.max(x.predict_seconds,p.predict_seconds),width=w-left-right;
 let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="선택 단계의 추론 시간"><title>선택 단계의 전체 test 추론 시간</title>`;
 [['xgb',x,80],['tabpfn',p,157]].forEach(([m,r,y])=>{let n=width*r.predict_seconds/max;svg+=`<text x="${left-15}" y="${y+5}" text-anchor="end" font-size="14">${m==='xgb'?'XGBoost':'TabPFN'}</text><rect data-model="${m}" x="${left}" y="${y-16}" width="${n}" height="32" fill="${COLORS[m]}"/><text x="${left+n+9}" y="${y+5}" font-size="14">${dec(r.predict_seconds)}초</text>`;});
 svg+=`<text x="${left}" y="218" font-size="13" fill="#566773">동일 test ${num(p.rows)}행 · TabPFN / XGB ${dec(p.predict_seconds/x.predict_seconds)}배</text></svg>`;el('inference-chart').innerHTML=svg;
}
function resetStages(){let {ds,c,p}=selected(),rr=records(ds,c,'xgb',p);el('stage-select').innerHTML=rr.map(r=>`<option value="${r.stage}">${r.stage===0?'초기':r.stage+' · '+r.new_classes.map(c=>N[c]).join(' / ')}</option>`).join('');el('stage-select').value=rr.at(-1).stage;render();}
function renderHistory(){
 const {ds,c,s}=selected(),m=el('history-method').value,b=records(ds,c,m,'bounded')[s],a=records(ds,c,m,'cumulative')[s],bm=Object.fromEntries(b.classes.map(q=>[q.name,q]));
 el('history-classes').tBodies[0].innerHTML=[...a.classes].sort((x,y)=>y.support-x.support).map(q=>{const z=bm[q.name];return row([N[q.name],num(q.support),num(b.fit_class_counts[q.name]),num(a.fit_class_counts[q.name]),pct(z.precision),pct(z.recall),pct(z.f1),pct(q.precision),pct(q.recall),pct(q.f1)]);}).join('');
 const v=DATA.policy_comparison.stages.find(r=>r.dataset===ds&&r.case===c&&r.method===m&&r.stage===s);
 el('history-corrections').textContent=`같은 단계의 제한 보존 → 누적 전체: 교정 ${num(v.corrected)}건 / 훼손 ${num(v.damaged)}건. 입력량과 클래스 비율이 함께 바뀐다.`;
}
function render(){
 const {ds,c,p:policy,s}=selected(),xs=records(ds,c,'xgb'),ps=records(ds,c,'tabpfn'),x=xs[s],p=ps[s];
 const description=`${DN[ds]} · ${CN[c]} · ${PN[policy]}`;
 el('selection-caption').textContent=description+' — 아래 그래프와 상세 표에 적용';
 const cumulative=rr=>{let total=0;return rr.map((r,i)=>i?(total+=r.update_seconds):0);};
 lineChart('chart',[{model:'xgb',values:cumulative(xs)},{model:'tabpfn',values:cumulative(ps)}],xs.map(r=>r.stage===0?'초기':r.stage+'단계'),'후속 갱신 시간 누적');inferenceChart(x,p);
 el('inference-caption').textContent=`추론 그래프: ${s===0?'초기':s+'단계'} · 전체 test ${num(p.rows)}행. 선택 단계의 갱신은 XGB ${dec(x.update_seconds)}초 / TabPFN ${dec(p.update_seconds)}초. 추론 시간에는 배치 예측과 결과 기록이 포함된다.`;
 el('stages').tBodies[0].innerHTML=xs.map((r,i)=>row([i===0?'초기':i+' · '+r.new_classes.map(n=>N[n]).join(' / '),num(r.incoming_rows),num(r.history_rows),num(r.context_rows),dec(r.update_seconds),dec(ps[i].update_seconds),dec(r.predict_seconds),dec(ps[i].predict_seconds)])).join('');
 const fields=[['입력 구성(초)','select_seconds'],['모델 생성(초)','construct_seconds'],['학습 / 전처리·캐시(초)','fit_seconds'],['갱신 합계(초)','update_seconds'],['전체 test 추론(초)','predict_seconds'],['첫 batch 추론(초)','first_batch_seconds']];
 el('cost-detail').tBodies[0].innerHTML=fields.map(([name,k])=>row([name,dec(x[k]),dec(p[k])])).join('');
 lineChart('new-attack-chart',[{model:'xgb',values:xs.slice(1).map(r=>100*r.new_class_macro_f1)},{model:'tabpfn',values:ps.slice(1).map(r=>100*r.new_class_macro_f1)}],xs.slice(1).map(r=>r.stage+'단계'),'신규 공격 F1',true);
 el('class-caption').textContent=`${description} · ${s===0?'초기':s+'단계'} · ${num(p.rows)} test 행 / 신규 ${num(p.incoming_rows)} + 과거 ${num(p.history_rows)} = 입력 ${num(p.context_rows)}행`;
 const xm=Object.fromEntries(x.classes.map(v=>[v.name,v]));
 el('class-table').tBodies[0].innerHTML=[...p.classes].sort((a,b)=>b.support-a.support).map(q=>{const a=xm[q.name],d=100*(q.f1-a.f1);return `<tr class="${p.new_classes.includes(q.name)?'new':''}">${[N[q.name],num(q.support),num(p.incoming_class_counts[q.name]),num(p.fit_class_counts[q.name]),pct(a.precision),pct(a.recall),pct(a.f1),pct(q.precision),pct(q.recall),pct(q.f1)].map(v=>cell(v)).join('')}${cell((d>0?'+':'')+dec(d),d>0?'gain':d<0?'loss':'')}</tr>`;}).join('');
 el('retention-table').tBodies[0].innerHTML=xs.slice(1).map(r=>{const q=ps[r.stage];return row([r.stage+' · '+r.new_classes.map(n=>N[n]).join(' / '),pct(r.retention_before_macro_f1),pct(r.retention_after_macro_f1),pct(q.retention_before_macro_f1),pct(q.retention_after_macro_f1),num(r.old_corrected),num(r.old_damaged),num(q.old_corrected),num(q.old_damaged)]);}).join('');
 el('overlap-note').textContent=`동일 feature test ${num(p.exact_feature_overlap_test_rows)}개를 제외한 ${num(p.without_context_vector_overlap.rows)}개에서 Macro-F1: XGB ${pct(x.without_context_vector_overlap.macro_f1)}%, TabPFN ${pct(p.without_context_vector_overlap.macro_f1)}%. 전체 test의 주 평가와 구별되는 보조 결과다.`;
 renderHistory();
}
function setup(){
 el('timelines').innerHTML=Object.entries(DATA.datasets).map(([ds,d])=>`<div class="timeline"><strong>${DN[ds]}</strong><div class="steps">${d.stages.map((cs,i)=>`<div class="step"><small>${i===0?'초기':i+'단계'}</small><span>${cs.map(c=>N[c]).join(' · ')}</span>${i?'<small>+ Benign</small>':''}</div>`).join('')}</div></div>`).join('');
 el('allocations').innerHTML=Object.entries(DATA.datasets).map(([ds,d])=>`<h3>${DN[ds]}</h3><div class="table-wrap"><table><thead><tr><th>클래스</th><th>Test 행</th><th>정제 train 가용</th><th>균등 공급</th><th>자연 공급</th><th>최초 출현(UTC)</th></tr></thead><tbody>${Object.keys(d.fixed_test_class_counts).sort((a,b)=>d.fixed_test_class_counts[b]-d.fixed_test_class_counts[a]).map(c=>row([N[c],num(d.fixed_test_class_counts[c]),num(d.available_train[c]),num(d.allocation.uniform[c]),num(d.allocation.natural[c]),d.first_appearance_utc[c].slice(0,10)])).join('')}</tbody></table></div>`).join('');
 el('cost-summary').tBodies[0].innerHTML=summaryRows().map(s=>{const x=records(s.dataset,s.case,'xgb',s.policy).at(-1),p=records(s.dataset,s.case,'tabpfn',s.policy).at(-1);return row([label(s),num(p.context_rows),dec(s.xgb.update_sum),dec(s.tabpfn.update_sum),dec(x.predict_seconds),dec(p.predict_seconds),num(p.rows)]);}).join('');
 el('performance-summary').tBodies[0].innerHTML=summaryRows().map(s=>{const d=100*(s.tabpfn.final_macro_f1-s.xgb.final_macro_f1);return `<tr>${[label(s),pct(s.xgb.final_macro_f1),pct(s.tabpfn.final_macro_f1)].map(v=>cell(v)).join('')}${cell((d>0?'+':'')+dec(d),d>0?'gain':d<0?'loss':'')}${cell(rate(s.xgb.final_benign_fpr))}${cell(rate(s.tabpfn.final_benign_fpr))}</tr>`;}).join('');
 el('cost-verdict').textContent=DATA.scenario_narrative.cost;
 el('performance-verdict').textContent=DATA.scenario_narrative.performance;
 el('environment-note').textContent=DATA.comparison_environment_note;
 for(const text of DATA.scenario_narrative.adoption){const p=document.createElement('p');p.textContent=text;el('adoption-evidence').appendChild(p);}
 for(const text of DATA.comparison_text.slice(1)){const p=document.createElement('p');p.textContent=text;el('detailed-findings').appendChild(p);}
 const sources=[['제한 실행',DATA.run],['누적 실행',DATA.cumulative.run],['제한 단계별 CSV',DATA.run+'/summary.csv'],['누적 단계별 CSV',DATA.cumulative.run+'/summary.csv'],['클래스별 CSV',DATA.cumulative.run+'/class_metrics.csv'],['서술형 연구 기록','docs/research/20261013/exp71_cumulative_results.md']];
 el('source-links').innerHTML=sources.map(([name,path])=>location.protocol==='data:'?`${name}: <code>${path}</code>`:`<a href="../../${path}">${name}</a>`).join('<br>');
 for(const id of ['dataset','case','policy'])el(id).addEventListener('change',resetStages);
 el('stage-select').addEventListener('change',render);el('history-method').addEventListener('change',renderHistory);resetStages();
}
setup();
</script></body></html>'''

EXTRA_CSS = r'''
main{max-width:1320px}section{scroll-margin-top:100px}.chart-grid,.model-grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}.chart svg{max-height:none}.model-card,.finding-box{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px 20px}.model-card h3{margin-top:0}.model-card.xgb{border-top:4px solid #D55E00}.model-card.pfn{border-top:4px solid #0072B2}.xgb-label{color:#b44400}.pfn-label{color:#0072B2}.legend{display:flex;gap:24px;flex-wrap:wrap;font-size:14px;font-weight:600}.timeline{margin:16px 0}.steps{display:flex;gap:8px;margin-top:7px;flex-wrap:wrap}.step{flex:1;min-width:120px;border:1px solid var(--line);background:#fff;border-radius:6px;padding:9px 11px}.step span,.step small{display:block}.step span{font-size:13px;font-weight:600}.step small{font-size:11px}.selection-caption{font-weight:600}.control{margin-top:26px;box-shadow:0 2px 5px #00000008}#design-table{white-space:normal;min-width:620px}#design-table td{vertical-align:top}#design-table td:first-child{min-width:135px}.gain{color:#12623b}.loss{color:#a33434}tr.new td:first-child{border-left-color:#64748b}h3{scroll-margin-top:100px}@media(max-width:800px){.chart-grid,.model-grid{grid-template-columns:1fr}.chart{padding:6px}.legend{gap:12px}.control{position:static}.step{min-width:135px}}
'''


def narrative(data):
    def summary(policy, ds, case):
        d=data if policy=='bounded' else data['cumulative']
        return next(s for s in d['summary'] if (s['dataset'],s['case'])==(ds,case))
    def final(policy, ds, case, method):
        d=data if policy=='bounded' else data['cumulative']
        return max((r for r in d['results'] if (r['dataset'],r['case'],r['method'])==(ds,case,method)),key=lambda r:r['stage'])
    fast=[sum(s['pfn_xgb_update_ratio']<1 for s in d['summary']) for d in [data,data['cumulative']]]
    c=summary('bounded','cic2018','uniform');u=summary('cumulative','cic2018','uniform')
    t=summary('cumulative','toniot','natural')
    ic=final('cumulative','cic2018','uniform','tabpfn');ix=final('cumulative','cic2018','uniform','xgb')
    inf={m:next(q for q in r['classes'] if q['name']=='infiltration')['f1'] for m,r in [('xgb',ix),('tabpfn',ic)]}
    tn=final('cumulative','toniot','natural','tabpfn');bn=final('bounded','toniot','natural','tabpfn')
    cm={q['name']:q for q in tn['classes']};bcm={q['name']:q for q in bn['classes']}
    cost=(f'제한 보존에서는 4개 구성 중 {fast[0]}개에서 TabPFN의 후속 갱신 합계가 더 짧았고, 누적 전체에서는 {fast[1]}개였다. '
          '최종 test의 대량 추론은 8개 구성 모두 XGBoost가 더 빨랐다. 따라서 갱신 비용 절감과 추론 비용 절감을 구별해야 한다.')
    performance=(f"CIC2018 균등·제한 보존에서는 두 모델의 최종 Macro-F1이 약 {100*c['xgb']['final_macro_f1']:.2f}%로 유사했다. "
                 f"같은 균등 공급의 누적 조건에서는 TabPFN {100*u['tabpfn']['final_macro_f1']:.2f}% / XGB {100*u['xgb']['final_macro_f1']:.2f}%였다. "
                 'CIC2018 자연 배정과 ToN의 두 배정에서는 최종 Macro-F1이 XGBoost보다 낮아, 데이터·구성 전반의 우위로 이어지지는 않았다.')
    adoption=[
        f"채택을 검토할 근거 ① — CIC2018 균등·제한 보존: 후속 갱신 합계는 TabPFN {c['tabpfn']['update_sum']:.2f}초 / XGB {c['xgb']['update_sum']:.2f}초, 최종 F1은 {100*c['tabpfn']['final_macro_f1']:.2f}% / {100*c['xgb']['final_macro_f1']:.2f}%였다. 정상 오탐률은 {100*c['tabpfn']['final_benign_fpr']:.4f}% / {100*c['xgb']['final_benign_fpr']:.4f}%다. 비슷한 전체 F1에서 갱신과 오탐의 이점이 함께 관찰됐다. 다만 최종 test 추론은 TabPFN이 더 느려 운영 총비용 우위를 뜻하지 않는다.",
        f"채택을 검토할 근거 ② — CIC2018 균등·누적 전체: Infiltration F1은 TabPFN {100*inf['tabpfn']:.2f}% / XGB {100*inf['xgb']:.2f}%로 특정 소수 공격에서 이점이 있었다. 그러나 이 조건의 TabPFN 갱신은 더 느렸고 Web attack까지 같은 이점이 나타난 것은 아니다. 공격별 탐지 품질을 선택적으로 검증할 이유다.",
        f"아직 해결할 점 — ToN 자연비율·누적 전체: TabPFN 최종 F1 {100*t['tabpfn']['final_macro_f1']:.2f}%는 XGB {100*t['xgb']['final_macro_f1']:.2f}%보다 낮다. TabPFN의 제한→누적 개선에는 Ransomware F1 {100*bcm['ransomware']['f1']:.2f}%→{100*cm['ransomware']['f1']:.2f}%가 포함되지만, Scanning F1은 {100*cm['scanning']['f1']:.2f}%이고 MitM precision은 {100*cm['mitm']['precision']:.2f}%에 머문다. 과거 사례를 더 남기는 것만으로 tail 문제가 해결되지는 않았다.",
        '현재의 판단 — 가중치 재학습 없이 새 클래스를 반영할 수 있다는 작동 방식과 제한된 조건의 이점은 확인했다. 모든 공격의 tail 성능 개선이나 XGBoost보다 낮은 운영 총비용은 아직 확인하지 못했다. TabPFN을 채택하려면 확보 라벨·context 예산·갱신 빈도·갱신당 추론량을 명시하고, 필요한 탐지 품질을 충족하는 조건에서 비교해야 한다.'
    ]
    return dict(cost=cost,performance=performance,adoption=adoption)


def make_page(data, base_page):
    data['scenario_narrative']=narrative(data)
    data['comparison_environment_note']=environment_note(data)
    css=re.search(r'<style>(.*?)</style>',base_page,re.S).group(1)
    return ('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>'+TITLE+'</title><style>'+css+EXTRA_CSS+'</style></head><body><main>'+BODY)


def notes(data):
    n=data['scenario_narrative']
    paragraphs=['# EXP71 · 새 공격 도입 시나리오에서 TabPFN을 채택할 근거',
        '2026-10-14 보고서 재구성. 10/13에 완료한 제한 보존·누적 전체 실험을 사용했으며 추가 추론이나 새 실험은 수행하지 않았다.',
        '연구 의도는 새 공격의 라벨 사례가 확보됐을 때 사전학습 가중치를 고정한 TabPFN이 context 변경만으로 유용한 탐지 품질을 유지하면서 갱신 비용을 줄일 수 있는지 확인하는 것이다. 원본 클래스의 최초 출현 순서를 유지하되 초기 및 일부 인접 공격을 묶었고, 각 단계에서 신규 공격과 Benign 사례를 공급했다. 실제 시간축의 트래픽 재현이나 미관측 공격의 라벨 없는 탐지를 검증한 것은 아니다.',
        '변화 조건은 공격별 공급량(균등 / 공격 내 자연 비율)과 과거 보존(최대 4,096개 / 누적 전체)이다. 두 데이터에서 8개 구성을 비교했다. 최종 공급은 CIC2018 22,280행, ToN 36,864행이고 Benign은 각 75%를 매 단계 나눠 공급한다. 구성 내 두 모델의 학습·test IDs와 보존 정책 간 신규 공급은 고정했다.',
        n['cost'],
        '갱신 시간은 초기 구축을 제외한 후속 단계 합계다. 추론은 비교 범위를 맞추기 위해 최종 단계의 전체 test 1회 시간을 별도로 제시한다. 배치 예측과 결과 기록이 포함되고, 운영 배포·직렬화는 미측정이다. 두 열의 측정 범위가 다르므로 단순 합계를 운영 총비용으로 해석하지 않는다.',
        n['performance'],*n['adoption'],
        '후속 검증은 한 번 갱신한 뒤 처리할 트래픽 규모와 필요한 새 공격 탐지 품질을 먼저 정하고, 같은 라벨 공급·출현 순서에서 입력 예산을 변화시키는 것이다. 갱신·추론 시간을 반복 측정하고 새 공격의 P/R/F1, 기존 공격 유지, 정상 오탐을 함께 평가한다. 제한 보존 및 전체 누적 XGBoost와 비교해 TabPFN의 이점이 유지되는 범위를 확인한다. 라벨 공급 확대와 과거 보존 확대는 별도 요인으로 구별한다.',
        environment_note(data),
        '단일 seed·단계별 1회 timing·반복 관찰한 개발 holdout이라는 범위에서 해석한다. 10만 행 공급 확대는 미실행이며, expert·S/V가 들어간 제안 모델의 기여를 이 결과로 주장하지 않는다.',
        '## 상세 수치',*data['comparison_text'][1:],
        '## 출처',f"- 제한 실행: `{data['run']}`\n- 누적 실행: `{data['cumulative']['run']}`\n- 각 실행의 `summary.csv`, `class_metrics.csv`, `jobs/*/stage_*/metrics.json`\n- 비교 검증: `{data['cumulative']['run']}/comparison_validation.json`\n- HTML: `lablog/html_report/bounded_updates_1013.html`"]
    return '\n\n'.join(paragraphs)+'\n'
