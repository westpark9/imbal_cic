"""Present EXP71 by scenario, cost, detection quality, and adoption evidence."""
import re


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
<h3>과거 사례는 모두 누적하고 공격별 공급량을 비교</h3>
<div class="table-wrap"><table id="design-table"><thead><tr><th>조건</th><th class="left">공격 균등</th><th class="left">공격 내 자연 비율</th></tr></thead><tbody>
<tr><td>공격별 배정</td><td class="left">공격 예산을 클래스별로 균등하게 배분한다. 가용 사례가 부족한 클래스는 상한을 적용한다.</td><td class="left">같은 공격 예산을 정제 train의 공격별 빈도에 비례해 배분한다.</td></tr>
<tr><td>공통 설정</td><td class="left" colspan="2">Benign은 최종 공급의 75%를 매 단계 나눠 공급한다. 신규 사례와 이전에 확보한 모든 사례를 누적하여 두 모델에 동일하게 제공한다.</td></tr>
</tbody></table></div>
<p id="input-summary"></p>
<p class="note">공격 내 자연 비율은 전체 트래픽의 자연 분포를 뜻하지 않는다. 두 모델은 같은 조건에서 동일한 학습·test IDs를 사용한다. 균등/자연 배정은 최종 총량과 Benign IDs를 맞추며, 공격별 사례 수와 중간 단계의 입력 크기는 달라진다.</p>
<div id="scale-status" class="callout" hidden></div>
<details><summary>클래스별 공급량·test 표본 수와 최초 출현일</summary><div id="allocations"></div><p class="note">표는 해당 데이터셋의 전체 test 표본 수 내림차순. 공급량은 실험 전체에 제공한 서로 다른 행 수이며 최종 누적 context에 모두 포함된다. 최초 출현일은 원본 기록의 UTC 기준이다.</p></details></section>

<div class="control" id="explore-controls" aria-label="비용과 성능 상세 조건">
<label>데이터 <select id="dataset"><option value="cic2018">CIC2018</option><option value="toniot">ToN</option></select></label>
<label>공급 <select id="case"><option value="uniform">공격 균등</option><option value="natural">공격 내 자연 비율</option></select></label>
<label>상세 단계 <select id="stage-select"></select></label></div>

<section id="cost"><h2>2. 갱신이 빠른가, 추론까지 효율적인가</h2>
<p><strong>갱신</strong>은 새 라벨 사례를 반영해 다음 예측을 준비하는 비용이고, <strong>추론</strong>은 준비된 모델로 test 행을 예측하는 비용이다. TabPFN이 가중치를 재학습하지 않는다는 사실이 두 비용 모두 낮다는 뜻은 아니다.</p>
<div class="callout" id="cost-verdict"></div><p><strong>누적 전체에서 갱신이 느려질 수 있는 이유:</strong> 현재 구현은 매 단계 누적 전체 context로 <code>fit_with_cache</code>를 다시 호출한다. TabPFN은 가중치를 학습하지 않지만, ensemble 4개의 전처리·context 인코딩·캐시를 다시 구성한다. 이는 새 사례만 캐시에 추가하는 방식이 아니며, 입력이 커질수록 갱신 비용도 늘 수 있다. XGB도 같은 전체 사례로 300개 트리를 새로 학습한다. 구간별 실측 시간은 아래 토글에서 확인할 수 있다.</p>
<div class="legend" aria-label="모델 범례"><span class="xgb-label">■ ┄ XGBoost · 주황 / 점선</span><span class="pfn-label">● ━ TabPFN · 파랑 / 실선</span></div>
<p id="selection-caption" class="selection-caption"></p>
<div class="chart-grid"><div><h3>후속 갱신 시간의 누적</h3><div id="chart" class="chart"></div></div><div><h3>선택한 단계의 test 추론</h3><div id="inference-chart" class="chart"></div></div></div>
<p class="note" id="inference-caption"></p>
<h3>4개 구성의 비용 비교</h3><p>갱신 열은 <strong>초기 구축을 제외한 후속 단계의 합</strong>, 추론 열은 <strong>최종 단계의 전체 test 1회</strong>다. 두 측정 범위가 다르므로 이 표의 열을 더해 운영 총비용으로 부르지 않는다.</p>
<div class="table-wrap"><table id="cost-summary"><thead><tr><th>데이터 · 공급</th><th>최종 입력</th><th class="xgb-label">XGB 갱신(초)</th><th class="pfn-label">TabPFN 갱신(초)</th><th class="xgb-label">XGB 추론(초)</th><th class="pfn-label">TabPFN 추론(초)</th><th>최종 test 행</th></tr></thead><tbody></tbody></table></div>
<details><summary>선택한 구성의 초기 구축·단계별 비용과 입력량</summary><div class="table-wrap"><table id="stages"><thead><tr><th>단계 · 새 공격</th><th>신규 사례</th><th>과거 보존</th><th>전체 입력</th><th>XGB 갱신(초)</th><th>TabPFN 갱신(초)</th><th>XGB 추론(초)</th><th>TabPFN 추론(초)</th></tr></thead><tbody></tbody></table></div><div class="table-wrap"><table id="cost-detail"><thead><tr><th>선택 단계의 항목</th><th>XGB</th><th>TabPFN</th></tr></thead><tbody></tbody></table></div><p class="note">갱신 = 입력 선택·구성 + 모델 생성 + 학습 또는 전처리·캐시 구성. 초기 행은 첫 모델 구축 비용을 포함한다. 프로세스 시작·데이터 사전 준비·운영 배포·직렬화는 미측정이다. 추론은 배치 예측과 결과 기록을 포함하며, 첫 batch는 최대 32,768행이므로 단일 행 지연 시간이 아니다.</p></details>
<details><summary>측정 환경과 재개 이력</summary><p id="environment-note"></p><p class="note">RTX 4090, GPU 모델 worker 순차 실행. 단계별 1회 timing·단일 seed이므로 근접한 차이는 반복 측정이 필요하다.</p></details></section>

<section id="performance"><h2>3. 새 공격을 잘 탐지하고 기존 탐지를 유지했는가</h2>
<div id="performance-verdict" class="callout"></div>
<p>아래 요약은 모든 공격이 도입된 <strong>최종 단계의 같은 test</strong>를 비교한다. 전체 Macro-F1과 Benign 오탐률을 함께 보고, 소수 공격은 클래스별 Precision·Recall·F1로 확인한다.</p>
<div class="table-wrap"><table id="performance-summary"><thead><tr><th>데이터 · 공급</th><th class="xgb-label">XGB F1(%)</th><th class="pfn-label">TabPFN F1(%)</th><th>TabPFN−XGB(%p)</th><th>XGB 정상 오탐(%)</th><th>TabPFN 정상 오탐(%)</th></tr></thead><tbody></tbody></table></div>
<div class="chart-grid"><div><h3>선택한 구성의 신규 공격 F1</h3><div id="new-attack-chart" class="chart"></div></div><div class="finding-box"><h3>성능 차이를 읽는 기준</h3><p>각 단계의 신규 공격만 평균하되, Precision의 FP는 정상·기존 공격을 포함한 그 단계 전체 test에서 계산한다.</p><p>단계마다 평가 공격이 바뀌므로 선의 상승·하락을 같은 공격의 학습 곡선으로 해석하지 않는다. 기존 탐지 유지 여부는 아래의 동일 기존 test 행 비교로 확인한다.</p></div></div>
<details id="class-details"><summary id="classes">단계별 클래스 성능 보기 · Precision / Recall / F1</summary><p id="class-caption"></p>
<div class="table-wrap"><table id="class-table"><thead><tr><th>클래스</th><th>Test 행</th><th>이번 유입</th><th>누적 사례</th><th>XGB P(%)</th><th>XGB R(%)</th><th>XGB F1(%)</th><th>TabPFN P(%)</th><th>TabPFN R(%)</th><th>TabPFN F1(%)</th><th>ΔF1(%p)</th></tr></thead><tbody></tbody></table></div>
<p class="note">해당 단계 test 표본 수 내림차순. 왼쪽 표시는 이번에 새로 도입한 공격이며, 색은 모델을 의미하지 않는다.</p></details>
<details><summary>같은 기존 test 행에서 갱신 전후 교정·훼손</summary><p>새 공격의 정답 행은 제외하고 갱신 전부터 있던 같은 표본군만 비교한다. 기존 행을 새 공격으로 잘못 예측한 경우도 오류에 포함한다.</p><div class="table-wrap"><table id="retention-table"><thead><tr><th>갱신 단계</th><th>XGB 기존 F1 전(%)</th><th>XGB 후(%)</th><th>TabPFN 기존 F1 전(%)</th><th>TabPFN 후(%)</th><th>XGB 교정</th><th>XGB 훼손</th><th>TabPFN 교정</th><th>TabPFN 훼손</th></tr></thead><tbody></tbody></table></div></details>
<details><summary>Context와 동일 feature인 test 행을 제외한 보조 평가</summary><p id="overlap-note"></p></details>
</section>

<section id="conclusion"><h2>4. 이 결과가 TabPFN을 채택할 근거가 되는가</h2>
<div id="adoption-evidence"></div>
<h3>입력 규모를 늘려 필요한 탐지 품질을 확보할 수 있는지 확인</h3>
<p>현재 후속 실험은 <strong>최종 누적 입력을 두 데이터 모두 10만 행으로 확대</strong>하는 것이다. Benign 7.5만·공격 2.5만을 공급하며 공격 균등과 공격 내 자연비율을 비교한다. 과거 사례는 모두 유지하고 클래스 출현 순서·평가 test·모델 설정을 고정한다. 성능 개선 여부와 함께 갱신·추론 비용 증가를 확인한다.</p>
<p>규모를 확대해도 자연비율 조건의 소수 공격 사례 수는 여전히 적을 수 있다. 균등 조건과 함께 보면 전체 입력량뿐 아니라 공격별 확보 라벨 수에 따른 차이를 확인할 수 있다. 단일 seed 비교이므로 분포 효과나 개별 클래스의 실패 원인을 확정하는 인과 실험으로 해석하지 않는다.</p>
<p class="note">이 결과는 단일 TabPFN·XGBoost 비교다. 제안 모델의 expert·S/V 기여와 구별하며, 가중치 재학습이 없다는 사실만으로 갱신 시간이나 운영 총비용의 우위를 주장하지 않는다.</p>
<details id="scale-comparison" hidden><summary>이전 누적 규모 → 10만 행 결과 비교</summary><div id="scale-results"></div></details>
<details><summary>상세 수치 해석·실행 출처</summary><div id="detailed-findings"></div><p id="source-links"></p><p class="note">TabPFN v3_20260417_multiclass, ensemble 4, frozen backbone, fit_with_cache, CPU cache. XGB 300 rounds, depth 8, learning rate 0.05, CUDA hist. 같은 개발 holdout을 반복 관찰한 결과이며 최종 논문의 확정 성과와 구별한다.</p></details></section>
<footer><p>10/13 실험 · 단일 seed 43 · 같은 구성의 두 모델에 동일 학습·평가 IDs · 로컬 HTML과 통합 탭 동기화</p></footer>
</main><script id="report-data" type="application/json">__DATA__</script><script>
'use strict';
const DATA=JSON.parse(document.getElementById('report-data').textContent);
const DN={cic2018:'CIC2018',toniot:'ToN'},CN={uniform:'공격 균등',natural:'공격 내 자연'};
const N={benign:'Benign',brute_force:'Brute force',dos:'DoS',ddos:'DDoS',web_attacks:'Web attack',infiltration:'Infiltration',bot:'Bot',scanning:'Scanning',injection:'Injection',password:'Password',xss:'XSS',ransomware:'Ransomware',backdoor:'Backdoor',mitm:'MitM'};
const COLORS={xgb:'#D55E00',tabpfn:'#0072B2'};
const el=id=>document.getElementById(id),num=x=>Number(x).toLocaleString('en-US'),dec=x=>x==null?'—':Number(x).toFixed(2),pct=x=>x==null?'—':dec(x*100),rate=x=>Number(x*100).toFixed(4);
const cell=(v,cls='')=>`<td class="${cls}">${v}</td>`,row=c=>`<tr>${c.map(x=>cell(x)).join('')}</tr>`;

const records=(ds,c,m)=>DATA.results.filter(r=>r.dataset===ds&&r.case===c&&r.method===m).sort((a,b)=>a.stage-b.stage);
const selected=()=>({ds:el('dataset').value,c:el('case').value,s:Number(el('stage-select').value)});
const summaryRows=()=>DATA.summary;
const label=s=>`${DN[s.dataset]} · ${CN[s.case]}`;
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
function render(){
 const {ds,c,s}=selected(),xs=records(ds,c,'xgb'),ps=records(ds,c,'tabpfn'),x=xs[s],p=ps[s];
 const description=`${DN[ds]} · ${CN[c]} · 누적 전체`;
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
}
function setup(){
 const sizes=Object.entries(DATA.datasets).map(([ds,d])=>`${DN[ds]} ${num(d.final_stream_rows)}행`).join(' / ');
 el('input-summary').innerHTML=`현재 완료 결과의 최종 누적 입력은 <strong>${sizes}</strong>이다. 두 데이터 × 두 배정 = 4개 구성을 비교했다. 과거 사례를 버리지 않으므로 최종 공급량과 입력량은 같다.`;
 if(DATA.pending){el('scale-status').hidden=false;el('scale-status').textContent=`10만 행 확대 실험 실행 중 · ${DATA.pending.checked_at} 확인. 아래 비용·성능은 위에 명시한 기존 규모의 완료 결과다. 두 데이터 각각 Benign 75,000 + 공격 25,000, seed 43, 누적 전체로 실행하고 있다. 검증 완료 후 10만 행 결과로 갱신한다.`;}
 if(DATA.previous){
   el('scale-comparison').hidden=false;
   const rows=DATA.summary.map(s=>{const b=DATA.previous.summary.find(q=>q.dataset===s.dataset&&q.case===s.case);return row([label(s),num(b.tabpfn.final_context_rows),num(s.tabpfn.final_context_rows),pct(b.xgb.final_macro_f1),pct(s.xgb.final_macro_f1),pct(b.tabpfn.final_macro_f1),pct(s.tabpfn.final_macro_f1),dec(b.tabpfn.update_sum),dec(s.tabpfn.update_sum)]);}).join('');
   el('scale-results').innerHTML=`<p>같은 클래스 출현 순서·test·모델 설정. 기존 공급 IDs를 같은 단계에 포함하면서 확대했다. 일부 이전 worker와 driver/kernel이 달라 비용 차이는 입력량만의 효과로 단정하지 않는다.</p><div class="table-wrap"><table><thead><tr><th>데이터 · 공급</th><th>이전 입력</th><th>확대 입력</th><th>XGB 이전 F1(%)</th><th>XGB 확대 F1(%)</th><th>TabPFN 이전 F1(%)</th><th>TabPFN 확대 F1(%)</th><th>TabPFN 이전 갱신(초)</th><th>TabPFN 확대 갱신(초)</th></tr></thead><tbody>${rows}</tbody></table></div>`;
 }
 el('timelines').innerHTML=Object.entries(DATA.datasets).map(([ds,d])=>`<div class="timeline"><strong>${DN[ds]}</strong><div class="steps">${d.stages.map((cs,i)=>`<div class="step"><small>${i===0?'초기':i+'단계'}</small><span>${cs.map(c=>N[c]).join(' · ')}</span>${i?'<small>+ Benign</small>':''}</div>`).join('')}</div></div>`).join('');
 el('allocations').innerHTML=Object.entries(DATA.datasets).map(([ds,d])=>`<h3>${DN[ds]}</h3><div class="table-wrap"><table><thead><tr><th>클래스</th><th>Test 행</th><th>정제 train 가용</th><th>균등 공급</th><th>자연 공급</th><th>최초 출현(UTC)</th></tr></thead><tbody>${Object.keys(d.fixed_test_class_counts).sort((a,b)=>d.fixed_test_class_counts[b]-d.fixed_test_class_counts[a]).map(c=>row([N[c],num(d.fixed_test_class_counts[c]),num(d.available_train[c]),num(d.allocation.uniform[c]),num(d.allocation.natural[c]),d.first_appearance_utc[c].slice(0,10)])).join('')}</tbody></table></div>`).join('');
 el('cost-summary').tBodies[0].innerHTML=summaryRows().map(s=>{const x=records(s.dataset,s.case,'xgb',s.policy).at(-1),p=records(s.dataset,s.case,'tabpfn',s.policy).at(-1);return row([label(s),num(p.context_rows),dec(s.xgb.update_sum),dec(s.tabpfn.update_sum),dec(x.predict_seconds),dec(p.predict_seconds),num(p.rows)]);}).join('');
 el('performance-summary').tBodies[0].innerHTML=summaryRows().map(s=>{const d=100*(s.tabpfn.final_macro_f1-s.xgb.final_macro_f1);return `<tr>${[label(s),pct(s.xgb.final_macro_f1),pct(s.tabpfn.final_macro_f1)].map(v=>cell(v)).join('')}${cell((d>0?'+':'')+dec(d),d>0?'gain':d<0?'loss':'')}${cell(rate(s.xgb.final_benign_fpr))}${cell(rate(s.tabpfn.final_benign_fpr))}</tr>`;}).join('');
 el('cost-verdict').textContent=DATA.scenario_narrative.cost;
 el('performance-verdict').textContent=DATA.scenario_narrative.performance;
 el('environment-note').textContent=DATA.environment_note;
 for(const text of DATA.scenario_narrative.adoption){const p=document.createElement('p');p.textContent=text;el('adoption-evidence').appendChild(p);}
 for(const text of DATA.interpretation){const p=document.createElement('p');p.textContent=text;el('detailed-findings').appendChild(p);}
 const sources=[['누적 실행',DATA.run],['단계별 CSV',DATA.run+'/summary.csv'],['클래스별 CSV',DATA.run+'/class_metrics.csv'],['서술형 연구 기록','docs/lablog/reports/20261014.tex']];
 el('source-links').innerHTML=sources.map(([name,path])=>location.protocol==='data:'?`${name}: <code>${path}</code>`:`<a href="../../${path}">${name}</a>`).join('<br>');
 for(const id of ['dataset','case'])el(id).addEventListener('change',resetStages);
 el('stage-select').addEventListener('change',render);resetStages();
}
setup();
</script></body></html>'''

EXTRA_CSS = r'''
main{max-width:1320px}section{scroll-margin-top:100px}.chart-grid,.model-grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}.chart svg{max-height:none}.model-card,.finding-box{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px 20px}.model-card h3{margin-top:0}.model-card.xgb{border-top:4px solid #D55E00}.model-card.pfn{border-top:4px solid #0072B2}.xgb-label{color:#b44400}.pfn-label{color:#0072B2}.legend{display:flex;gap:24px;flex-wrap:wrap;font-size:14px;font-weight:600}.timeline{margin:16px 0}.steps{display:flex;gap:8px;margin-top:7px;flex-wrap:wrap}.step{flex:1;min-width:120px;border:1px solid var(--line);background:#fff;border-radius:6px;padding:9px 11px}.step span,.step small{display:block}.step span{font-size:13px;font-weight:600}.step small{font-size:11px}.selection-caption{font-weight:600}.control{margin-top:26px;box-shadow:0 2px 5px #00000008}#design-table{white-space:normal;min-width:620px}#design-table td{vertical-align:top}#design-table td:first-child{min-width:135px}.gain{color:#12623b}.loss{color:#a33434}tr.new td:first-child{border-left-color:#64748b}h3{scroll-margin-top:100px}@media(max-width:800px){.chart-grid,.model-grid{grid-template-columns:1fr}.chart{padding:6px}.legend{gap:12px}.control{position:static}.step{min-width:135px}}
'''


def narrative(data):
    ss = data['summary']
    faster = sum(s['pfn_xgb_update_ratio'] < 1 for s in ss)
    better = sum(s['tabpfn']['final_macro_f1'] > s['xgb']['final_macro_f1'] for s in ss)
    cost = (f'누적 전체의 4개 구성 중 {faster}개에서 TabPFN의 후속 갱신 합계가 더 짧았다. '
            '가중치 재학습이 없더라도 전체 context 캐시 재구성 비용이 발생한다. '
            '갱신과 최종 test 대량 추론을 분리해서 비교해야 한다.')
    performance = (f'최종 Macro-F1은 4개 구성 중 {better}개에서 TabPFN이 XGB보다 높았다. '
                   '동일한 test에서 데이터·공급 배정별 결과를 비교하며, 전체 평균 외에 정상 오탐률과 공격별 P/R/F1도 확인한다.')
    adoption = []
    for ds, title in [('cic2018', 'CIC2018'), ('toniot', 'ToN')]:
        parts = []
        for s in ss:
            if s['dataset'] != ds:
                continue
            name = '공격 균등' if s['case'] == 'uniform' else '공격 내 자연비율'
            x, p = s['xgb'], s['tabpfn']
            parts.append(f"{name}: TabPFN F1 {100*p['final_macro_f1']:.2f}% / XGB {100*x['final_macro_f1']:.2f}%, "
                         f"후속 갱신 {p['update_sum']:.2f}초 / {x['update_sum']:.2f}초")
        adoption.append(title + ' — ' + '; '.join(parts) + '.')
    adoption.append('현재의 판단 — TabPFN은 사전학습 가중치를 바꾸지 않고 새 공격의 라벨 사례를 반영할 수 있다. '
                    '그러나 이 작동 방식만으로 XGB보다 낮은 갱신 비용이나 필요한 탐지 품질이 보장되지는 않는다. '
                    '갱신·추론 비용을 따로 측정하고, 소수 공격의 탐지 품질을 충족하는 입력 구성과 규모를 확인해야 한다.')
    return dict(cost=cost, performance=performance, adoption=adoption)


def make_page(data, base_page):
    data['scenario_narrative'] = narrative(data)
    css = re.search(r'<style>(.*?)</style>', base_page, re.S).group(1)
    body = BODY
    if all(m['final_stream_rows'] == 100000 for m in data['datasets'].values()):
        body = body.replace('2026-10-13 실험 · 10-14 구성 개편', '2026-10-14 실험 · 최종 입력 100,000행')
        body = body.replace('10/13 실험 · 단일 seed 43', '10/14 실험 · 최종 입력 100,000행 · 단일 seed 43')
        body = body.replace('현재 후속 실험은 <strong>최종 누적 입력을 두 데이터 모두 10만 행으로 확대</strong>하는 것이다.',
                            '이번 실험에서는 <strong>최종 누적 입력을 두 데이터 모두 10만 행으로 확대</strong>했다.')
        body = body.replace('성능 개선 여부와 함께 갱신·추론 비용 증가를 확인한다.',
                            '확대 전후 성능과 비용은 아래 토글에서 비교할 수 있다. 이 결과에 따라 다음 입력 구성과 규모를 판단한다.')
    return ('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>'+TITLE+'</title><style>'+css+EXTRA_CSS+'</style></head><body><main>'+body)


def notes(data):
    n = data['scenario_narrative']
    sizes = ', '.join(f"{ds} {m['final_stream_rows']:,}행" for ds, m in data['datasets'].items())
    paragraphs = ['# EXP71 · 누적 전체의 갱신 비용과 탐지 품질',
        '2026-10-14 사용자 요청으로 보고서의 비교 범위를 누적 전체로 정리했다. 제한 보존 실행과 이전 기록은 원본에 보존하며 이번 화면의 근거로 사용하지 않는다.',
        '목적은 새 공격의 라벨 사례가 확보될 때 TabPFN이 가중치 재학습 없이 유용한 탐지 품질을 유지하면서 갱신 비용을 줄일 수 있는지 확인하는 것이다. 클래스 최초 출현 순서를 유지한 단계별 공급이며 raw timestamp 재생은 아니다. 각 단계의 라벨 공급 후 seen-class 전체 test로 평가한다.',
        f'현재 완료 결과의 최종 입력은 {sizes}다. Benign 75%를 매 단계 나눠 공급하고 공격 25%는 균등 또는 정제 train의 공격 내 자연비율로 배분한다. 과거 사례를 전부 누적하고 각 조건의 두 모델에 동일 학습·test IDs를 제공했다.',
        n['cost'],
        '현재 TabPFN 호출은 매 단계 전체 context로 fit_with_cache를 다시 실행한다. 4개 ensemble의 전처리와 context 캐시를 재구성하므로 가중치 최적화를 하지 않는다는 사실이 갱신 계산량이 없거나 XGB보다 작다는 뜻은 아니다. 입력 선택·모델 생성·fit/cache를 합해 갱신으로 측정하고 초기 구축과 후속 합계를 분리한다. 추론은 전체 test 예측 및 결과 기록 시간으로 별도 측정한다. 전처리와 캐시 생성의 내부 세부 시간은 분리 측정하지 않았으므로 특정 연산의 원인 비중을 주장하지 않는다.',
        n['performance'], *n['adoption'], *data['interpretation'], data['environment_note']]
    if data.get('pending'):
        paragraphs.append(f"10만 행 후속 실험은 실행 중이며 아직 완료 결과가 아니다. 실행: `{data['pending']['run']}`. 기존 공급 IDs를 같은 도입 단계에 유지하면서 새 train 사례를 추가한다. 고정 test와 모델 설정은 유지한다.")
    if data.get('previous'):
        paragraphs.append(f"입력 확대 전 누적 실행 `{data['previous']['run']}`과 동일 test·출현 순서·모델 설정으로 비교했다. scale_validation.json에서 기존 공급 IDs가 같은 확대 단계에 포함되는 것을 검증했다. 일부 이전 worker와 driver/kernel이 달라 비용 변화를 입력 규모만의 인과 효과로 해석하지 않는다.")
    paragraphs += ['후속 판단은 자연비율에서 부족한 공격의 실제 context 표본 수와 균등 조건의 P/R/F1을 함께 확인하여, 입력 확대가 누락된 공격 탐지와 정상 오탐을 동시에 개선하는지 평가하는 것이다. 특정 실패를 사례 수나 context 비율 하나로 확정하지 않는다. 단일 seed이며 같은 개발 holdout을 반복 관찰한 탐색 결과다.',
        f"출처: `{data['run']}/summary.csv`, `class_metrics.csv`, `jobs/*/stage_*/metrics.json`, `validation.json`. HTML: `docs/lablog/reports/20261014.tex`. 프로토콜: `docs/lablog/reports/20261014.tex`."]
    return '\n\n'.join(paragraphs)+'\n'
