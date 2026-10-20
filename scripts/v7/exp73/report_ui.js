'use strict';
const REPORT = JSON.parse(document.getElementById('exp73-data').textContent);
const M = {xgb:'XGB',tabpfn:'TabPFN',ours:'제안 모델',boostpfn:'BoostPFN',localpfn:'LoCalPFN',distpfn:'DistPFN'};
const N = {benign:'Benign',brute_force:'Brute force',web_attacks:'Web attack',infiltration:'Infiltration',bot:'Bot',dos:'DoS',ddos:'DDoS',xss:'XSS',password:'Password',injection:'Injection',backdoor:'Backdoor',scanning:'Scanning',mitm:'MitM',ransomware:'Ransomware'};
const D = {cic2018:'CIC2018',toniot:'ToN'}, C = {natural:'자연비율',uniform:'균등'};
const P = {ideal:'갱신 지연 없음',rate_100:'100행/초',rate_1000:'1,000행/초'};
const COLORS = {xgb:'#c55b00',tabpfn:'#0072b2',ours:'#7b3fa1',boostpfn:'#826b43',localpfn:'#b83d65',distpfn:'#536773'};
const methods=REPORT.methods, el=id=>document.getElementById(id);
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(x,n=0)=>x===null||x===undefined?'—':Number(x).toLocaleString('en-US',{minimumFractionDigits:n,maximumFractionDigits:n});
const pct=(x,n=2)=>x===null||x===undefined?'—':num(x*100,n);
const signed=(x,n=2)=>(x>=0?'+':'')+num(x,n);
const job=(ds,c,m)=>REPORT.jobs[`${ds}_${c}_${m}`];
const result=(ds,c,m,p)=>job(ds,c,m).results[p];
function table(id,headers,rows,start=1,rowClasses=[]){
  el(id).innerHTML='<div class="table-wrap"><table><thead><tr>'+headers.map((v,i)=>`<th${i>=start?' class="numcell" data-num':''}>${esc(v)}</th>`).join('')+'</tr></thead><tbody>'+rows.map((row,r)=>`<tr class="${rowClasses[r]||''}">`+row.map((v,i)=>`<td${i>=start?' class="numcell" data-num':''}>${esc(v)}</td>`).join('')+'</tr>').join('')+'</tbody></table></div>';
}
function classRows(rs){
  const names=[...rs[0].classes].filter(c=>c.support>0).sort((a,b)=>b.support-a.support);
  const maps=rs.map(r=>Object.fromEntries(r.classes.map(c=>[c.name,c])));
  return names.map(c=>[N[c.name],num(c.support),...maps.map(m=>pct(m[c.name].f1))]);
}
function render(){
  const ds=el('dataset').value,c=el('case').value,p=el('policy').value;
  const rs=methods.map(m=>result(ds,c,m,p)), title=`${D[ds]} · ${C[c]} · ${P[p]}`;
  el('active-scope').textContent=`전체 데이터 요약과 S/V 요약은 선택한 평가 정책을 따릅니다. 상세 표·그래프: ${title}.`;
  const overall=[],policy=[];
  for(const d of Object.keys(D))for(const a of Object.keys(C)){
    overall.push([`${D[d]} · ${C[a]}`,...methods.map(m=>pct(result(d,a,m,p).macro_f1))]);
    const r=result(d,a,'ours',p),g=r.global_same_version,q=r.policy;
    policy.push([`${D[d]} · ${C[a]}`,pct(g.macro_f1),pct(r.macro_f1),signed(100*(r.macro_f1-g.macro_f1)),pct(q.call_rate,3),pct(q.accept_rate,3),num(q.called),num(q.helpful),num(q.harmful)]);
  }
  table('overall',['데이터 · 공격 배분',...methods.map(m=>`${M[m]} F1 (%)`)],overall);
  [...el('overall').querySelectorAll('tbody tr')].forEach((row,i)=>{
    const [d,a]=[Object.keys(D)[Math.floor(i/2)],Object.keys(C)[i%2]];
    const vals=methods.map(m=>result(d,a,m,p).macro_f1),best=Math.max(...vals);
    vals.forEach((v,k)=>{if(v===best)row.cells[k+1].classList.add('best');});
  });
  el('performance-chart').innerHTML=`<div class="plot-title">${esc(title)} · Macro-F1 (%)</div>`+methods.map((m,i)=>`<div class="bar-row"><span>${M[m]}</span><div class="bar-track"><div class="bar-fill" style="width:${rs[i].macro_f1*100}%;background:${COLORS[m]}"></div></div><span class="bar-value">${pct(rs[i].macro_f1)}</span></div>`).join('')+'<div class="bar-axis"><span>0</span><span>50</span><span>100</span></div>';
  table('classwise',['클래스','Test 표본 수',...methods.map(m=>`${M[m]} F1 (%)`)],classRows(rs));
  el('class-pr').innerHTML=methods.map(m=>`<h4>${M[m]}</h4><div id="pr-${m}"></div>`).join('');
  methods.forEach((m,i)=>table(`pr-${m}`,['클래스','Test 표본 수','Precision (%)','Recall (%)','F1 (%)'],[...rs[i].classes].sort((a,b)=>b.support-a.support).map(v=>[N[v.name],num(v.support),pct(v.precision),pct(v.recall),pct(v.f1)])));
  const oldStage=el('stage').value;
  el('stage').innerHTML=rs[0].stages.map(s=>`<option value="${s.stage}">${s.stage===0?'초기':`단계 ${s.stage}`} · ${s.new_classes.map(x=>N[x]).join(' · ')||'초기 클래스'}</option>`).join('');
  if([...el('stage').options].some(o=>o.value===oldStage))el('stage').value=oldStage;
  const stageRows=rs[0].stages.map((s,i)=>[s.stage,s.new_classes.map(x=>N[x]).join(' · ')||'초기 클래스',num(s.rows),...rs.map(r=>pct(r.stages[i].macro_f1))]);
  table('stage-summary',['평가 구간','신규 클래스','Test 표본 수',...methods.map(m=>`${M[m]} F1 (%)`)],stageRows,2);
  renderStage();
  table('policy-summary',['데이터 · 공격 배분','내부 Global F1 (%)','제안 F1 (%)','변화 (%p)','호출률 (%)','수락률 (%)','호출 (건)','교정 (건)','훼손 (건)'],policy);
  const ours=result(ds,c,'ours',p),g=ours.global_same_version,pa=ours.policy;
  const gm=Object.fromEntries(g.classes.map(x=>[x.name,x])),pm=Object.fromEntries(pa.classes.map(x=>[x.name,x]));
  table('paired-classes',['클래스','Test 표본 수','Global P (%)','Global R (%)','Global F1 (%)','제안 P (%)','제안 R (%)','제안 F1 (%)','교정 (건)','훼손 (건)'],[...ours.classes].sort((a,b)=>b.support-a.support).map(x=>[N[x.name],num(x.support),...['precision','recall','f1'].map(k=>pct(gm[x.name][k])),...['precision','recall','f1'].map(k=>pct(x[k])),num(pm[x.name].helpful),num(pm[x.name].harmful)]));
  const oj=job(ds,c,'ours');
  table('selections',['학습 버전','도입 클래스','선택된 정책','검증 Macro-F1 변화 (%p)','검증 교정 (건)','검증 훼손 (건)'],oj.fits.map((f,i)=>[i,ours.stages[i].new_classes.map(x=>N[x]).join(' · ')||'초기 클래스',f.selection.reason==='global_fallback'?'Global 유지':'Expert 활성',signed(100*f.selection.delta_macro_f1),num(f.selection.helpful),num(f.selection.harmful)]),3);
  table('policy-stages',['평가 구간','Test 표본 수','호출률 (%)','수락률 (%)','교정 (건)','훼손 (건)'],pa.stages.map(s=>[s.stage,num(s.rows),pct(s.call_rate,3),pct(s.accept_rate,3),num(s.helpful),num(s.harmful)]));
  const costs=methods.map((m,i)=>{
    const j=job(ds,c,m),r=rs[i];return [M[m],num(j.updates[0],2),num(j.updates.slice(1).reduce((a,b)=>a+b,0),2),num(j.updates.at(-1),2),num(r.predict_service_seconds,2),num(r.latency?.p95_seconds,3)];
  });
  table('costs',['모델','초기 구축 (초)','후속 갱신 합계 (초)','최종 갱신 1회 (초)','재생 추론 합계 (초)','응답 p95 (초)'],costs,1,methods.map(m=>m==='ours'?'ours-row':''));
  el('fit-costs').innerHTML=methods.map(m=>`<h4>${M[m]}</h4><div id="fit-${m}"></div>`).join('');
  methods.forEach(m=>{
    const j=job(ds,c,m);
    table(`fit-${m}`,['버전','누적 train 행','Global context 행','제공 가능 validation 행','사용 validation 행','갱신 (초)'],j.fits.map((f,i)=>[i,num(f.train_rows),num(m==='ours'?f.global_rows:(m==='tabpfn'||m==='distpfn'?f.train_rows:null)),num(f.validation_available_rows??oj.fits[i].validation_available_rows),num(f.validation_used_rows??0),num(j.updates[i],3)]));
  });
  table('delay-effects',['모델','지연 없음 F1 (%)','100행/초 F1 (%)','1,000행/초 F1 (%)','1,000행/초 변화 (%p)','구버전 처리 행 (1,000행/초)'],methods.map(m=>{
    const z=job(ds,c,m).results;return [M[m],pct(z.ideal.macro_f1),pct(z.rate_100.macro_f1),pct(z.rate_1000.macro_f1),signed(100*(z.rate_1000.macro_f1-z.ideal.macro_f1),3),num(z.rate_1000.latency.rows_using_outdated_version)];
  }));
  const latency=[];
  for(const q of ['rate_100','rate_1000'])for(const m of methods){const l=result(ds,c,m,q).latency;latency.push([M[m],P[q],num(l.mean_seconds,3),num(l.p95_seconds,3),num(l.queue_wait_max_seconds,3),num(l.drain_after_last_arrival_seconds,3)]);}
  table('latencies',['모델','유입률','평균 응답 (초)','응답 p95 (초)','최대 큐 대기 (초)','입력 종료 후 잔여 (초)'],latency,2);
  table('error-metrics',['모델','Macro-F1 (%)','공격 탐지 Recall (%)','Benign 오탐 (%)','공격→Benign (건)','다른 공격 유형 (건)'],methods.map((m,i)=>[M[m],pct(rs[i].macro_f1),pct(rs[i].attack_detection_recall),pct(rs[i].benign_false_positive_rate,3),num(rs[i].attack_to_benign),num(rs[i].wrong_attack_family)]));
}
function renderStage(){
  const ds=el('dataset').value,c=el('case').value,p=el('policy').value,s=Number(el('stage').value);
  table('stage-classes',['클래스','구간 Test 표본 수',...methods.map(m=>`${M[m]} F1 (%)`)],classRows(methods.map(m=>result(ds,c,m,p).stages[s])));
}
for(const id of ['dataset','case','policy'])el(id).addEventListener('change',render);
el('stage').addEventListener('change',renderStage);
render();
