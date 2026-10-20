"""Add the completed cumulative-history comparison to the existing EXP71 report."""
import json
import re


SECTION = r'''<section id="cumulative-comparison">
<h2>누적 전체 추가 실험 · 같은 사례를 모두 남기면</h2>
<div class="callout" id="comparison-verdict"></div>
<p class="note" id="comparison-environment"></p>
<p>신규 공급 사례·출현 순서·test·모델 설정은 그대로 두고 <strong>과거 사례를 버리지 않았다.</strong> 최종 입력은 CIC2018 22,280행, ToN 36,864행이다. 아래는 초기 구축을 제외한 후속 갱신 시간 합계와 최종 성능이다. 비용은 단계별 1회 관측값이다.</p>
<div class="table-wrap"><table id="comparison-summary"><thead><tr><th>데이터 · 배정 · 모델</th><th>제한 입력</th><th>누적 입력</th><th>제한 갱신(초)</th><th>누적 갱신(초)</th><th>제한 F1(%)</th><th>누적 F1(%)</th><th>제한 정상 오탐(%)</th><th>누적 정상 오탐(%)</th></tr></thead><tbody></tbody></table></div>
<div id="comparison-findings"></div>
<p class="note">제한 조건은 직전 학습 집합에서 최대 4,096행 보존 후 신규 사례를 추가한다. 누적 조건은 도입 시점까지 공급된 모든 사례를 사용한다. 입력량과 클래스별 잔존 비율이 함께 달라지므로 결과를 데이터 양 하나의 효과로 해석하지 않는다. 아래 기존 제한 조건의 상세 기록도 보존했다.</p>
<div class="control"><label>데이터 <select id="cmp-dataset"><option value="cic2018">CIC2018</option><option value="toniot" selected>ToN</option></select></label><label>배정 <select id="cmp-case"><option value="uniform">공격 균등</option><option value="natural" selected>공격 내 자연</option></select></label><label>모델 <select id="cmp-method"><option value="tabpfn">TabPFN</option><option value="xgb">XGBoost</option></select></label><label>단계 <select id="cmp-stage"></select></label></div>
<h3>선택한 모델의 단계별 변화</h3>
<div class="table-wrap"><table id="comparison-stages"><thead><tr><th>단계 · 새 공격</th><th>제한 입력</th><th>누적 입력</th><th>제한 갱신(초)</th><th>누적 갱신(초)</th><th>제한 F1(%)</th><th>누적 F1(%)</th><th>누적 시 교정</th><th>누적 시 훼손</th></tr></thead><tbody></tbody></table></div>
<p class="note">여기서 교정·훼손은 <strong>같은 단계의 제한 보존 → 누적 보존</strong> 예측을 동일 test 행에서 비교한 값이다. 뒤의 기존 기록에 있는 갱신 전후 비교와 구별한다. 단계가 늘면 평가 클래스도 늘어나므로 서로 다른 단계의 전체 F1을 직접적인 유지 지표로 사용하지 않는다.</p>
<h3>선택한 단계의 클래스별 성능과 보존 사례</h3>
<p id="comparison-selection"></p>
<div class="table-wrap"><table id="comparison-classes"><thead><tr><th>클래스</th><th>Test 행</th><th>제한 사례</th><th>누적 사례</th><th>제한 P(%)</th><th>제한 R(%)</th><th>제한 F1(%)</th><th>누적 P(%)</th><th>누적 R(%)</th><th>누적 F1(%)</th><th>ΔF1(%p)</th></tr></thead><tbody></tbody></table></div>
<p class="note">클래스는 해당 단계 test 표본 수 내림차순이다. 사례 수는 선택한 단계의 실제 학습/context 행 수다. 최초 공급량과 구별한다.</p>
<details><summary>선택한 단계의 비용 분해와 추론 시간</summary><div class="table-wrap"><table id="comparison-cost"><thead><tr><th>항목</th><th>제한 보존</th><th>누적 전체</th></tr></thead><tbody></tbody></table></div><p class="note">전체 추론은 동일 test의 예측·결과 기록 시간이다. 첫 batch는 최대 32,768행이며 단일 행 지연 시간이 아니다. 전처리·context 캐시 재구성은 갱신 시간에 포함한다.</p></details>
<p><strong>누적 실행:</strong> <code id="cumulative-run-path"></code></p>
<p id="cumulative-source-links"></p>
<p>10만 행 공급 확대는 아직 실행하지 않았다. 누적 조건에서 소수 공격의 오탐·누락이 어떻게 변했는지를 확인한 뒤, 같은 평가 집합에서 공급 규모를 확대하는 단계로 이어간다.</p>
</section>'''


JAVASCRIPT = r'''
function cmpRecords(ds,c,m){return DATA.cumulative.results.filter(r=>r.dataset===ds&&r.case===c&&r.method===m).sort((a,b)=>a.stage-b.stage);}
function cmpSelected(){return {ds:el('cmp-dataset').value,c:el('cmp-case').value,m:el('cmp-method').value,s:Number(el('cmp-stage').value)};}
function resetComparison(){const {ds,c,m}=cmpSelected(),rr=cmpRecords(ds,c,m);el('cmp-stage').innerHTML=rr.map(r=>`<option value="${r.stage}">${r.stage===0?'초기':r.stage+' · '+r.new_classes.map(n=>N[n]).join(' / ')}</option>`).join('');el('cmp-stage').value=rr.at(-1).stage;renderComparison();}
function renderComparison(){
 const {ds,c,m,s}=cmpSelected(),bs=records(ds,c,m),cs=cmpRecords(ds,c,m),b=bs[s],a=cs[s];
 el('comparison-selection').textContent=`${DN[ds]} · ${CN[c]} · ${m==='tabpfn'?'TabPFN':'XGBoost'} · ${s===0?'초기':s+'단계'} / 같은 test ${num(a.rows)}행`;
 el('comparison-stages').tBodies[0].innerHTML=cs.map((r,i)=>{const q=bs[i],v=DATA.policy_comparison.stages.find(v=>v.dataset===ds&&v.case===c&&v.method===m&&v.stage===i);return row([i===0?'초기':i+' · '+r.new_classes.map(n=>N[n]).join(' / '),num(q.context_rows),num(r.context_rows),dec(q.update_seconds),dec(r.update_seconds),pct(q.macro_f1),pct(r.macro_f1),num(v.corrected),num(v.damaged)]);}).join('');
 const bm=Object.fromEntries(b.classes.map(r=>[r.name,r]));
 el('comparison-classes').tBodies[0].innerHTML=[...a.classes].sort((x,y)=>y.support-x.support).map(r=>{const q=bm[r.name],d=100*(r.f1-q.f1);return `<tr class="${a.new_classes.includes(r.name)?'new':''}">${[N[r.name],num(r.support),num(b.fit_class_counts[r.name]),num(a.fit_class_counts[r.name]),pct(q.precision),pct(q.recall),pct(q.f1),pct(r.precision),pct(r.recall),pct(r.f1)].map(v=>cell(v)).join('')}${cell((d>0?'+':'')+dec(d),d>0?'gain':d<0?'loss':'')}</tr>`;}).join('');
 const fields=[['입력 구성(초)','select_seconds'],['모델 생성(초)','construct_seconds'],['학습 / 전처리·캐시(초)','fit_seconds'],['갱신 합계(초)','update_seconds'],['전체 test 추론(초)','predict_seconds'],['첫 batch 추론(초)','first_batch_seconds']];
 el('comparison-cost').tBodies[0].innerHTML=fields.map(([label,k])=>row([label,dec(b[k]),dec(a[k])])).join('');
}
function setupComparison(){
 if(!DATA.cumulative)return;
 el('comparison-summary').tBodies[0].innerHTML=DATA.cumulative.summary.flatMap(a=>{const b=DATA.summary.find(b=>b.dataset===a.dataset&&b.case===a.case);return ['xgb','tabpfn'].map(m=>row([DN[a.dataset]+' · '+CN[a.case]+' · '+(m==='xgb'?'XGB':'TabPFN'),num(b[m].final_context_rows),num(a[m].final_context_rows),dec(b[m].update_sum),dec(a[m].update_sum),pct(b[m].final_macro_f1),pct(a[m].final_macro_f1),rate(b[m].final_benign_fpr),rate(a[m].final_benign_fpr)]));}).join('');
 el('comparison-verdict').textContent=DATA.comparison_text[0];
 el('comparison-environment').textContent=DATA.comparison_environment_note;
 for(const text of DATA.comparison_text.slice(1)){const p=document.createElement('p');p.textContent=text;el('comparison-findings').appendChild(p);}
 el('cumulative-run-path').textContent=DATA.cumulative.run;
 const sources=[['누적 단계별 CSV',DATA.cumulative.run+'/summary.csv'],['누적 클래스별 CSV',DATA.cumulative.run+'/class_metrics.csv'],['비교 해석 기록','docs/research/20261013/exp71_cumulative_results.md']];
 el('cumulative-source-links').innerHTML=sources.map(([label,path])=>location.protocol==='data:'?`${label}: <code>${path}</code>`:`<a href="../../${path}">${label}</a>`).join(location.protocol==='data:'?'<br>':' · ');
 for(const id of ['cmp-dataset','cmp-case','cmp-method'])el(id).addEventListener('change',resetComparison);
 el('cmp-stage').addEventListener('change',renderComparison);resetComparison();
}
setupComparison();
'''


def comparison_text(data):
    cumulative=data['cumulative'];paragraphs=[]
    faster=sum(r['pfn_xgb_update_ratio']<1 for r in cumulative['summary'])
    paragraphs.append(f'누적 전체의 4개 구성 중 {faster}개에서 TabPFN의 후속 갱신 시간 합계가 XGBoost보다 짧았다. '
        '아래에서는 같은 모델의 제한/누적 차이와, 누적 조건 안에서의 두 모델 차이를 함께 확인한다. '
        '유지 사례 수를 늘렸을 때 정상 오탐과 소수 공격 탐지가 함께 개선되는지가 이번 비교의 판단 기준이다.')
    for a in cumulative['summary']:
        b=next(b for b in data['summary'] if (b['dataset'],b['case'])==(a['dataset'],a['case']))
        name=('CIC2018' if a['dataset']=='cic2018' else 'ToN')+' · '+('균등' if a['case']=='uniform' else '공격 내 자연')
        parts=[]
        for m,label in [('xgb','XGB'),('tabpfn','TabPFN')]:
            old,new=b[m],a[m]
            parts.append(f"{label}의 최종 F1은 {100*old['final_macro_f1']:.2f}%→{100*new['final_macro_f1']:.2f}%, "
                f"후속 갱신 합계는 {old['update_sum']:.2f}초→{new['update_sum']:.2f}초, "
                f"정상 오탐은 {100*old['final_benign_fpr']:.4f}%→{100*new['final_benign_fpr']:.4f}%다")
        paragraphs.append(name+': '+'; '.join(parts)+'.')
    for name,label in [('mitm','MitM'),('ransomware','Ransomware')]:
        parts=[]
        for method,mn in [('xgb','XGB'),('tabpfn','TabPFN')]:
            rs=[]
            for d in [data,cumulative]:
                r=next(r for r in d['results'] if (r['dataset'],r['case'],r['method'],r['stage'])==('toniot','natural',method,5))
                q=next(q for q in r['classes'] if q['name']==name)
                rs.append((r,q))
            (b,bq),(a,aq)=rs;i=a['seen_classes'].index(name);ben=a['seen_classes'].index('benign')
            parts.append(f"{mn}: F1 {100*bq['f1']:.2f}%→{100*aq['f1']:.2f}%, "
                f"P {100*bq['precision']:.2f}%→{100*aq['precision']:.2f}%, R {100*bq['recall']:.2f}%→{100*aq['recall']:.2f}%, "
                f"Benign→{label} {b['confusion_matrix'][ben][i]:,}건→{a['confusion_matrix'][ben][i]:,}건")
        paragraphs.append(f"ToN 자연비율의 {label}: 최종 사용 사례는 제한 {b['fit_class_counts'][name]:,}개, 누적 {a['fit_class_counts'][name]:,}개다. "+'; '.join(parts)+'.')
    paragraphs.append('MitM의 신규 11개는 두 보존 정책에서 같다. 따라서 이 비교에서 관찰한 MitM 변화는 신규 MitM 라벨을 더 확보한 효과가 아니다. '
        '다만 과거 정상·공격 사례의 수, 다양성 및 클래스 비율이 함께 바뀌므로 그중 하나를 단독 원인으로 확정하지 않는다. '
        'Ransomware는 마지막 잔존 수가 1개에서 8개로 달라져, 보존 정책에 따른 소수 클래스 유지 여부를 함께 확인할 수 있다.')
    return paragraphs


def environment_note(data):
    changes=data['policy_comparison'].get('system_environment_changes',{})
    if not changes:return '비교 실행의 GPU·드라이버·커널을 확인했다.'
    return ('재개 환경: ToN 자연비율의 누적 조건은 재부팅 후 커널 7.0.0-38 / NVIDIA 580.178.04에서 두 모델을 처음부터 다시 측정했다. '
            '제한 조건과 나머지 누적 조건은 커널 7.0.0-31 / NVIDIA 580.173.02에서 측정했다. '
            'GPU·Python 패키지·모델 소스·checkpoint는 같지만, ToN 자연비율의 제한→누적 비용 차이를 보존 정책만의 효과로 해석하지 않는다.')


def extend_page(page, data):
    data['comparison_environment_note']=environment_note(data)
    page=page.replace('제한된 과거 사례와 갱신 비용','제한·누적 보존과 갱신 비용')
    page=re.sub(r'<p class="lead">.*?</p>',
        '<p class="lead">새 공격 사례를 반영할 때, <strong>과거 일부 보존과 전체 누적</strong>에서 TabPFN의 context 갱신은 같은 사례로 다시 학습하는 XGBoost와 비교해 어떤 비용과 탐지 품질을 보이는가?</p>',page,count=1)
    page=page.replace('<nav><a href="#finding">','<nav><a href="#cumulative-comparison">제한·누적 비교</a><a href="#finding">',1)
    page=page.replace('<section id="finding">',SECTION+'\n<section id="finding">',1)
    page=page.replace('1. 갱신 비용과 탐지 품질을 함께 판단','1. 기존 제한 보존 결과 · 갱신 비용과 탐지 품질')
    page=page.replace('해당 조건은 아직 실행하지 않았다.','해당 비교는 완료했으며 이 탭 상단의 누적 전체 추가 실험에서 확인할 수 있다.')
    page=page.replace('다음 누적 전체 조건에서는','추가한 누적 전체 조건에서는')
    page=page.replace('비용 절감과 탐지 품질이 함께 유지되는지 확인하려면, 다음에는 같은 사례를 누적 보존한 결과와 비교해야 한다.',
        '이 결과는 제한 보존 조건이다. 같은 사례의 누적 전체 비교는 탭 상단에 추가했다.')
    pos=page.rfind('</script>');page=page[:pos]+JAVASCRIPT+page[pos:]
    return page


def notes(data):
    text=['# EXP71 · 같은 공급 사례의 제한/누적 보존 비교','',
        'Seed 43, 동일 신규 공급 IDs·출현 순서·정제 test·모델 설정을 사용했다. 과거 최대 4,096행 보존과 도입 시점까지의 전체 보존을 비교한다. '
        '초기 구축과 후속 갱신, 추론 비용을 구별하며 expert/S/V는 포함하지 않는다. 10만 행 공급 확대는 아직 실행하지 않았다.','']
    text.extend([environment_note(data),''])
    for paragraph in data['comparison_text']:text.extend([paragraph,''])
    text.extend(['단일 seed와 단계별 1회 비용 관측이다. 같은 test를 반복 관찰한 개발 실험이며 최종 논문 성능 우위로 자동 채택하지 않는다. '
        '후속 규모 확대에서는 같은 평가 집합과 출현 순서를 유지하면서 공급량을 늘리고, 과거 보존 범위를 명시해 갱신 비용과 소수 공격의 오탐·누락을 함께 측정한다.','',
        '## 출처','',f"- 제한 실행: `{data['run']}`",f"- 누적 실행: `{data['cumulative']['run']}`",
        f"- 누적 수치: `{data['cumulative']['run']}/summary.csv`, `class_metrics.csv`",
        f"- 정책 간 대응·교정·훼손 검증: `{data['cumulative']['run']}/comparison_validation.json`",
        '- 설계: `docs/research/20261013/exp71_cumulative_protocol.md`',
        '- 보고서: `lablog/html_report/bounded_updates_1013.html`',''])
    return '\n'.join(text)
