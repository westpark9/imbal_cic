"""Verified direct expert evaluation for the K sweep; no label-based subsets."""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

import json
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_order import support_order, permute
import numpy as np
import pandas as pd

from build_exp56_report_section import table

ROOT=repo_root(__file__)
RUN=ROOT/'tabpfn/results/v4/exp63/20260930_exp63_k_sweep_s43'
DOC=ROOT/'docs/research/20260930'
LABELS={'cic2018':'CIC2018','toniot':'ToN-IoT'}
TARGETS={'cic2018':['infiltration','web_attacks'],'toniot':['scanning','mitm','ransomware']}


def read(p):return read_record(Path(p))
def records(df):return json.loads(df.to_json(orient='records',double_precision=15))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def load_sweep():
    from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
    from nfv3_v3_exp51_scorer_target import apply_thresholds
    assert read(RUN/'status.json')['state']=='complete'
    checks=[];data={'seed':43,'ks':[2,4,6,8],'datasets':{},'source':str(RUN.relative_to(ROOT))}
    for ds,title in LABELS.items():
        first=RUN/f'{ds}_k2';names=read(first/'cache/COMPLETE.json')['class_names'];C=len(names)
        anchor=pd.read_csv(first/'contexts/composition.csv').query('role == "anchor"').iloc[0]
        anchors={c:int(anchor[c]) for c in names};banks={}
        baseline_ids={s:np.load(first/f'cache/{s}_ids.npy',mmap_mode='r') for s in ['eval','cal']}
        for k in data['ks']:
            out=RUN/f'{ds}_k{k}';assert (out/'COMPLETE.json').exists()
            classes=pd.read_csv(out/'diagnostics/expert_class_metrics.csv')
            summary=pd.read_csv(out/'diagnostics/expert_summary.csv')
            sys_summary=pd.read_csv(out/'summary.csv')
            if k==4:
                old=ROOT/f'tabpfn/results/v4/exp59/20260928_exp59_residual_oracle_s43/{ds}_s43/fresh_bank'
                composition=pd.read_csv(old/'context_compositions.csv').query('arm == "designed"')
                contexts=[dict(expert=int(r.expert),block={c:int(r[c]) for c in names},anchor=anchors,block_rows=int(r.block_rows),context_rows=int(r.block_rows)+sum(anchors.values())) for _,r in composition.iterrows()]
            else:
                composition=pd.read_csv(out/'contexts/composition.csv')
                contexts=[dict(expert=int(r.expert),block={c:int(r[c]) for c in names},anchor=anchors,block_rows=int(r.rows),context_rows=int(r.rows)+sum(anchors.values())) for _,r in composition.query('role == "block"').iterrows()]
                for _,r in composition.query('role == "anchor"').iterrows():assert {c:int(r[c]) for c in names}==anchors
            models={};confusions={}
            for split,raw,positions in [('full_test','eval',slice(None)),('cal_confirm','cal',np.load(out/'policy/cal_confirm_positions.npy'))]:
                np.testing.assert_array_equal(baseline_ids[raw],np.load(out/f'cache/{raw}_ids.npy',mmap_mode='r'))
                if raw=='cal':np.testing.assert_array_equal(positions,np.load(first/'policy/cal_confirm_positions.npy'))
                y=np.load(out/f'cache/{raw}_y.npy',mmap_mode='r')[positions]
                gp=np.load(out/f'cache/{raw}_p0.npy',mmap_mode='r')[positions].argmax(1)
                models[split]={};confusions[split]={}
                for e in range(k+1):
                    model='global' if not e else f'expert{e}'
                    pred=np.load(out/f'cache/{raw}_p{e}.npy',mmap_mode='r')[positions].argmax(1)
                    cm=confusion_matrix(y,pred,labels=np.arange(C))
                    np.testing.assert_array_equal(cm,np.load(out/f'diagnostics/{split}_{model}_confusion.npy'))
                    p,r,f,s=precision_recall_fscore_support(y,pred,labels=np.arange(C),zero_division=0)
                    frame=classes[(classes.split==split)&(classes.model==model)].set_index('class').loc[names].reset_index()
                    for key,val in [('precision',p),('recall',r),('f1',f),('support',s),('TP',cm.diagonal()),('FP',cm.sum(0)-cm.diagonal()),('FN',cm.sum(1)-cm.diagonal())]:
                        np.testing.assert_allclose(frame[key],val,rtol=0,atol=1e-12)
                    helpful=(pred==y)&(gp!=y);harmful=(pred!=y)&(gp==y)
                    for c,name in enumerate(names):
                        assert frame.iloc[c].helpful==helpful[y==c].sum() and frame.iloc[c].harmful==harmful[y==c].sum()
                    row=records(summary[(summary.split==split)&(summary.model==model)])[0]
                    np.testing.assert_allclose(row['macro_f1'],f.mean(),rtol=0,atol=1e-12)
                    assert row['helpful']==helpful.sum() and row['harmful']==harmful.sum()
                    row['dominant_class']=names[int(cm.sum(0).argmax())]
                    models[split][model]={'classes':records(frame),'summary':row}
                    confusions[split][model]=cm.tolist()
                    checks.append(dict(dataset=ds,K=k,split=split,model=model,rows=len(y),verified=True))
            policy=records(sys_summary.query('split == "full_test" and arm == "s1v0"'))[0]
            y=np.load(out/'cache/eval_y.npy');g=np.load(out/'cache/eval_p0.npy',mmap_mode='r').argmax(1)
            with np.load(out/'policy/eval_s1v0_scores.npz') as score:
                final,calls,accepted=apply_thresholds(g,score['candidate'],score['pre'],score['post'],read(out/'policy/selected_thresholds.json'))
            np.testing.assert_array_equal(final,np.load(out/'policy/s1v0_final.npy'))
            np.testing.assert_allclose(policy['macro_f1'],precision_recall_fscore_support(y,final,labels=np.arange(C),zero_division=0)[2].mean(),rtol=0,atol=1e-12)
            assert int(calls.sum())==policy['proposed'] and int(accepted.sum())==policy['accepted']
            both={c:sum(models['full_test'][f'expert{e}']['classes'][i]['delta_f1']>1e-12 and models['cal_confirm'][f'expert{e}']['classes'][i]['delta_f1']>1e-12 for e in range(1,k+1)) for i,c in enumerate(names)}
            banks[str(k)]=dict(K=k,models=models,confusions=confusions,contexts=contexts,policy=policy,both_improved=both,total_context_rows=sum(c['context_rows'] for c in contexts),single_class_blocks=sum(sum(v>0 for v in c['block'].values())==1 for c in contexts))
        # Report order rule: classes by full-test sample count, descending (scripts/report_order.py).
        order=support_order([c['support'] for c in banks[str(data['ks'][0])]['models']['full_test']['global']['classes']])
        names=permute(names,order)
        for b in banks.values():
            for split in b['models']:
                for model in b['models'][split]:
                    b['models'][split][model]['classes']=permute(b['models'][split][model]['classes'],order)
                    b['confusions'][split][model]=[permute(row,order) for row in permute(b['confusions'][split][model],order)]
        data['datasets'][ds]=dict(title=title,names=names,targets=TARGETS[ds],anchor=anchors,banks=banks)
    dump(DOC/'exp63_capability_numeric_validation.json',dict(expert_populations_checked=len(checks),checks=checks,same_eval_ids=True,same_confirm_ids=True,policy_thresholds_verified=True))
    dump(DOC/'exp63_capability_report_data.json',data)
    return data


CSS='''
#sweep-paired td,#sweep-counts td,#sweep-confusions td,.sweep-matrix td{text-align:right;font-variant-numeric:tabular-nums}#sweep-confusions td:first-of-type{text-align:left}
.sweep-controls{display:flex;gap:14px;flex-wrap:wrap;align-items:end;margin:16px 0;padding:15px;background:#edf1f5;border-radius:7px}.sweep-controls label{font-size:12px;font-weight:700}.sweep-controls select{display:block;margin:5px 0 0}.sweep-matrix{font-size:12px}.sweep-matrix td{text-align:right;vertical-align:middle}.sweep-matrix th:first-child,.sweep-matrix td:first-child{text-align:left;position:sticky;left:0;background:#fff;z-index:1}.sweep-matrix .sweep-selected th:first-child{background:#e5eef5}.sweep-matrix .sweep-positive{background:#edf6f2}.sweep-matrix .sweep-negative{background:#fcf1ed}.sweep-matrix .sweep-global{font-weight:700;background:#f0f3f6}.sweep-matrix th{white-space:normal;min-width:91px}.sweep-delta{display:block;font-size:10px}.sweep-choice{border:0;background:none;color:var(--blue);font:inherit;text-decoration:underline;cursor:pointer;padding:0}.sweep-detail{margin-top:24px;border:1px solid var(--line);border-radius:8px;background:white;padding:20px}.sweep-detail h4{margin-top:0}.sweep-detail td{font-variant-numeric:tabular-nums}.sweep-pair{white-space:nowrap}.sweep-small{font-size:12px;color:var(--muted)}.sweep-stats{display:flex;gap:10px 24px;flex-wrap:wrap;margin:10px 0 16px;font-size:13px}.sweep-stats b{color:var(--blue)}.sweep-overview table{font-size:12px}.sweep-overview td{vertical-align:middle}.sweep-matrix-wrap{margin-bottom:6px}.sweep-matrix-wrap table{min-width:830px}.sweep-paired-table{min-width:980px!important}.sweep-tag{font-size:11px;border-radius:3px;padding:2px 6px;background:#edf1f5;white-space:nowrap}.sweep-explain{font-size:13px}.sweep-detail select{margin-left:0}
@media(max-width:760px){.sweep-detail{padding:12px}.sweep-controls{gap:12px}.sweep-controls label{flex:1 1 40%}.sweep-controls select{width:100%}.sweep-stats{display:block}.sweep-stats span{display:block;margin:4px 0}}
'''


def render(data,labels):
    rows=[]
    for ds,d in data['datasets'].items():
        for k,b in d['banks'].items():
            counts=' / '.join(f'{labels[c]} {b["both_improved"][c]}/{k}' for c in d['targets'])
            rows.append([d['title'],k,counts,f'{b["single_class_blocks"]}/{k}',f'{b["total_context_rows"]:,}',f'{b["policy"]["macro_f1"]:.4f}',f'{100*b["policy"]["proposed"]/b["policy"]["rows"]:.0f}%'])
    return '''<section id="capability"><h2>3. Expert 자체의 분류 역량: 같은 표본에서 Global과 비교</h2>
<p class="source-label">09/30 EXP63 완료 · CIC2018·ToN · seed 43 · K=2·4·6·8 · K=4는 EXP62 fitted bank 재사용</p>
<p class="lede"><b>ToN의 Scanning을 잘 구분하는 expert는 있다. 그러나 expert 수를 늘리는 것만으로 CIC2018 Infiltration·Web attack, ToN MitM·Ransomware의 개선이 확보되지는 않았다.</b></p>
<div class="callout"><b>평가 방법:</b> 모든 expert가 같은 전체 test를 각각 분류한 결과를 Global과 비교한다. 담당영역으로 표본을 골라내지 않고, 다른 클래스를 해당 클래스로 잘못 예측한 <b>FP도 모두 포함</b>한다. 동일 expert의 validation 확인 구간도 함께 본다. S/V 선택·채택은 이 직접 평가에 개입하지 않는다.</div>
<div class="hero-grid"><article class="card"><h3>CIC2018 · 취약 클래스의 개선은 제한적</h3>
<p><b>Infiltration:</b> 네 K 구성 모두, 전체 test에서 Global F1 0.2880을 넘는 expert가 없다. Recall을 높여도 낮은 precision 때문에 F1이 낮아지는 expert가 많다.</p>
<p><b>Web attack:</b> K=4 e1은 test F1 0.2102 → <b>0.3526</b>, precision 0.1185 → 0.2172로 개선한다. 그러나 같은 expert의 validation 확인 구간은 0.7442 → 0.7356이다. 해당 구간의 양성은 33개다.</p>
<p class="note">호출 0건만으로 모든 expert가 무능하다고 할 수는 없다. 다만 취약 클래스에서 양쪽 평가 구간에 걸친 이득은 아직 확인되지 않았다.</p></article>
<article class="card"><h3>ToN · Scanning의 역량과 전체 교체 위험을 구분</h3>
<p><b>Scanning:</b> 모든 K 구성의 모든 expert에서 Global보다 F1이 높고, 같은 expert의 validation 확인 구간에서도 개선 방향이 유지된다.</p>
<p>K=4 e3의 test는 precision <b>0.7722 → 0.8299</b>, recall <b>0.0202 → 0.7024</b>, F1 <b>0.0394 → 0.7609</b>다. 하지만 전체 Macro-F1은 0.6796 → 0.6651이며 훼손이 209,586건이다.</p>
<p class="note">Scanning에 유용한 출력을 만든다는 근거다. 다른 클래스까지 이 expert로 교체해도 안전하다는 뜻은 아니다. MitM·Ransomware는 양쪽 구간에서 함께 개선하는 expert가 없다.</p></article></div>
<details id="assignment-correction"><summary>기존 A/B로 무엇을 알 수 있었고, 왜 이번 평가로 바꿨는가?</summary>
<p>A는 test 정답이 들어가는 residual로 expert를 배정한 <b>oracle 진단</b>, B는 정답 없이 입력 표현·Global 확률로 최근접 expert에 배정한 <b>고정 배정 정책 진단</b>이었다. B는 학습된 S/V의 선택 정책과도 다르다.</p>
<p>두 경우 모두 expert는 고정 context와 test feature로 분류했다. Test 정답이나 Global 확률을 expert의 분류 입력으로 준 것은 아니다. 달랐던 것은 <b>각 expert를 채점하는 표본군</b>이었다.</p>
<p>따라서 A/B 차이에는 배정과 표본 구성의 효과가 섞인다. 이를 근거로 “담당영역에서는 잘하고 유사 클래스 구분은 못 한다”고 결론 내린 이전 해석은 철회한다. 이번 표는 기존의 전체 test 직접 평가를 다시 사용하되, 전체 Macro-F1만으로 판단하지 않고 <b>클래스별 이득·FP와 validation 확인 결과</b>를 함께 제시한다.</p></details>
<h3>K를 늘렸을 때: 취약 클래스의 이득이 함께 확인되는가?</h3>
<p class="sweep-explain">“양쪽 개선”은 <b>같은 expert·같은 클래스</b>가 test와 validation 확인 구간 각각에서 Global보다 높은 F1을 보인 경우다. 두 구간의 절대 F1을 서로 비교하거나, 통계적 유의성·반복 seed 재현성을 뜻하지 않는다.</p>
<div class="sweep-overview">'''+table(['데이터','K','양쪽 개선 expert 수 · 클래스별','단일 클래스 block','전체 context 행 합계','S+V Macro-F1','호출률'],rows,'sweep-overview')+'''</div>
<p class="note">Global·공통 anchor·데이터 분할·선택 규칙·S/V 설정을 고정했다. Context는 공통 anchor + residual 군집의 특화 block이며 Global 정답 사례도 포함한다. Expert당 block 상한은 186,000행이다. 총 context는 anchor 중복을 포함한 행 합계로, K 간 총예산은 같지 않다. K마다 다시 군집화하므로 e1 등의 번호는 K 사이에서 같은 역할을 뜻하지 않는다.</p>
<p class="interpretation">CIC2018은 모든 K에서 S/V 호출 0건이다. ToN S+V는 K=2·4·6·8에서 각각 0.7636·0.7617·0.7677·0.7463이며 모두 호출률 100%다. <b>K 증가에 따른 일관된 개선은 없다.</b> 이 표의 test 최고값으로 K를 선택하지 않는다. 1·4장은 기존 K=4 결과를 설명한다.</p>
<h3>Expert × 클래스: 성능과 context를 연결해서 보기</h3>
<div class="sweep-controls">
<label>데이터<select id="sweep-dataset"><option value="cic2018">CIC2018</option><option value="toniot">ToN-IoT</option></select></label>
<label>Expert 수<select id="sweep-k"><option>2</option><option selected>4</option><option>6</option><option>8</option></select></label>
<label>평가 표본<select id="sweep-split"><option value="full_test">전체 test</option><option value="cal_confirm">Validation 확인 구간</option></select></label>
<label>지표<select id="sweep-metric"><option value="f1">F1</option><option value="precision">Precision</option><option value="recall">Recall</option></select></label>
</div>
<p class="sweep-small" id="sweep-population"></p>
<div id="sweep-matrix" class="wrap sweep-matrix-wrap"></div>
<p class="note">윗줄은 해당 지표, 아랫줄은 같은 표본의 Global 대비 변화다. 녹색은 증가, 주황색은 감소다. Precision과 recall은 함께 읽는다. Expert 이름을 누르면 아래 상세표가 바뀐다.</p>
<article class="sweep-detail" id="sweep-detail"><h4 id="sweep-detail-title"></h4>
<label class="expert-picker">상세 expert <select id="sweep-expert"></select></label>
<p id="sweep-expert-verdict" class="sweep-explain"></p><div id="sweep-stats" class="sweep-stats"></div>
<p id="sweep-context-note" class="sweep-small"></p>
<div id="sweep-paired" class="wrap"></div>
<p class="note">P/R·FP는 전체 test 기준이다. “양쪽 ↑”는 같은 expert의 양쪽 F1 개선을 뜻한다. Context 클래스 비율은 학습 구성이고, 성능 개선 자체를 증명하지 않는다. 모든 클래스가 anchor에 들어가도 특정 클래스의 오탐 억제가 보장되지는 않는다.</p>
<details id="sweep-count-details"><summary>클래스별 TP·FP·FN, 교정·훼손과 주요 혼동</summary><p class="note">교정·훼손은 해당 <b>정답 클래스</b> 안의 변화다. 특정 클래스로 잘못 예측한 FP는 다른 정답 클래스에서 발생하므로 별도로 확인한다.</p><div id="sweep-counts" class="wrap"></div><h4>이 expert가 가장 많이 혼동한 클래스 쌍 · 전체 test</h4><div id="sweep-confusions" class="wrap"></div></details></article>
<h3>이 결과로 다음에 바꿀 것</h3>
<ol class="plan"><li><b>ToN:</b> Scanning의 개선을 보이는 후보를 validation에서 먼저 정하고, 해당 후보가 만드는 Benign 등 다른 클래스의 훼손을 제한하는 선택·채택 기준을 검증한다. MitM·Ransomware는 높은 recall만으로 후보를 채택하지 않고 FP와 precision 개선을 요구한다.</li>
<li><b>CIC2018:</b> Infiltration의 recall 증가가 어떤 클래스의 FP 증가를 동반하는지 상세표의 혼동과 context를 대조한다. 단순히 K를 더 늘리기보다, 취약 클래스의 precision·recall을 함께 개선하는 context 후보를 만든 뒤 validation에서 확인한다.</li>
<li><b>Context 효과의 근거:</b> 같은 평가 표본에서 anchor·block 길이·클래스별 개수를 맞춘 context 비교를 사용한다. 이번 K 비교는 군집·샘플·총예산이 함께 바뀌므로 residual 정보가 결정경계 학습에 도움이 됐는지를 분리해 증명하지 않는다.</li></ol>
<p class="note">선택한 context와 단일 seed에서 관찰한 성능이다. Validation과 test는 클래스 비율이 다르므로 각 구간 내 Global 대비 변화로 읽는다. 기존에 관찰한 test 결과는 원인 탐색에 사용하고, 최종 후보는 test를 보고 선택하지 않는다.</p></section>'''


JS=r'''
const sweep=JSON.parse(document.getElementById('merged-data').textContent).capability_sweep;
const sweepLabels=JSON.parse(document.getElementById('merged-data').textContent).class_names;
const se=id=>document.getElementById('sweep-'+id), sn=v=>Number(v).toLocaleString('en-US'), sf=v=>Number(v).toFixed(4);
const sd=v=>(v>=0?'+':'')+sf(v), sp=v=>(100*v).toFixed(2)+'%';
const sweepTable=(heads,rows,kind,cls='')=>'<table class="'+cls+'" data-kind="'+kind+'"><thead><tr>'+heads.map(h=>'<th scope="col">'+h+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map((v,i)=>'<'+(i?'td':'th')+(i?'':' scope="row"')+'>'+v+'</'+(i?'td':'th')+'>').join('')+'</tr>').join('')+'</tbody></table>';
function sweepState(){const d=sweep.datasets[se('dataset').value];return {d,b:d.banks[se('k').value],split:se('split').value,metric:se('metric').value,e:se('expert').value||'1'};}
function renderSweepDetail(){
 const {d,b,e}=sweepState(),model='expert'+e,ctx=b.contexts[Number(e)-1];
 const t=b.models.full_test[model],v=b.models.cal_confirm[model],g=b.models.full_test.global,vg=b.models.cal_confirm.global;
 se('detail-title').textContent=d.title+' · K='+b.K+' · e'+e+' — 같은 expert의 test·validation 비교';
 const both=d.names.filter((c,i)=>t.classes[i].delta_f1>1e-12&&v.classes[i].delta_f1>1e-12);
 const only=d.names.filter((c,i)=>t.classes[i].delta_f1>1e-12&&v.classes[i].delta_f1<=1e-12);
 se('expert-verdict').textContent='양쪽 구간에서 F1 개선: '+(both.map(c=>sweepLabels[c]).join(', ')||'없음')+'. Test에서만 F1 개선: '+(only.map(c=>sweepLabels[c]).join(', ')||'없음')+'. 전체 지표와 다른 클래스의 손실은 아래에서 함께 확인한다.';
 const s=t.summary;
 se('stats').innerHTML='<span>Test Macro-F1 <b>'+sf(g.summary.macro_f1)+' → '+sf(s.macro_f1)+'</b></span><span>교정 <b>'+sn(s.helpful)+'</b> / 훼손 <b>'+sn(s.harmful)+'</b></span><span>출력 클래스 <b>'+s.predicted_class_count+'/'+d.names.length+'</b></span><span>최다 출력 <b>'+sweepLabels[s.dominant_class]+' '+sp(s.dominant_prediction_share)+'</b></span>';
 const major=Object.entries(ctx.block).filter(([c,n])=>n>0).sort((a,b)=>b[1]-a[1]);
 se('context-note').textContent='공통 anchor '+sn(Object.values(ctx.anchor).reduce((a,b)=>a+b,0))+' + 특화 block '+sn(ctx.block_rows)+' = '+sn(ctx.context_rows)+'행. Block '+major.length+'개 클래스; 주요 구성: '+major.slice(0,3).map(([c,n])=>sweepLabels[c]+' '+sn(n)+' ('+sp(n/ctx.block_rows)+')').join(', ')+'.';
 const paired=d.names.map((c,i)=>{
  const a=t.classes[i],z=v.classes[i],base=g.classes[i],vb=vg.classes[i],improved=a.delta_f1>1e-12&&z.delta_f1>1e-12;
  const pair=(x,y)=>'<span class="sweep-pair">'+sf(x)+' → '+sf(y)+'</span>';
  return [sweepLabels[c]+(improved?'<br><span class="sweep-tag">양쪽 ↑</span>':''),sn(ctx.anchor[c])+' + '+sn(ctx.block[c])+'<small class="count-note">전체 context의 '+sp((ctx.anchor[c]+ctx.block[c])/ctx.context_rows)+'</small>',pair(base.f1,a.f1)+'<small class="count-note">양성 '+sn(a.support)+'</small>',pair(vb.f1,z.f1)+'<small class="count-note">양성 '+sn(z.support)+'</small>',sf(a.precision),sf(a.recall),sn(base.FP)+' → '+sn(a.FP)];
 });
 se('paired').innerHTML=sweepTable(['클래스','Context · anchor + block','Test F1 · G → E','Validation 확인 F1 · G → E','Test P','Test R','Test FP · G → E'],paired,'sweep-paired','sweep-paired-table');
 const counts=d.names.map((c,i)=>{const a=t.classes[i],base=g.classes[i];return [sweepLabels[c],sn(a.support),sn(base.TP)+' → '+sn(a.TP),sn(base.FP)+' → '+sn(a.FP),sn(base.FN)+' → '+sn(a.FN),sn(a.helpful),sn(a.harmful)];});
 se('counts').innerHTML=sweepTable(['정답 클래스','Test 수','TP · G → E','FP · G → E','FN · G → E','교정','훼손'],counts,'sweep-counts');
 const cm=b.confusions.full_test[model],gm=b.confusions.full_test.global,pairs=[];
 cm.forEach((r,i)=>r.forEach((n,j)=>{if(i!==j&&n>0)pairs.push({i,j,n,g:gm[i][j]});}));
 pairs.sort((a,b)=>b.n-a.n||a.i-b.i||a.j-b.j);
 se('confusions').innerHTML=sweepTable(['정답 클래스','예측 클래스','Global 건수','Expert 건수','해당 정답 클래스 내 비율'],pairs.slice(0,6).map(p=>[sweepLabels[d.names[p.i]],sweepLabels[d.names[p.j]],sn(p.g),sn(p.n),sp(p.n/t.classes[p.i].support)]),'sweep-confusions');
 for(const button of se('matrix').querySelectorAll('.sweep-choice')){const active=button.dataset.expert===e;button.setAttribute('aria-pressed',String(active));button.closest('tr').classList.toggle('sweep-selected',active);}
}
function renderSweep(reset=false){
 const {d,b,split,metric,e}=sweepState(),keep=reset?'1':String(Math.min(Number(e),b.K));
 se('expert').innerHTML=Array.from({length:b.K},(_,i)=>'<option value="'+(i+1)+'">e'+(i+1)+'</option>').join('');se('expert').value=keep;
 const scope=b.models[split],base=scope.global;
 se('population').textContent=d.title+' · K='+b.K+' · '+(split==='full_test'?'전체 test':'Validation 확인 구간')+' '+sn(base.summary.rows)+'행. Global과 모든 expert의 평가 표본이 같다.';
 const head='<thead><tr><th scope="col">분류기</th>'+d.names.map(c=>'<th scope="col">'+sweepLabels[c]+'</th>').join('')+'</tr></thead>';
 const rows=Object.entries(scope).map(([model,a])=>'<tr class="'+(model==='global'?'sweep-global':'')+'"><th scope="row">'+(model==='global'?'Global':'<button type="button" class="sweep-choice" data-expert="'+model.slice(6)+'">e'+model.slice(6)+'</button>')+'</th>'+a.classes.map((c,i)=>{const delta=c[metric]-base.classes[i][metric];return '<td data-class="'+c.class+'" data-model="'+model+'" class="'+(delta>1e-12?'sweep-positive':delta< -1e-12?'sweep-negative':'')+'"><span>'+sf(c[metric])+'</span><small class="sweep-delta">'+(model==='global'?'기준':sd(delta))+'</small></td>';}).join('')+'</tr>').join('');
 se('matrix').innerHTML='<table class="sweep-matrix" data-kind="sweep-matrix">'+head+'<tbody>'+rows+'</tbody></table>';
 for(const button of se('matrix').querySelectorAll('.sweep-choice'))button.addEventListener('click',()=>{se('expert').value=button.dataset.expert;renderSweepDetail();});
 renderSweepDetail();
}
for(const id of ['dataset','k','split','metric'])se(id).addEventListener('change',()=>renderSweep(id==='dataset'||id==='k'));
se('expert').addEventListener('change',renderSweepDetail);renderSweep(true);
'''
