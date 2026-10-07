#!/usr/bin/env python3
"""Validate completed SOTA predictions and publish the 0908 report update."""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

import base64
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_order import support_order, permute, apply_rules

ROOT=repo_root(__file__)
RUN=ROOT/'tabpfn/results/v4/exp61/20260929_exp61_sota_local_s43'
OUT=ROOT/'docs/research/20260930'
REPORT=ROOT/'lablog/html_report/sota_pfn_comparison.html'
INDEX=REPORT.with_name('post_tabpfn.html')
METHODS=['global_raw','xgb','boostpfn','localpfn','distpfn','xgb_full']
NAMES=dict(global_raw='Global TabPFN v3',xgb='XGBoost · 100k',boostpfn='BoostPFN',localpfn='LoCalPFN · FT',distpfn='DistPFN · v3',xgb_full='XGBoost · 전체 train')
OURS_RUN=ROOT/'tabpfn/results/v4/exp63/20260930_exp63_k_sweep_s43'
OURS_K=6
OURS='ours'
NAMES[OURS]=f'제안 모델 · K={OURS_K} · S+V'
SHARED=[m for m in METHODS if m!='xgb_full']  # methods eligible for the shared-100k best mark
LABELS=dict(benign='Benign',bot='Bot',brute_force='Brute force',ddos='DDoS',dos='DoS',infiltration='Infiltration',web_attacks='Web attack',backdoor='Backdoor',injection='Injection',mitm='MitM',password='Password',ransomware='Ransomware',scanning='Scanning',xss='XSS')
TITLES=dict(cic2018='CIC2018',toniot='ToN-IoT')


def collect():
    status=read_record(RUN/'status.json')
    assert status['state']=='complete' and not status['failed'] and len(status['completed'])==10
    data=dict(seed=43,methods=METHODS,method_names=NAMES,class_names=LABELS,datasets={},run=str(RUN.relative_to(ROOT)),completed_epoch=status['updated_epoch'])
    checks=[]
    for ds,title in TITLES.items():
        results={};reference=None;context=None
        for method in ['xgb','xgb_full','distpfn','boostpfn','localpfn']:
            folder=RUN/f'{ds}_{method}'
            assert read_record(folder/'COMPLETE.json')['synthetic'] is False
            identity=np.load(folder/'evaluation_identity.npz');ids=identity['test_ids'];y=identity['y']
            if reference is None:reference=(ids.copy(),y.copy())
            assert np.array_equal(ids,reference[0]) and np.array_equal(y,reference[1])
            if method!='xgb_full':
                if context is None:context=identity['train_ids'].copy()
                assert len(context)==100000 and np.array_equal(identity['train_ids'],context)
            entries=read_record(folder/'results.json')
            for r in entries:
                assert not r['synthetic'] and r['rows']==len(y)
                pred=np.load(folder/f'{r["method"]}_pred.npy');C=len(r['classes'])
                cm=np.bincount(y.astype('int64')*C+pred,minlength=C*C).reshape(C,C)
                for i,c in enumerate(r['classes']):
                    assert c['TP']==int(cm[i,i]) and c['FP']==int(cm[:,i].sum()-cm[i,i]) and c['FN']==int(cm[i,:].sum()-cm[i,i])
                    assert c['support']==int(cm[i,:].sum())
                    p=c['TP']/(c['TP']+c['FP']) if c['TP']+c['FP'] else 0
                    recall=c['TP']/c['support'] if c['support'] else 0
                    f1=2*c['TP']/(2*c['TP']+c['FP']+c['FN']) if 2*c['TP']+c['FP']+c['FN'] else 0
                    assert abs(c['precision']-p)<1e-12 and abs(c['recall']-recall)<1e-12 and abs(c['f1']-f1)<1e-12
                assert abs(np.mean([c['f1'] for c in r['classes']])-r['macro_f1'])<1e-12
                assert abs(np.trace(cm)/cm.sum()-r['accuracy'])<1e-12
                results[r['method']]=r
                checks.append(f'{ds}/{r["method"]}: full-test predictions, counts and metrics matched')
        assert set(results)==set(METHODS)
        order=support_order([c['support'] for c in results['global_raw']['classes']])   # report order rule: test count desc
        for r in results.values():r['classes']=permute(r['classes'],order)
        data['datasets'][ds]=dict(title=title,results=results,class_order=[c['name'] for c in results['global_raw']['classes']],test_rows=len(reference[0]))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'exp61_report_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (OUT/'exp61_numeric_validation.json').write_text(json.dumps(dict(checks=checks,methods=12,classes=sum(len(d['class_order'])*6 for d in data['datasets'].values()),source_sha256={f:hashlib.sha256((RUN/f).read_bytes()).hexdigest() for f in ['summary.csv','class_metrics.csv','request.json']}),ensure_ascii=False,indent=2)+'\n')
    return data


def collect_ours(data):
    """Proposed model (EXP63 K sweep, S1V0 policy) recomputed from saved full-test predictions and labels."""
    sweep=read_record(OUT/'exp63_capability_report_data.json')
    ours=dict(K=OURS_K,run=str(OURS_RUN.relative_to(ROOT)),policy='S1V0 · class-balanced direct decision-gain scorer + NLL-gain quantile verifier · 임계값은 validation에서 선택',
              k_choice='EXP63 K=2·4·6·8 중 ToN test Macro-F1 최고 K (사용자 요청 표기 · test 기준 선택 · 단일 seed)',datasets={})
    for ds,d in data['datasets'].items():
        base=OURS_RUN/f'{ds}_k{OURS_K}'
        names=read_record(base/'cache/COMPLETE.json')['class_names'];C=len(names)
        y=np.load(base/'cache/eval_y.npy').astype('int64');pred=np.load(base/'policy/s1v0_final.npy').astype('int64')
        identity=np.load(RUN/f'{ds}_xgb/evaluation_identity.npz')
        assert np.array_equal(np.load(base/'cache/eval_ids.npy'),identity['test_ids']) and np.array_equal(y,identity['y'])
        cm=np.bincount(y*C+pred,minlength=C*C).reshape(C,C)
        metrics=pd.read_csv(base/'class_metrics.csv');metrics=metrics[(metrics.arm=='s1v0')&(metrics.split=='full_test')].set_index('class')
        summary=pd.read_csv(base/'summary.csv');row=summary[(summary.arm=='s1v0')&(summary.split=='full_test')].iloc[0]
        classes=[]
        for name in d['class_order']:
            i=names.index(name);tp=int(cm[i,i]);fp=int(cm[:,i].sum()-tp);fn=int(cm[i,:].sum()-tp);support=int(cm[i,:].sum())
            p=tp/(tp+fp) if tp+fp else 0;r=tp/support if support else 0;f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
            m=metrics.loc[name];assert int(m['support'])==support
            assert abs(m['precision']-p)<1e-9 and abs(m['recall']-r)<1e-9 and abs(m['system_f1']-f1)<1e-9
            classes.append(dict(name=name,support=support,TP=tp,FP=fp,FN=fn,precision=p,recall=r,f1=f1))
        macro=float(np.mean([c['f1'] for c in classes]));acc=float(np.trace(cm)/cm.sum())
        assert abs(macro-row['macro_f1'])<1e-9 and abs(acc-row['accuracy'])<1e-9
        cost=read_record(base/'cost.json');phases={p['phase']:p['seconds'] for p in cost['phases']}
        expert_predict=sum(v for k,v in phases.items() if k.endswith('_eval') or k=='test_policy_scoring')
        preparation=sum(v for k,v in phases.items() if not (k.endswith('_eval') or k in ('test_policy_scoring','input_loading')))
        g=d['results']['global_raw']
        per_k={k:float(pd.read_csv(OURS_RUN/f'{ds}_k{k}/summary.csv').query("arm=='s1v0' and split=='full_test'").macro_f1.iloc[0]) for k in sweep['ks']}
        bank=sweep['datasets'][ds]['banks'][str(OURS_K)]
        ours['datasets'][ds]=dict(macro_f1=macro,accuracy=acc,classes=classes,rows=int(row['rows']),proposed=int(row['proposed']),accepted=int(row['accepted']),changed=int(row['changed']),helpful=int(row['helpful']),harmful=int(row['harmful']),
            context_rows=int(bank['total_context_rows']),route_rows=int(len(np.load(base/'cache/route_ids.npy'))),cal_rows=int(len(np.load(base/'cache/cal_ids.npy'))),
            global_fit_seconds=g['fit_seconds'],preparation_seconds=preparation,fit_seconds=g['fit_seconds']+preparation,
            global_predict_seconds=g['predict_seconds'],expert_predict_seconds=expert_predict,predict_seconds=g['predict_seconds']+expert_predict,
            seconds_per_1000_rows=(g['predict_seconds']+expert_predict)/int(row['rows'])*1000,
            peak_gpu_sampled_gib=cost['peak_gpu_sampled_gib'],peak_rss_gib=cost['peak_rss_gib'],wall_seconds=cost['wall_seconds_this_attempt'],
            per_k_macro_f1=per_k,thresholds=read_record(base/'policy/selected_thresholds.json'))
    data['ours']=ours
    return data


def table(headers,rows,kind,classes='',row_classes=None):
    row_classes=row_classes or [None]*len(rows)
    return f'<div class="wrap"><table class="{classes}" data-kind="{kind}"><thead><tr>'+''.join(f'<th scope="col">{c}</th>' for c in headers)+'</tr></thead><tbody>'+''.join(('<tr class="'+rc+'">' if rc else '<tr>')+''.join((c if str(c).startswith('<td') else f'<td>{c}</td>') for c in row)+'</tr>' for row,rc in zip(rows,row_classes))+'</tbody></table></div>'


def section(data):
    s=['<!-- EXP61_START --><section id="exp61-results" aria-labelledby="exp61-title">',
       '<h2 id="exp61-title">모순 제거 데이터의 SOTA 비교 <span class="pill run">2026-09-30 갱신 · seed 43</span></h2>',
       '<p><strong>같은 10만 학습 행 기준, CIC2018은 DistPFN 0.7839, ToN은 XGBoost 0.6983이 가장 높다.</strong> 단일 방법이 두 데이터의 모든 클래스를 개선하지는 않는다. 클래스별 오탐과 미탐을 함께 보고 expert와 S/V의 개선 대상을 정한다.</p>',
       '<p class="note">모순 feature 그룹을 제거한 고정 split · CIC2018 test 3,042,473행 / ToN test 2,271,723행 · 전체 test 평가, 평가 표본 축소 없음. 정제 전 seed 42 결과는 아래 과거 기록에 별도 보존했다.</p>',
       '<h3>전체 성능 · Macro-F1과 Accuracy</h3>']
    o=data['ours'];oc,ot=o['datasets']['cic2018'],o['datasets']['toniot']
    s.insert(3,f'<p><strong>제안 모델(seed 43 · K={o["K"]} · S+V)은 ToN {ot["macro_f1"]:.4f}, CIC2018 {oc["macro_f1"]:.4f}다.</strong> ToN은 비교군 전체보다 높지만 CIC2018은 호출 0건으로 Global과 같다. 제안 모델은 expert·route 풀의 추가 학습 데이터를 쓰므로 공통 100k 비교군과 같은 예산이 아니다. 클래스별 표의 모든 행은 test 샘플 수 내림차순이다.</p>')
    rows=[];row_classes=[]
    for m in METHODS:
        a,b=(data['datasets'][ds]['results'][m] for ds in TITLES)
        rows.append([NAMES[m],f'{a["macro_f1"]:.4f}',f'{a["accuracy"]*100:.2f}%',f'{b["macro_f1"]:.4f}',f'{b["accuracy"]*100:.2f}%', '공통 100,000행' if m!='xgb_full' else '8,666,430 / 4,669,119행']);row_classes.append(None)
    rows.append([NAMES[OURS],f'{oc["macro_f1"]:.4f}',f'{oc["accuracy"]*100:.2f}%',f'{ot["macro_f1"]:.4f}',f'{ot["accuracy"]*100:.2f}%',f'100,000 + expert context {oc["context_rows"]:,} / {ot["context_rows"]:,}행<br><small>S/V route 풀 {oc["route_rows"]:,} / {ot["route_rows"]:,}행</small>']);row_classes.append('ours')
    for j in range(1,5):   # bold rule: only the highest value in each numeric column
        best=max(float(r[j].rstrip('%')) for r in rows)
        for r in rows:
            if float(r[j].rstrip('%'))==best:r[j]=f'<strong>{r[j]}</strong>'
    s.append(table(['방법','CIC · F1','CIC · Acc.','ToN · F1','ToN · Acc.','학습 풀 · CIC / ToN'],rows,'exp61-summary',row_classes=row_classes))
    ks=' / '.join(f'K={k} {v:.4f}' for k,v in ot['per_k_macro_f1'].items())
    s.append(f'<p class="note"><strong>제안 모델의 K.</strong> EXP63 K=2·4·6·8 sweep({ks}) 중 ToN test Macro-F1이 가장 높은 K={o["K"]}를 사용자 요청으로 표기했다. CIC2018은 모든 K에서 호출 0건이라 Global과 같은 {oc["macro_f1"]:.4f}다. test 기준 선택이고 단일 seed·이미 관찰한 holdout이므로 최종 K 선택이나 SOTA 우위 주장으로 읽지 않는다. 호출·채택은 저장 예측에 정책을 적용한 논리적 건수(ToN 호출 {ot["proposed"]:,} · 채택 {ot["accepted"]:,} · 교정 {ot["helpful"]:,} / 훼손 {ot["harmful"]:,}; CIC 호출 0). 상세는 Expert 역량·S/V 검증 탭 3장.</p>')
    s.append('<p class="note">앞의 5개 방법은 EXP59 Global C0와 정확히 같은 100,000개 학습 ID를 사용했다. 전체 train XGB는 추가 학습량을 허용한 참고 기준이다. Global은 기본 TabPFN이며 expert·S/V를 포함한 제안 모델 전체의 결과가 아니다. LoCalPFN은 validation을 이용한 미세조정·checkpoint 선택을 포함하고, backbone은 BoostPFN·LoCalPFN v1 / Global·DistPFN v3이므로 학습 절차까지 동일한 비교는 아니다.</p>')
    s.append('<div class="exp61-controls"><label for="exp61-metric">클래스별 비교 지표</label> <select id="exp61-metric"><option value="f1">F1</option><option value="precision">Precision</option><option value="recall">Recall</option></select><span class="note">소수점 4자리 · 굵은 값: 행에서 가장 높은 값 · 음영: 제안 모델 (expert·route 추가 학습 데이터 사용)</span></div>')
    for ds,d in data['datasets'].items():
        s.append(f'<h3>{d["title"]} · 클래스별 <span class="exp61-metric-name">F1</span> · test {d["test_rows"]:,}행</h3>')
        rows=[]
        for i,c in enumerate(d['class_order']):
            cells=[LABELS[c],f'{d["results"]["global_raw"]["classes"][i]["support"]:,}']
            vals=[d['results'][m]['classes'][i] for m in METHODS]+[data['ours']['datasets'][ds]['classes'][i]]
            assert all(v['name']==c for v in vals)
            best=max(round(v['f1'],4) for v in vals)   # bold rule: only the highest value in the row
            for m,v in zip(METHODS+[OURS],vals):
                attrs=' '.join(f'data-{k}="{v[k]}"' for k in ['f1','precision','recall'])
                css=(' b' if round(v['f1'],4)==best else '')+(' ours' if m==OURS else '')
                span=f'<span class="exp61-value{css}" data-method="{m}" {attrs}>{v["f1"]:.4f}</span>'
                cells.append(f'<td class="ours">{span}</td>' if m==OURS else span)
            rows.append(cells)
        s.append(table(['클래스','Test 행','Global v3','XGB 100k','BoostPFN','LoCalPFN FT','DistPFN v3','XGB 전체',f'제안 K={OURS_K}'],rows,f'exp61-{ds}-classes','exp61-matrix'))
        s.append(f'<details class="exp61-detail"><summary>{d["title"]} 상세 수치 · Precision / Recall / F1 / TP / FP / FN</summary><label for="exp61-{ds}-method">방법 선택</label> <select id="exp61-{ds}-method" class="exp61-method" data-dataset="{ds}">'+''.join(f'<option value="{m}">{NAMES[m]}</option>' for m in METHODS+[OURS])+'</select>')
        for m in METHODS+[OURS]:
            source=d['results'][m]['classes'] if m!=OURS else data['ours']['datasets'][ds]['classes']
            rows=[[LABELS[c['name']],f'{c["support"]:,}',*[f'{c[k]:.4f}' for k in ['precision','recall','f1']],*[f'{c[k]:,}' for k in ['TP','FP','FN']]] for c in source]
            hidden='' if m=='global_raw' else ' hidden'
            s.append(f'<div class="exp61-method-panel" data-dataset="{ds}" data-method="{m}"{hidden}>'+table(['클래스','Test 행','Precision','Recall','F1','TP','FP','FN'],rows,f'exp61-{ds}-{m}-counts')+'</div>')
        s.append('<p class="note">TP: 해당 클래스 정답 예측 · FP: 다른 클래스를 해당 클래스로 예측 · FN: 해당 클래스를 다른 클래스로 예측. 비율 계산의 분모가 0인 경우 0으로 표기한다.</p></details>')
        if ds=='cic2018':
            s.append('<p><strong>CIC2018 해석.</strong> DistPFN은 Global 대비 Macro-F1 +0.0015이며, Infiltration F1은 0.2879 → 0.3055로 개선됐다. 하지만 Web attack은 0.2102 → 0.2024로 하락했다. Web에서 Global은 127건 중 118건을 찾지만 FP 878건으로 Precision이 0.1185이고, XGB 100k는 TP 88건·FP 315건으로 F1 0.3321이다. <strong>다음 개선은 Web 탐지 수를 보존하면서 오탐을 줄이는 expert 선택·채택 기준 검증</strong>이다. BoostPFN의 Web TP는 0건이며, LoCalPFN은 Infiltration FP가 33,693건으로 많다.</p>')
        else:
            s.append('<p><strong>ToN 해석.</strong> XGB 100k는 Global 대비 Macro-F1 +0.0187이다. 전체 train XGB는 0.7162까지 올라가지만 학습량이 다르다. DistPFN은 Global의 MitM FP를 13,580 → 38,223건으로 늘려 F1이 0.1322 → 0.0512로 하락했다. 반면 BoostPFN은 Scanning F1 0.3281로 공통 100k 방법 중 가장 높지만 전체 Macro-F1은 0.5866이다. <strong>다음 비교는 클래스별 이득을 유지하면서 다른 클래스의 훼손을 막는 S/V 검증</strong>이다. 한 방법을 전역 적용하는 것만으로 모든 클래스의 문제를 해결하지는 못했다.</p>')
    s.append('<h3>측정 비용 · RTX 4090 24GB, 최대 2개 작업 병렬</h3><p class="note">아래 시간은 자원 경합을 포함한 실행 경과 시간이며 단독 실행 지연시간이나 모델 간 고유 속도 배수로 해석하지 않는다. LoCalPFN 학습 시간은 validation을 포함하며 이를 다시 더하지 않는다. DistPFN은 Global fit·추론을 공유하고, 보정 자체의 추가 시간은 데이터별 약 0.19초다.</p>')
    for ds,d in data['datasets'].items():
        rows=[]
        for m in METHODS:
            r=d['results'][m]
            rows.append([NAMES[m],f'{r["fit_seconds"]:,.2f}',f'{r["validation_seconds_included_in_fit"]:,.2f}',f'{r["predict_seconds"]:,.2f}',f'{r["seconds_per_1000_rows"]:.4f}',f'{r["peak_gpu_sampled_gib"]:.2f}',f'{r["peak_model_rss_sampled_gib"]:.2f}'])
        oo=data['ours']['datasets'][ds]
        rows.append([NAMES[OURS]+'*',f'{oo["fit_seconds"]:,.2f}','—',f'{oo["predict_seconds"]:,.2f}',f'{oo["seconds_per_1000_rows"]:.4f}',f'{oo["peak_gpu_sampled_gib"]:.2f}',f'{oo["peak_rss_gib"]:.2f}†'])
        s.append(f'<h3>{d["title"]} · 학습 및 전체 test 추론</h3>'+table(['방법','학습 (초)','학습 중 검증 (초)','전체 추론 (초)','1,000행당 (초)','GPU 최대 (GiB)','모델 RAM 최대 (GiB)'],rows,f'exp61-{ds}-cost',row_classes=[None]*len(METHODS)+['ours']))
        s.append(f'<p class="note">* 제안 모델 K={OURS_K}: 학습 = Global fit(EXP61 {oo["global_fit_seconds"]:,.1f}초) + EXP63 준비 단계(residual 군집·block 선택·expert {OURS_K}개 fit·route/validation 예측·S/V 학습·임계값 선택 {oo["preparation_seconds"]:,.1f}초). 전체 추론 = Global 추론(EXP61 {oo["global_predict_seconds"]:,.1f}초) + 모든 expert의 전체 test 예측과 정책 적용(EXP63 {oo["expert_predict_seconds"]:,.1f}초). 준비 단계에서 모든 expert를 전체 test에 적용한 측정값이며, 희소 호출의 온라인 추론 비용이 아니다. 두 run의 시간을 합산한 참고치다. † RSS는 입력 로딩을 포함한 프로세스 최대값이라 다른 행의 모델 RAM과 기준이 다르다.</p>')
    s.append('<p class="note">GPU/RAM은 1초 간격으로 측정한 프로세스별 최대 사용량이다. 모델 RAM은 전처리·모델 로딩·학습·검증·추론 단계 기준이며 원본 PKL 로딩을 제외한다. 데이터 로딩까지 포함한 RSS 최대값은 각 작업 약 25.0–26.6GiB이다. LoCalPFN의 전체 test 추론에는 kNN 검색과 index 구성이 포함된다.</p>')
    s.append(f'<details class="exp61-detail"><summary>재현 조건과 결과 파일</summary><ul><li>제안 모델: {data["ours"]["run"]} · K={OURS_K} · {data["ours"]["policy"]} · {data["ours"]["k_choice"]}</li>'+'<li>seed 43 · Global v3 4 estimators · XGB 300 trees / depth 8</li><li>BoostPFN: v1 · 50 rounds × 500 context · CE / exphadamard</li><li>LoCalPFN: v1 · context 1,000 · FT 21 epochs × 30 steps · validation AUC 선택</li><li>DistPFN: Global v3 확률의 prior 보정 · test 정답 미사용 · DistPFN-T 제외</li><li>LoCalPFN validation: CIC2018 30,194행 / ToN 41,744행</li><li>현재 test는 이미 관찰한 development holdout이며 단일 seed 결과다. 학습 예산 통제는 Global 및 기준선 사이에 적용되며 expert·route 학습까지 같은 예산이라는 의미는 아니다.</li></ul><p><code>tabpfn/results/v4/exp61/20260929_exp61_sota_local_s43/summary.csv</code><br><code>tabpfn/results/v4/exp61/20260929_exp61_sota_local_s43/class_metrics.csv</code></p></details>')
    s.append('<script id="exp61-data" type="application/json">'+json.dumps(data,ensure_ascii=False).replace('</',r'<\/')+'</script>')
    s.append('''<script>
(()=>{const names={f1:'F1',precision:'Precision',recall:'Recall'};
document.getElementById('exp61-metric').addEventListener('change',e=>{const metric=e.target.value;
document.querySelectorAll('.exp61-metric-name').forEach(el=>el.textContent=names[metric]);
document.querySelectorAll('.exp61-matrix tbody tr').forEach(row=>{const cells=[...row.querySelectorAll('.exp61-value')];const shown=c=>Number(Number(c.dataset[metric]).toFixed(4));const best=Math.max(...cells.map(shown));
cells.forEach(c=>{c.textContent=Number(c.dataset[metric]).toFixed(4);c.classList.toggle('b',shown(c)===best);});});});
document.querySelectorAll('.exp61-method').forEach(select=>select.addEventListener('change',()=>document.querySelectorAll('.exp61-method-panel').forEach(panel=>{if(panel.dataset.dataset===select.dataset.dataset)panel.hidden=panel.dataset.method!==select.value;})));
})();</script></section><!-- EXP61_END -->''')
    return '\n'.join(s)


def main():
    data=collect();collect_ours(data);t=REPORT.read_text()
    t=re.sub(r'<!-- EXP61_START -->.*?<!-- EXP61_END -->\s*','',t,flags=re.S)
    if '<!-- EXP61_ARCHIVE_START -->' not in t:
        start=t.index('<div class="verdict">');end=t.index('</main>',start)
        t=t[:start]+'<!-- EXP61_ARCHIVE_START --><details class="fold" id="historical-sota"><summary><span class="chev">›</span><span class="sum-title">과거 기록 · 2026-09-08 정제 전 seed 42 비교와 논문 재현</span><span class="sum-hint">현재 정제 split과 test 행 수가 다름 · 현재 비교 수치는 위 최신 결과 사용</span></summary><div class="fold-body">'+t[start:end]+'</div></details><!-- EXP61_ARCHIVE_END -->\n'+t[end:]
    insertion=t.index('<!-- EXP61_ARCHIVE_START -->')
    t=t[:insertion]+section(data)+'\n\n'+t[insertion:]
    t=re.sub(r'<p class="lede">.*?</p>','<p class="lede">공개 PFN 확장 방법의 소개와 IDS 비교 결과. 논문 소개 다음에 <strong>모순 제거 CIC2018·ToN의 최신 SOTA 성능과 비용(2026-09-30, seed 43)</strong>을 제시한다. 정제 전 seed 42 비교와 논문 벤치마크 재현은 하단 과거 기록에서 확인할 수 있다.</p>',t,count=1,flags=re.S)
    if '/* EXP61_STYLE */' not in t:
        t=t.replace('</style>','''/* EXP61_STYLE */
#exp61-results{border-top:2px solid var(--acc);margin-top:30px;padding-top:4px}#exp61-results p{max-width:none}.exp61-controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;background:var(--bg-2);padding:14px;margin-top:24px}#exp61-results select{padding:7px 10px;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--ink);font:inherit}.exp61-detail{border:1px solid var(--line);border-radius:6px;padding:12px 14px;margin-top:12px}.exp61-detail>summary{cursor:pointer;font-weight:600}.exp61-detail[open]>summary{margin-bottom:12px}.exp61-method-panel{margin-top:12px}.exp61-matrix th:last-child,.exp61-matrix td:last-child{border-left:2px solid var(--line);background:var(--bg-2)}#exp61-results table{font-size:13px}#exp61-results th{font-size:11px}#exp61-results details p{overflow-wrap:anywhere}@media(max-width:600px){main{padding:24px 16px 50px}h1{font-size:27px}#exp61-results h2{font-size:21px}}
</style>''',1)
    t=re.sub(r'/\* EXP61_OURS_STYLE \*/\n[^\n]*\n','',t)
    t=t.replace('</style>','/* EXP61_OURS_STYLE */\n#exp61-results td.ours,#exp61-results tr.ours td{background:var(--acc-soft)}#exp61-results tr.ours td:first-child,#exp61-results .exp61-value.ours{color:var(--acc)}\n</style>',1)
    REPORT.write_text(apply_rules(t))
    index=INDEX.read_text()
    for kind in ['FLOW','FINDING','NEXT']:
        index=re.sub(rf'<!-- EXP60_{kind}_START -->.*?<!-- EXP60_{kind}_END -->','',index,flags=re.S)
    match=re.search(r'const REPORTS = (.*?);\n',index);reports=json.loads(match[1]);reports=[r for r in reports if r['id']!='counterexample_context_0929']
    for r in reports:
        if r['id']=='sota_pfn_comparison':
            r['verdict']='09-30 정제 후 CIC·ToN · SOTA 클래스별 성능·비용';r['verdictClass']='warn'
    index=index[:match.start(1)]+json.dumps(reports,ensure_ascii=False)+index[match.end(1):]
    index=index.replace('2026-08-18 &ndash; 2026-09-29','2026-08-18 &ndash; 2026-09-30')
    count=sum(r.get('group')=='report' for r in reports)
    index=re.sub(r'8&ndash;9월의 보고서 (?:\d+|아홉)편?을 최신순으로 묶었습니다\.',f'8&ndash;9월의 보고서 {count}편을 최신순으로 묶었습니다.',index)
    if '<!-- EXP61_OVERVIEW -->' not in index:
        target='<div class="status-grid">'
        assert target in index
        index=index.replace(target,target+'\n<!-- EXP61_OVERVIEW --><div class="status-cell"><p class="label">09-30 · 정제 후 SOTA 비교 완료</p><p class="value">Seed 43 · 같은 C0 100k 기준 Macro-F1: CIC2018 DistPFN <strong>0.7839</strong>, ToN XGB <strong>0.6983</strong>. 클래스별 Precision·Recall·F1과 측정 비용은 <button class="jump" data-jump="sota_pfn_comparison">0908 SOTA 탭</button> 상단에 반영.</p></div>',1)
    INDEX.write_text(index)
    removed=REPORT.with_name('counterexample_context_0929.html')
    if removed.exists():
        archive=ROOT/'docs/research/20260929/archive';archive.mkdir(exist_ok=True)
        removed.replace(archive/removed.name)
    print(f'Updated {REPORT}; removed withdrawn report from active HTML; {len(data["datasets"])} datasets validated')


if __name__=='__main__':main()
