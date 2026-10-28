#!/usr/bin/env python3
"""Validate available EXP84 outputs and maintain the Korean source-only lablog."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap,repo_root
bootstrap(_root)
ROOT=repo_root(__file__)
import argparse
import csv
import json
import hashlib
import numpy as np
from exp61_utils import write,sha
from noft_dist_budget import load_source
from replay_protocol import confusion_metrics
from lablog import write_record,build_report,check

METHODS=['xgb','pfn_accumulate','pfn_fifo','pfn_random','pfn_similar']
LABELS=['XGB 누적 재학습','PFN 누적','PFN FIFO','PFN 무작위','PFN 유사도']


def read(p):return json.loads(Path(p).read_text())


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true',help='Write validation/CSV only inside this run; preserve the original report')
    args=parser.parse_args()
    run=args.run.resolve();cfg=read(run/'config.json')
    methods=cfg.get('methods',METHODS)
    expected=len(cfg['datasets'])*len(cfg['new_budgets'])*len(methods)
    data=read(run/'results.json') if (run/'results.json').exists() else dict(state='preparing',jobs=[],expected_jobs=60)
    rows=[];classes=[];stage_rows=[];verified=[]
    for ds in cfg['datasets']:
        meta,arr,_=load_source(ds);names=meta['class_names']
        for n in cfg['new_budgets']:
            for method in methods:
                job=run/f'new_{n}'/'jobs'/f'{ds}_{method}'
                if not (job/'COMPLETE.json').exists():continue
                d=read(job/'results.json');plans=read(run/f'new_{n}'/'data'/ds/'plans.json')
                assert d['seed']==cfg['seed']
                pred=np.load(job/'predictions.npy',mmap_mode='r');assert len(pred)==len(arr['test_y'])
                assert np.all((pred>=0)&(pred<len(names)))
                actual=confusion_metrics(arr['test_y'],np.asarray(pred),names)
                assert actual==d['metrics']
                assert sha(job/'predictions.npy')==read(job/'COMPLETE.json')['predictions_sha256']
                assert abs(sum(x['update_seconds'] for x in d['fits'][1:])-d['update_seconds'])<1e-9
                panel=np.load(run/'data'/ds/'panel_ids.npy');py=arr['test_y'][panel]
                previous=None;old_classes=None;evicted=set();old=None
                for s,(fit,plan,pm) in enumerate(zip(d['fits'],plans,d['stage_panel_metrics'])):
                    ids=np.asarray(fit['context_indices']);incoming=set(plan['incoming']);removed=set(fit['removed_indices'])
                    assert incoming<=set(ids) and not set(ids)&evicted
                    assert fit['admitted_rows']==len(incoming) and fit['rejected_rows']==0
                    if method in ['xgb','pfn_accumulate']:assert ids.tolist()==plan['released']
                    else:
                        assert len(ids)==len(set(ids))==1000
                        if s:assert len(removed)==len(incoming) and set(ids)==(set(old)-removed)|incoming
                        assert set(arr['train_y'][ids])==set(arr['train_y'][plan['released']])
                    evicted|=removed;old=ids
                    pp=np.load(job/f'stage_{s}'/'panel_predictions.npy')
                    known=np.asarray(fit['classes']);mask=np.isin(py,known)
                    assert confusion_metrics(py[mask],pp[mask],names,known)==pm['known']
                    if previous is not None:
                        mask=np.isin(py,old_classes)
                        before=confusion_metrics(py[mask],previous[mask],names,old_classes)
                        after=confusion_metrics(py[mask],pp[mask],names,old_classes)
                        assert before==pm['old_before'] and after==pm['old_after']
                        assert abs(after['macro_f1']-before['macro_f1']-pm['old_macro_f1_change'])<1e-12
                        stage_rows.append(dict(dataset=ds,n=n,method=method,stage=s+1,old_rows=int(mask.sum()),
                            old_before_pct=100*before['macro_f1'],old_after_pct=100*after['macro_f1'],old_change_pp=100*pm['old_macro_f1_change']))
                    previous=pp;old_classes=known
                row=dict(dataset=ds,n=n,method=method,macro_f1_pct=actual['macro_f1']*100,
                         new_attack_f1_pct=d['new_attack_mean_f1']*100,initial_attack_f1_pct=d['initial_attack_mean_f1']*100,
                         benign_fpr_pct=actual['benign_false_positive_rate']*100,update_seconds=d['update_seconds'],
                         initial_retention_change_pp=d['initial_panel_final_change']*100,
                         worst_update_old_change_pp=d['worst_update_old_change']*100,context_rows=d['context_rows'],
                         labels=d['label_budget'],initial_seconds=d['initial_seconds'],inference_seconds=d['serving_seconds'])
                rows.append(row)
                for c in actual['classes']:
                    classes.append(dict(dataset=ds,n=n,method=method,**c))
                verified.append(dict(dataset=ds,n=n,method=method,full_test_rows=len(pred),validated=True))
    if data['state']=='complete':assert len(rows)==expected
    dest=run if args.validate_only else ROOT/'docs/lablog/data';dest.mkdir(exist_ok=True)
    for name,values in [('exp84_budget_values.csv',rows),('exp84_per_class.csv',classes),('exp84_retention.csv',stage_rows)]:
        if values:
            for target in dict.fromkeys([dest/name,run/name]):
                with target.open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    validation=dict(passed=True,scope='Completed jobs only',state=data['state'],completed=len(rows),expected=expected,jobs=verified)
    write(run/'validation.json',validation)
    if args.validate_only:
        print(json.dumps(dict(state=data['state'],validated_jobs=len(rows),seed=cfg['seed'])),flush=True)
        return
    figs=[]
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt,font_manager
        import pandas as pd
        font='/home/user/.local/share/fonts/Pretendard/Pretendard-Regular.otf';font_manager.fontManager.addfont(font)
        plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
        plt.rcParams['axes.unicode_minus']=False
        frame=pd.DataFrame(rows)
        definitions=[('quality','macro_f1_pct','최종 전체 test Macro-F1 (%)'),
                     ('new_attacks','new_attack_f1_pct','신규 공격군 평균 F1 (%)'),
                     ('retention','initial_retention_change_pp','초기 클래스 고정 패널: 최종−초기 (pp)'),
                     ('cost','update_seconds','초기 이후 갱신 시간 합계 (초)')]
        colors=['#617080','#00306C','#007BC6','#24A4B8','#8C5AA8']
        for suffix,metric,title in definitions:
            fig,axs=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
            for ax,ds in zip(axs,cfg['datasets']):
                for method,label,color in zip(METHODS,LABELS,colors):
                    part=frame[(frame.dataset==ds)&(frame.method==method)].sort_values('n')
                    if len(part):ax.plot(part.n,part[metric],marker='o',label=label,color=color,linewidth=1.5)
                ax.set_title(ds);ax.set_xlabel('신규 공격 클래스당 라벨 수');ax.set_ylabel(title)
                ax.set_xticks(cfg['new_budgets']);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
                if suffix=='retention':ax.axhline(0,color='#333333',linewidth=.7)
                if suffix=='cost':ax.set_ylim(bottom=0)
            axs[0].legend(fontsize=8,frameon=False)
            path=ROOT/f'docs/lablog/figures/exp84_{suffix}.pdf';fig.savefig(path);plt.close(fig);figs.append((title,path))
    state={'complete':'완료','running':'실행 중','preparing':'준비 중','failed':'실패·부분 완료'}.get(data['state'],data['state'])
    text=['# EXP84 · 소수 신규 공격 표본과 1,000개 context 버퍼',
          '## 목적과 현재 상태',
          f'연구 버전 v7, 실행 ID `{run.name}`, 실행 날짜 {cfg["report_date"]}, seed 43 한 번. 현재 상태는 **{state}**이며 60조건 중 {len(rows)}조건의 전체 test 예측을 검증했다. '
          '사용자는 2026-10-20에 신규 공격당 5·10·20·30·50·100개 실행을 요청했다. 슬라이드와 논문 결과는 자동 편집하지 않는다.',
          'XGB 누적 재학습과 FT 없는 PFN의 누적·FIFO·무작위·유사도 context 갱신을 비교한다. 공급한 신규 사례를 전부 투입하며 편입 상한이나 추가 클래스별 할당량을 두지 않는다.',
          '## 공통 설정',
          '두 벤치마크는 독립 평가하며 같은 train에서 초기 정상 750개와 공격 두 클래스 각각 125개를 선택한다. CIC 초기 공격은 Brute force·DoS, ToN 초기 공격은 DoS·Scanning이다. '
          'CIC은 DDoS→Web attacks→Infiltration→Bot, ToN은 DDoS/Injection→Password→XSS→Backdoor/Ransomware→MitM 순으로 신규 클래스당 n개를 제공한다. '
          'CIC 최종 누적은 1,000+4n, ToN은 1,000+7n이다. n=100에서 각각 1,400/1,700개이며 버퍼는 항상 1,000개다. '
          'ToN의 두 클래스 동시 도입에서는 최대 200개를 교체한다. EXP83의 기존 클래스 최소 1개 규칙만 유지하고 모든 신규 행을 수용한다.',
          '가중치는 동일한 TabPFN v3 checkpoint로 고정하며 초기·후속 IDS FT 없음, estimator 4개, fit_with_cache, CPU cache, subsampling 없음이다. XGB는 300 trees, depth 8, learning rate 0.05, CUDA histogram, threads 8이다. '
          '전체 예측과 갱신 시간을 새로 측정하며 과거 결과·시간을 재사용하지 않는다. 최초 CUDA 연산 warm-up은 별도 기록한다.',
          '마지막 모델로 CIC 3,042,473개·ToN 2,271,723개 전체 test를 평가한다. Query는 동일 seed-43 혼합 순서, batch 8,192개다. '
          '신규 공격군은 초기 이후 도입된 공격이며 평균 F1은 전체 test 혼동행렬의 클래스별 F1을 평균한다. 클래스별 precision/recall/F1은 다른 클래스의 오탐을 포함한다.',
          '기존 성능 유지는 클래스당 최대 2,000개 고정 test 패널의 같은 행·같은 query 구성에서 갱신 전후 비교한다. 미래 클래스 query 특징이 포함될 수 있는 오프라인 진단이며 정답은 fitting이나 context 선택에 전달하지 않는다. '
          '초기 클래스 최종−초기 변화와 각 갱신의 직전까지 알려진 클래스 변화는 다른 지표다. 기존 행만의 F1과 현재까지 도입된 전체 행의 클래스별 F1을 함께 보존한다.',
          '주 비용은 초기 단계를 제외한 context 구성/삭제·모델 구성·fit/cache 준비 시간의 합계다. 초기 준비·모든 추론·모델 저장·검증은 분리한다. '
          'GPU 작업은 순차 실행한다. 원본 행 ID는 train/test에서 다르지만 같은 특징 벡터가 남을 수 있어 중복 특징 제외 진단을 함께 저장한다.',
          '## 도표와 검증된 관찰']
    if rows:
        text.append('아래는 완료된 조건의 실제 측정값이다. 미완료 조건을 보간하거나 최종 결과로 추정하지 않는다.')
        for title,path in figs:text.append(f'![{title}; 완료한 조건만 표시]({path})')
        for ds in cfg['datasets']:
            text.append('### '+ds)
            selected=sorted([r for r in rows if r['dataset']==ds],key=lambda r:(r['n'],METHODS.index(r['method'])))
            text.append(table(['n','방법','전체 F1 (%)','신규 F1 (%)','기존 변화 (pp)','갱신 (초)'],[
                [r['n'],LABELS[METHODS.index(r['method'])],f'{r["macro_f1_pct"]:.2f}',f'{r["new_attack_f1_pct"]:.2f}',
                 f'{r["initial_retention_change_pp"]:+.2f}',f'{r["update_seconds"]:.3f}'] for r in selected]))
            for n in [5,100]:
                pair={r['method']:r for r in selected if r['n']==n}
                if 'xgb' in pair and 'pfn_accumulate' in pair:
                    a,b=pair['xgb'],pair['pfn_accumulate']
                    text.append(f'n={n}에서 XGB/PFN 누적의 전체 F1은 {a["macro_f1_pct"]:.2f}/{b["macro_f1_pct"]:.2f}%, '
                        f'신규 공격군 평균 F1은 {a["new_attack_f1_pct"]:.2f}/{b["new_attack_f1_pct"]:.2f}%, '
                        f'후속 갱신은 {a["update_seconds"]:.3f}/{b["update_seconds"]:.3f}초다. 이는 해당 표본·seed의 관찰이며 최소 필요 표본 수나 동등성을 확정하지 않는다.')
        text.append('## 클래스별 결과')
        text.append('아래는 n=5와 n=100의 클래스별 F1이다. 클래스는 각 데이터셋 전체 test 표본 수 내림차순이다. 모든 n의 precision·recall·F1은 `docs/lablog/data/exp84_per_class.csv`에 있다.')
        for ds in cfg['datasets']:
            for n in [5,100]:
                vals=[c for c in classes if c['dataset']==ds and c['n']==n]
                if not vals:continue
                names=sorted(set(c['name'] for c in vals),key=lambda name:-next(c['support'] for c in vals if c['name']==name))
                values=[]
                for name in names:
                    row=[name,next(c['support'] for c in vals if c['name']==name)]
                    for method in METHODS:
                        c=next((c for c in vals if c['name']==name and c['method']==method),None)
                        row.append(f'{100*c["f1"]:.2f}' if c else '미완료')
                    values.append(row)
                text.append('### '+ds+f' · n={n}');text.append(table(['클래스','test 수','XGB','PFN 누적','FIFO','무작위','유사도'],values))
    else:text.append('모델 결과가 아직 없으므로 성능·비용을 기입하지 않았다. 작업 진행은 실행 폴더의 progress.json과 stdout.log에서 확인한다.')
    text+=['## 해석 범위와 다음 판단',
           '신규 표본 수를 늘리는 효과와 과거 사례 삭제의 영향이 동시에 존재한다. 같은 버퍼 크기에서 정책이 다르면 실제 보유 클래스 구성도 다르므로 단일 원인으로 단정하지 않는다. '
           '신규 공격 성능, 기존 클래스 변화, 정상 오탐, 갱신 비용을 함께 보고 판단한다. 버퍼를 유지해도 전체 context를 다시 처리하므로 비용 절감은 실측 대상이다. '
           '단일 seed이며 이전에 살펴본 holdout을 사용한 탐색 결과다. 점수가 높은 정책을 test로 선택해 확정하지 않고, 후속 개선은 학습/검증 자료에서 설계한다. '
           '필요 표본 수를 확정하려면 허용할 기존 성능 저하와 신규 탐지 품질 기준이 먼저 필요하다.',
           '## 실행 근거와 인계',f'실행 경로: `{run.relative_to(ROOT)}`. 설정·소스 snapshot·단계별 context ID·모델·패널 예측·최종 전체 test 예측·비용·검증을 이 폴더에 보관한다. '
           '설정은 `configs/experiments/exp84_protocol.json`, 비교 수치는 `docs/lablog/data/exp84_budget_values.csv`, 유지 진단은 `docs/lablog/data/exp84_retention.csv`에 연결한다.']
    if data['state']=='failed':
        text.append('실패 원인: '+data.get('error','FAILURE.json 참조')+'. 완료 표시는 검증된 조건에만 적용한다. 실패 단계의 로그와 자원 상태를 확인하고, 이미 완료된 결과의 소스 일치를 점검한 뒤 재개한다.')
    elif data['state']!='complete':
        text.append('남은 작업은 미완료 모델 조건 실행 및 최종 결과 검증·보고다. Worker가 정상 완료 또는 실패 시 이 보고서와 한 줄 로그를 자동 갱신하고 임시 PDF 컴파일을 수행한다.')
    target=write_record(ROOT/f'docs/lablog/{cfg["report_date"]}/exp84.md','\n\n'.join(text),
         summary=f'신규 5·10·20·30·50·100개 전량 투입, 누적 XGB/PFN 및 1,000개 버퍼 3종; {state} {len(rows)}/60조건, 최종 전체 test·기존 클래스 유지·갱신 비용 비교',compile_pdf=False)
    # Persist only source and verification metadata, not the generated report PDF.
    info=build_report(target)
    vp=ROOT/'docs/lablog/_support/validation.json';v=read(vp)
    previous={x['report']:x for x in v.get('reports',[])};previous[info['report']]=info;v['reports']=[previous[k] for k in sorted(previous)];v['structure']=check();write(vp,v)
    sp=ROOT/'docs/lablog/_support/sources.json';sources=read(sp)
    entry=dict(path=str(target.relative_to(ROOT)),title='EXP84 · 소수 신규 공격 표본과 1,000개 context 버퍼',
               sha256=sha(target),pdf_persisted=False,last_compiled_pdf_sha256=info['compiled_pdf_sha256'],
               state=data['state'],run=str(run.relative_to(ROOT)))
    # Use the existing report-list schema rather than creating a second index.
    key=next(k for k,val in sources.items() if isinstance(val,list) and any(isinstance(x,dict) and str(x.get('path','')).startswith('docs/lablog/reports/') for x in val))
    sources[key]=[x for x in sources[key] if x.get('path')!=entry['path']]+[entry];write(sp,sources)
    print(json.dumps(dict(state=data['state'],validated_jobs=len(rows),report=str(target),pdf_persisted=False)),flush=True)


if __name__=='__main__':main()
