#!/usr/bin/env python3
"""Aggregate validated EXP84 repetitions and compile a Korean source-only report."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap,repo_root
bootstrap(_root)
ROOT=repo_root(__file__)

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from exp61_utils import write,sha
from small_label_updates import read,METHODS
from build_small_label_record import LABELS,table
from lablog import write_record,build_report,check

METRICS=['macro_f1_pct','new_attack_f1_pct','initial_attack_f1_pct','benign_fpr_pct',
         'initial_retention_change_pp','worst_update_old_change_pp','update_seconds',
         'initial_seconds','inference_seconds']


def aggregate(frame, keys, metrics):
    records=[]
    for group,part in frame.groupby(keys,sort=True):
        if not isinstance(group,tuple):group=(group,)
        assert part.seed.nunique()==len(part), 'Duplicate seed in condition'
        row=dict(zip(keys,group))
        row.update(seed_count=len(part),seeds=','.join(map(str,sorted(part.seed.tolist()))))
        for metric in metrics:
            row[metric+'_mean']=part[metric].mean()
            row[metric+'_sd']=part[metric].std(ddof=1)
        records.append(row)
    return pd.DataFrame(records)


def format_stat(row, key, digits=2):
    mean=row[key+'_mean'];sd=row[key+'_sd']
    return f'{mean:.{digits}f} ± {sd:.{digits}f}' if pd.notna(sd) else f'{mean:.{digits}f} (1회)'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve()
    cfg=read(run/'config.json');state=read(run/'results.json')
    rows=[];classes=[];stages=[];validated=[];raw_completed={}
    for seed in cfg['seeds']:
        seed_run=Path(state['seed_runs'][str(seed)])
        raw_completed[seed]=len(list(seed_run.glob('new_*/jobs/*/COMPLETE.json')))
        vp=seed_run/'validation.json'
        if not vp.exists():continue
        val=read(vp)
        if not (val['passed'] and val['completed']==60 and val['state']=='complete'):continue
        validated.append(seed)
        for n in cfg['new_budgets']:
            for ds in cfg['datasets']:
                for method in METHODS:
                    job=seed_run/f'new_{n}'/'jobs'/f'{ds}_{method}'
                    assert (job/'COMPLETE.json').exists()
                    d=read(job/'results.json');assert d['seed']==seed and d['test_full']
                    m=d['metrics']
                    rows.append(dict(seed=seed,dataset=ds,n=n,method=method,
                        macro_f1_pct=100*m['macro_f1'],new_attack_f1_pct=100*d['new_attack_mean_f1'],
                        initial_attack_f1_pct=100*d['initial_attack_mean_f1'],
                        benign_fpr_pct=100*m['benign_false_positive_rate'],
                        initial_retention_change_pp=100*d['initial_panel_final_change'],
                        worst_update_old_change_pp=100*d['worst_update_old_change'],
                        update_seconds=d['update_seconds'],initial_seconds=d['initial_seconds'],
                        inference_seconds=d['serving_seconds'],context_rows=d['context_rows'],
                        labels=d['label_budget'],test_rows=d['test_rows'],source=str(job/'results.json')))
                    for c in m['classes']:
                        classes.append(dict(seed=seed,dataset=ds,n=n,method=method,name=c['name'],
                            support=c['support'],precision_pct=100*c['precision'],recall_pct=100*c['recall'],
                            f1_pct=100*c['f1'],TP=c['TP'],FP=c['FP'],FN=c['FN']))
        stage=pd.read_csv(seed_run/'exp84_retention.csv');stage['seed']=seed
        stages.extend(stage.to_dict('records'))
    assert 43 in validated
    if state['state']=='complete':assert validated==cfg['seeds'] and len(rows)==180
    frame=pd.DataFrame(rows);class_frame=pd.DataFrame(classes);stage_frame=pd.DataFrame(stages)
    summary=aggregate(frame,['dataset','n','method'],METRICS)
    class_summary=aggregate(class_frame,['dataset','n','method','name','support'],
                            ['precision_pct','recall_pct','f1_pct','TP','FP','FN'])
    stage_summary=aggregate(stage_frame,['dataset','n','method','stage'],
                            ['old_before_pct','old_after_pct','old_change_pp'])
    xgb=frame[frame.method=='xgb'][['seed','dataset','n']+METRICS]
    paired=frame[frame.method!='xgb'].merge(xgb,on=['seed','dataset','n'],suffixes=('','_xgb'),validate='many_to_one')
    paired_metrics=[]
    for metric in METRICS:
        key=metric+'_minus_xgb';paired[key]=paired[metric]-paired[metric+'_xgb'];paired_metrics.append(key)
    paired['update_cost_ratio_to_xgb']=paired.update_seconds/paired.update_seconds_xgb
    paired_metrics.append('update_cost_ratio_to_xgb')
    paired_summary=aggregate(paired,['dataset','n','method'],paired_metrics)
    dest=ROOT/'docs/lablog/data';dest.mkdir(exist_ok=True)
    outputs={'values':frame,'summary':summary,'per_class':class_frame,'per_class_summary':class_summary,
             'retention':stage_frame,'retention_summary':stage_summary,
             'paired_xgb':paired[['seed','dataset','n','method']+paired_metrics],
             'paired_xgb_summary':paired_summary}
    for suffix,data in outputs.items():
        for directory in [run,dest]:data.to_csv(directory/f'exp84_multiseed_{suffix}.csv',index=False)
    figures=[]
    if len(validated)>=2:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt,font_manager
        font='/home/user/.local/share/fonts/Pretendard/Pretendard-Regular.otf'
        font_manager.fontManager.addfont(font)
        plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
        plt.rcParams['axes.unicode_minus']=False
        definitions=[('quality','macro_f1_pct','최종 전체 test Macro-F1 (%)'),
            ('new_attacks','new_attack_f1_pct','신규 공격군 평균 F1 (%)'),
            ('retention','initial_retention_change_pp','초기 클래스 고정 패널: 최종−초기 (pp)'),
            ('cost','update_seconds','초기 이후 갱신 시간 합계 (초)'),
            ('false_positive','benign_fpr_pct','정상 사례의 공격 오탐률 (%)')]
        colors=['#617080','#00306C','#007BC6','#24A4B8','#8C5AA8']
        for suffix,metric,title in definitions:
            fig,axs=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
            for ax,ds in zip(axs,cfg['datasets']):
                for method,label,color in zip(METHODS,LABELS,colors):
                    part=summary[(summary.dataset==ds)&(summary.method==method)].sort_values('n')
                    ax.errorbar(part.n,part[metric+'_mean'],yerr=part[metric+'_sd'],marker='o',
                                label=label,color=color,linewidth=1.3,capsize=2,markersize=3)
                ax.set_title(ds);ax.set_xlabel('신규 공격 클래스당 라벨 수');ax.set_ylabel(title)
                ax.set_xticks(cfg['new_budgets']);ax.grid(axis='y',alpha=.2)
                ax.spines[['top','right']].set_visible(False)
                if suffix=='retention':ax.axhline(0,color='#333333',linewidth=.7)
                else:ax.set_ylim(bottom=0)
            axs[0].legend(fontsize=8,frameon=False)
            fig.suptitle(f'{len(validated)} seeds: 평균 ± 표본 표준편차 (신뢰구간 아님)',fontsize=10)
            path=ROOT/f'docs/lablog/figures/exp84_multiseed_{suffix}.pdf'
            fig.savefig(path);plt.close(fig);figures.append((title,path))
        fig,axs=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
        for ax,(ds,name) in zip(axs,[('cic2018','web_attacks'),('toniot','ransomware')]):
            for method,label,color in zip(METHODS,LABELS,colors):
                part=class_summary[(class_summary.dataset==ds)&(class_summary.name==name)&(class_summary.method==method)].sort_values('n')
                ax.errorbar(part.n,part.f1_pct_mean,yerr=part.f1_pct_sd,marker='o',label=label,color=color,capsize=2,markersize=3)
            ax.set_title(f'{ds}: {name}');ax.set_xlabel('신규 공격 클래스당 라벨 수');ax.set_ylabel('클래스 F1 (%)')
            ax.set_xticks(cfg['new_budgets']);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
        axs[0].legend(fontsize=8,frameon=False)
        path=ROOT/'docs/lablog/figures/exp84_multiseed_tail.pdf';fig.savefig(path);plt.close(fig)
        figures.append(('seed 43에서 낮은 F1을 보였던 두 공격의 반복 결과; 평균 ± 표본 표준편차',path))
    status={'complete':'완료','running':'실행 중','preparing':'준비 중','failed':'실패·부분 완료'}.get(state['state'],state['state'])
    now=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')
    new_done=sum(raw_completed[s] for s in cfg['new_seeds'])
    text=['# EXP84 · 소수 신규 공격 표본과 1,000개 버퍼: 3 seed 반복',
          '## 목적과 현재 상태',
          f'연구 버전 v7의 EXP84를 seed 43·44·45로 반복하여 표본 선택과 모델 난수에 따른 성능 변동을 확인한다. '
          f'사용자는 2026-10-21에 3개 또는 5개 seed 반복을 요청했으며 총 3개로 실행한다. '
          f'현재 상태는 **{status}**, 기록 시각은 {now}이다. '
          f'새 실행은 {new_done}/120조건의 예측 저장이 끝났고, 전체 조건 검증을 마친 seed는 {validated}이다.',
          '기존 seed 43의 60조건은 2026-10-20 완료 결과를 그대로 참조한다. seed 44·45 각각 60조건을 새로 실행하여 최종 180조건을 구성한다. '
          '기존 seed의 결과·실행 시간·보고서·도표를 덮어쓰지 않는다. 실험 실행 완료와 논문 근거 채택은 구분하며 슬라이드와 영어 원고는 자동 수정하지 않는다.',
          '## 공통 설정과 seed 변경 범위',
          '신규 공격당 n=5·10·20·30·50·100개를 제공하고 두 벤치마크를 독립적으로 평가한다. '
          '초기 1,000개는 정상 750개와 초기 공격 두 클래스 각각 125개다. '
          'CIC 초기 공격은 Brute force·DoS이며 DDoS → Web attacks → Infiltration → Bot 순서로 도입한다. '
          'ToN 초기 공격은 Scanning·DoS이며 Injection/DDoS → Password → XSS → Ransomware/Backdoor → MitM 순서다.',
          'XGB 누적 재학습, TabPFN 누적 context, 1,000개 FIFO·무작위·유사도 버퍼를 비교한다. '
          '누적 최종 사례 수는 CIC 1,000+4n, ToN 1,000+7n이다. 버퍼는 신규 행을 모두 넣고 같은 수의 기존 행을 삭제한다. '
          '기존 클래스 최소 1개 보존 규칙만 유지하며 별도의 편입 상한이나 클래스 할당량은 두지 않는다. '
          'FIFO는 확보 순서이며 실제 네트워크에서의 노후화를 판정하는 규칙은 아니다. 유사도 삭제는 기존 context에 맞춘 표준화와 신규 사례까지의 최소 거리를 사용한다.',
          'seed에 따라 초기 사례, 신규 클래스당 후보 100개, 각 n의 부분표본, 모델 random_state와 무작위 삭제 순서가 달라진다. '
          '같은 seed에서 모든 방법은 같은 초기·신규 사례를 받으며 작은 n의 표본은 큰 n의 부분집합이다. '
          '기존 seed 43의 초기·신규 행 ID와 순서를 완전히 재현하는지 검사하여 반복의 표본 선택 알고리즘이 동일함을 확인한다. '
          '이는 여러 난수 요인이 함께 바뀌는 반복으로, 표본 선택과 모델 내부 난수의 영향을 따로 분해하는 실험은 아니다.',
          'train/test 분할과 test의 query 순서 및 고정 진단 패널은 모든 seed에서 동일하다. '
          '학습 자료만 seed별로 달라지므로 test 분할 변동까지 추정하지 않는다. '
          'TabPFN v3의 가중치는 고정하며 초기·후속 가중치 fine-tuning은 없다. '
          '4 estimators, fit_with_cache, CPU cache, context subsampling 없음이며 XGB는 CUDA histogram, 300 trees, depth 8, learning rate 0.05, threads 8이다.',
          '## 평가와 집계 정의',
          '각 seed의 최종 모델을 CIC 3,042,473개·ToN 2,271,723개 고정 전체 test에 평가한다. '
          'query 순서는 seed 43으로 고정하고 batch는 8,192개다. 신규 공격군 평균 F1은 전체 test 혼동행렬에서 초기 이후 도입한 CIC 4개·ToN 7개 클래스의 F1을 같은 가중치로 평균한 값이다. '
          '단계별 예측을 합산하거나 클래스가 등장한 이후 단계 점수를 평균하지 않는다.',
          '기존 클래스 유지는 클래스당 최대 2,000개 고정 패널에서 초기 세 클래스의 최종 F1−초기 F1으로 비교한다. '
          '같은 행과 query 구성을 유지하며 이 지표의 오탐 모집단은 초기 클래스 행으로 한정된다. '
          '전체 test의 클래스별 F1·precision·recall과 정상 오탐률을 별도로 보고 새 클래스와의 혼동도 확인한다. '
          '고정 패널은 미래 클래스 특징을 포함할 수 있는 오프라인 진단이며 query 정답은 모델 학습이나 context 선택에 사용하지 않는다.',
          '갱신 비용은 초기 단계를 제외한 사례 선택/삭제·모델 구성·fit/cache 준비 시간의 합계다. '
          '초기 준비, 모든 추론, 모델 저장과 검증은 제외하고 별도 수치로 남긴다. '
          'GPU 작업은 순차 실행하지만 seed 43과 새 seed는 서로 다른 실행 세션이므로 시간의 표준편차에는 시스템 상태 차이도 포함될 수 있다. '
          '패키지 버전과 pretrained checkpoint의 동일성을 검사한다.',
          f'평균과 표준편차는 조건별 seed 결과에서 계산하며 표본 표준편차(ddof=1)를 사용한다. '
          f'현재 집계에는 60조건 전체 검증을 통과한 {len(validated)}개 seed {validated}만 포함한다. '
          '한 seed에서는 표준편차를 0으로 채우지 않는다. XGB 대비 차이는 같은 seed끼리 뺀 뒤 평균·표준편차를 계산한다. '
          '오차막대는 신뢰구간이 아니며 3회 반복만으로 동등성이나 통계적 우위를 확정하지 않는다.',
          '## 도표와 관찰 결과']
    if len(validated)<2:
        text.append('새 seed 결과가 아직 전체 검증을 마치지 않아 반복 평균 그래프를 제시하지 않는다. 기존 seed 43의 결과는 20261020_exp84.tex에 있으며, 새 반복의 결론으로 대신하지 않는다.')
    else:
        for title,path in figures:text.append(f'![{title}]({path})')
        for ds in cfg['datasets']:
            text.append('### '+ds+' · 모든 신규 표본 수')
            part=summary[summary.dataset==ds].copy()
            part['order']=part.method.map({m:i for i,m in enumerate(METHODS)})
            part=part.sort_values(['n','order'])
            text.append(table(['n','방법','전체 F1 (%)','신규 F1 (%)','기존 변화 (pp)','갱신 (초)'],[
                [r.n,LABELS[METHODS.index(r.method)],format_stat(r,'macro_f1_pct'),
                 format_stat(r,'new_attack_f1_pct'),format_stat(r,'initial_retention_change_pp'),
                 format_stat(r,'update_seconds',3)] for _,r in part.iterrows()]))
            for method in ['pfn_accumulate','pfn_fifo','pfn_random']:
                pair=paired_summary[(paired_summary.dataset==ds)&(paired_summary.method==method)&(paired_summary.n==100)].iloc[0]
                text.append(f'n=100의 {LABELS[METHODS.index(method)]}−XGB 대응 차이는 전체 F1 '
                    f'{format_stat(pair,"macro_f1_pct_minus_xgb")} pp, 신규 공격군 평균 F1 '
                    f'{format_stat(pair,"new_attack_f1_pct_minus_xgb")} pp, 갱신 시간 '
                    f'{format_stat(pair,"update_seconds_minus_xgb",3)}초다. 차이의 부호와 변동을 함께 확인하며 단일 seed 순위만으로 결론짓지 않는다.')
        text.append('## 클래스별 성능과 오탐')
        text.append('아래 n=5·100의 클래스별 F1은 seed 평균 ± 표본 표준편차이며 각 전체 test 표본 수 내림차순으로 정렬한다. 모든 n의 precision·recall·F1·TP·FP·FN은 클래스별 CSV에 보존한다. 신규 공격군 평균이 양호해도 희소 클래스의 오탐이나 정밀도 저하가 가려질 수 있으므로 클래스별 결과를 함께 본다.')
        for ds in cfg['datasets']:
            for n in [5,100]:
                part=class_summary[(class_summary.dataset==ds)&(class_summary.n==n)]
                names=part[['name','support']].drop_duplicates().sort_values('support',ascending=False)
                values=[]
                for _,c in names.iterrows():
                    values.append([c['name'],int(c.support)]+[
                        format_stat(part[(part.name==c['name'])&(part.method==method)].iloc[0],'f1_pct') for method in METHODS])
                text.append('### '+ds+f' · n={n}')
                text.append(table(['클래스','test 수','XGB','PFN 누적','FIFO','무작위','유사도'],values))
        for ds,name in [('cic2018','web_attacks'),('toniot','ransomware')]:
            text.append('### '+ds+' · '+name+' · n=100')
            part=class_summary[(class_summary.dataset==ds)&(class_summary.name==name)&(class_summary.n==100)]
            text.append(table(['방법','precision (%)','recall (%)','F1 (%)','FP'],[
                [LABELS[METHODS.index(method)]]+[format_stat(part[part.method==method].iloc[0],metric)
                for metric in ['precision_pct','recall_pct','f1_pct','FP']] for method in METHODS]))
    text += ['## 해석 범위와 후속 개선 판단',
             '반복에서도 누적 PFN이 해당 희소 클래스에 약하다면 삭제만으로 원인을 설명할 수 없다. '
             '동일 seed에서 누적과 버퍼의 클래스별 precision·recall·정상 오탐을 함께 비교하여 삭제의 추가 영향과 PFN 공통 패턴을 구분한다. '
             '버퍼 크기 확대나 보정 방법을 다음 실험으로 선택할 때는 이 구분을 근거로 삼고, 조정 규칙은 별도의 학습/검증 자료에서 정한다. '
             '현재 전체 test는 앞선 탐색에서 이미 확인한 자료이며 반복 seed가 독립적인 새 benchmark를 대신하지 않는다.',
             '원본 행 ID는 train/test에서 겹치지 않지만 동일 특징 벡터는 남을 수 있다. '
             '원래 프로토콜의 중복 특징 제외 진단을 각 실행에 유지한다. '
             '이번 반복은 알려지지 않은 공격의 라벨 확보 전 탐지나 실제 네트워크 간 전이를 검증하는 실험이 아니다.',
             '## 실행 근거와 인계',
             f'실행 ID는 `{run.name}`, 경로는 `{run.relative_to(ROOT)}`이다. '
             '설정은 `configs/experiments/exp84_multiseed_protocol.json`, 현재 진행은 실행 폴더의 progress.json과 seed별 progress.json에 남긴다. '
             'seed별 전체 test 예측·단계별 모델·context ID·패널 예측과 검증 기록을 보존한다.',
             '집계 수치는 `docs/lablog/data/exp84_multiseed_summary.csv`, 클래스별 수치는 `exp84_multiseed_per_class_summary.csv`, '
             '같은 seed XGB 대비 차이는 `exp84_multiseed_paired_xgb_summary.csv`다. '
             '전체 검증이 끝난 seed만 집계하며 완료/실패 시 이 보고서와 한 줄 로그를 갱신하고 임시 PDF로 컴파일 검증한다.']
    if state['state']=='failed':
        text.append('중단 원인: '+state.get('error','FAILURE.json 참조')+'. 완료된 seed만 비교에 포함한다. 원인과 소스 snapshot 일치를 확인한 뒤 같은 실행 폴더에서 미완료 조건을 재개할 수 있다.')
    elif state['state']!='complete':
        text.append(f'남은 작업은 신규 {120-new_done}조건의 실행 및 seed별 전체 예측 재검증이다. 현재 표시는 중간 상태이며 세 seed 반복의 최종 결론이 아니다.')
    target=write_record(ROOT/f'docs/lablog/{cfg["report_date"]}/exp84.md','\n\n'.join(text),
        summary=f'seed 43 기존 결과 + 44·45 반복; {status}, 새 예측 {new_done}/120조건, 전체 검증 seed {validated}; 평균·표본 표준편차·클래스별 변동',compile_pdf=False)
    info=build_report(target)
    vp=ROOT/'docs/lablog/_support/validation.json';v=read(vp)
    previous={x['report']:x for x in v.get('reports',[])};previous[info['report']]=info
    v['reports']=[previous[k] for k in sorted(previous)];v['structure']=check();assert v['structure']['passed']
    write(vp,v)
    sp=ROOT/'docs/lablog/_support/sources.json';sources=read(sp)
    entry=dict(path=str(target.relative_to(ROOT)),title='EXP84 · 3 seed 반복',sha256=sha(target),
               pdf_persisted=False,last_compiled_pdf_sha256=info['compiled_pdf_sha256'],
               state=state['state'],run=str(run.relative_to(ROOT)))
    key=next(k for k,val in sources.items() if isinstance(val,list) and any(isinstance(x,dict) and str(x.get('path','')).startswith('docs/lablog/reports/') for x in val))
    sources[key]=[x for x in sources[key] if x.get('path')!=entry['path']]+[entry];write(sp,sources)
    write(run/'aggregate_validation.json',dict(passed=True,state=state['state'],validated_seeds=validated,
          aggregate_rows=len(frame),conditions=len(summary),new_saved_jobs=new_done,
          standard_deviation='sample; ddof=1',report_compile=info,
          files={str(dest/f'exp84_multiseed_{suffix}.csv'):sha(dest/f'exp84_multiseed_{suffix}.csv') for suffix in outputs}))
    print(dict(state=state['state'],validated_seeds=validated,new_saved_jobs=new_done,report=str(target)),flush=True)


if __name__=='__main__':main()
