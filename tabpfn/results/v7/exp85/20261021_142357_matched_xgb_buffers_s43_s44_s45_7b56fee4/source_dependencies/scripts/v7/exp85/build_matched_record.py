#!/usr/bin/env python3
"""Korean report of exact-buffer XGB/PFN comparisons; no permanent report PDF."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap,repo_root
bootstrap(_root)
ROOT=repo_root(__file__)
import argparse,json
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from exp61_utils import write,sha
from small_label_updates import read
from build_small_label_record import table
from build_multiseed_record import aggregate,format_stat,METRICS
from lablog import write_record,build_report,check

POLICY_NAMES={'accumulate':'누적','fifo':'FIFO','random':'무작위','similar':'유사도'}


def extract(path):
    d=read(path);m=d['metrics'];method=d['method']
    model='xgb' if method.startswith('xgb') else 'pfn'
    policy='accumulate' if method in ['xgb','pfn_accumulate'] else method.split('_',1)[1]
    head=dict(seed=d['seed'],dataset=d['dataset'],n=d['new_per_class'],method=method,model=model,policy=policy)
    row=dict(**head,macro_f1_pct=100*m['macro_f1'],new_attack_f1_pct=100*d['new_attack_mean_f1'],
        initial_attack_f1_pct=100*d['initial_attack_mean_f1'],benign_fpr_pct=100*m['benign_false_positive_rate'],
        initial_retention_change_pp=100*d['initial_panel_final_change'],
        worst_update_old_change_pp=100*d['worst_update_old_change'],update_seconds=d['update_seconds'],
        initial_seconds=d['initial_seconds'],inference_seconds=d['serving_seconds'],
        context_rows=d['context_rows'],labels=d['label_budget'],source=str(path))
    classes=[dict(**head,name=c['name'],support=c['support'],precision_pct=100*c['precision'],
                  recall_pct=100*c['recall'],f1_pct=100*c['f1'],TP=c['TP'],FP=c['FP'],FN=c['FN']) for c in m['classes']]
    return row,classes


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    run=a.run.resolve();cfg=read(run/'config.json');state=read(run/'results.json')
    rows=[];classes=[];validated=[];matching=[];counts=[]
    for seed in cfg['seeds']:
        source=Path(cfg['source_runs'][str(seed)])
        dest=run/f'seed_{seed}'
        good=(dest/'validation.json').exists() and read(dest/'validation.json')['state']=='complete' and read(dest/'validation.json')['passed']
        if good:validated.append(seed)
        for n in cfg['new_budgets']:
            for ds in cfg['datasets']:
                names=read(source/'data'/ds/'manifest.json')['class_names']
                for method in ['xgb','pfn_accumulate']+['pfn_'+s for s in cfg['policies']]:
                    path=source/f'new_{n}'/'jobs'/f'{ds}_{method}'/'results.json'
                    row,c=extract(path);rows.append(row);classes.extend(c)
                    if method.startswith('pfn'):
                        d=read(path)
                        for f in d['fits']:
                            for name,count in zip(names,f['context_class_counts']):
                                counts.append(dict(seed=seed,dataset=ds,n=n,policy=row['policy'],stage=f['stage_number'],
                                    name=name,rows=count,total_rows=f['context_rows'],fraction=count/f['context_rows']))
                if good:
                    for policy in cfg['policies']:
                        path=dest/f'new_{n}'/'jobs'/f'{ds}_xgb_{policy}'/'results.json'
                        row,c=extract(path);rows.append(row);classes.extend(c)
                        x=read(path);f=read(Path(x['reference_pfn_result']))
                        assert x['matched_rows_all_stages']
                        assert all(a['context_indices']==b['context_indices'] for a,b in zip(x['fits'],f['fits']))
                        matching.append(dict(seed=seed,dataset=ds,n=n,policy=policy,matched_stages=len(x['fits'])))
    if state['state']=='complete':assert validated==cfg['seeds'] and len(matching)==cfg['expected_jobs']
    frame=pd.DataFrame(rows);class_frame=pd.DataFrame(classes)
    # Pair only seeds with all requested XGB jobs validated, to keep every curve comparable.
    comparable=frame[frame.seed.isin(validated)]
    class_comparable=class_frame[class_frame.seed.isin(validated)]
    summaries={};figs=[]
    if len(comparable):
        keys=['dataset','n','method','model','policy']
        summary=aggregate(comparable,keys,METRICS)
        cs=aggregate(class_comparable,keys+['name','support'],['precision_pct','recall_pct','f1_pct','TP','FP','FN'])
        pfn=comparable[comparable.model=='pfn'];xgb=comparable[comparable.model=='xgb']
        paired=pfn.merge(xgb,on=['seed','dataset','n','policy'],suffixes=('','_xgb'),validate='one_to_one')
        for metric in METRICS:paired[metric+'_minus_xgb']=paired[metric]-paired[metric+'_xgb']
        paired_summary=aggregate(paired,['dataset','n','policy'],[m+'_minus_xgb' for m in METRICS])
        summaries={'summary':summary,'per_class_summary':cs,'paired':paired,'paired_summary':paired_summary}
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt,font_manager
        font='/home/user/.local/share/fonts/Pretendard/Pretendard-Regular.otf';font_manager.fontManager.addfont(font)
        plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False})
        policies=['accumulate']+cfg['policies']
        for metric,title,suffix in [('macro_f1_pct','전체 Macro-F1 (%)','quality'),('new_attack_f1_pct','신규 공격군 평균 F1 (%)','new_attacks'),
              ('initial_retention_change_pp','초기 클래스 최종−초기 (pp)','retention'),('update_seconds','후속 갱신 시간 합계 (초)','cost')]:
            fig,axs=plt.subplots(len(policies),2,figsize=(10,2.5*len(policies)),squeeze=False,layout='constrained')
            for i,policy in enumerate(policies):
                for j,ds in enumerate(cfg['datasets']):
                    ax=axs[i,j]
                    for model,color,marker in [('xgb','#111111','s'),('pfn','#0072B2','o')]:
                        part=summary[(summary.dataset==ds)&(summary.policy==policy)&(summary.model==model)].sort_values('n')
                        ax.errorbar(range(len(part)),part[metric+'_mean'],yerr=part[metric+'_sd'].fillna(0),
                                    color=color,marker=marker,capsize=2,label=model.upper())
                    ax.set_title(ds+' · '+POLICY_NAMES[policy]);ax.set_xticks(range(6),cfg['new_budgets'])
                    ax.set_xlabel('신규 클래스당 라벨 수');ax.set_ylabel(title);ax.grid(axis='y',alpha=.2)
                    if i==0 and j==0:ax.legend(frameon=False)
            fig.suptitle(f'동일 사례 XGB/PFN · {len(validated)} seed 평균'+(' ± 표본 표준편차' if len(validated)>1 else ' · SD 미산출'))
            path=ROOT/f'docs/lablog/figures/exp85_{suffix}.pdf';fig.savefig(path);plt.close(fig);figs.append((title,path))
    dest=ROOT/'docs/lablog/data';outputs={'values':frame,'per_class':class_frame,'context_counts':pd.DataFrame(counts),**summaries}
    for suffix,data in outputs.items():
        for directory in [run,dest]:data.to_csv(directory/f'exp85_{suffix}.csv',index=False)
    status={'complete':'완료','running':'실행 중','preparing':'준비 중','failed':'실패·부분 완료'}[state['state']]
    now=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')
    text=['# EXP85 · 동일 고정 버퍼의 XGB 재학습과 PFN context 비교',
        '## 목적과 실행 상태',
        f'모델의 효과와 사례 구성의 효과를 분리하기 위해 EXP84에서 PFN이 사용한 각 단계의 버퍼를 XGB 학습에 그대로 제공한다. '
        f'연구 버전은 v7이며 현재 상태는 {status}, 새 실행 {state["completed_jobs"]}/{cfg["expected_jobs"]}조건이다. '
        f'전체 예측 검증을 마친 seed는 {validated}이며 기록 시각은 {now}다. '
        '사용자는 2026-10-21에 XGB 버퍼 대조군 추가와 슬라이드 개편을 요청했다.',
        '기존 EXP84의 seed 43·44·45는 완료되었으며 PFN 예측과 누적 XGB 결과를 재사용한다. '
        '무작위 정책도 단계별 실제 행 ID가 저장되어 있어 PFN 재실행이 필요 없다. '
        'XGB 버퍼 조건의 학습·예측·시간은 새로 측정하며, 기존 PFN과는 실행 세션이 다르다는 점을 비용 해석에 포함한다.',
        '## 설정과 단계별 재학습 이유',
        '신규 공격당 5·10·20·30·50·100개, CIC2018과 ToN-IoT, seed 43·44·45를 유지한다. '
        '초기 사례는 정상 750개와 초기 공격 두 클래스 각각 125개다. '
        'CIC은 초기 Benign/Brute force/DoS 이후 DDoS, Web attacks, Infiltration, Bot을 도입한다. '
        'ToN은 초기 Benign/Scanning/DoS 이후 Injection/DDoS, Password, XSS, Ransomware/Backdoor, MitM을 도입한다.',
        '고정 버퍼는 1,000개이며 공급된 신규 사례를 모두 편입하고 같은 수의 기존 사례를 삭제한다. '
        'FIFO·무작위·유사도는 EXP84의 동일 규칙이며 기존 클래스별 최소 1개를 보존한다. '
        f'이번 추가 정책은 {", ".join(POLICY_NAMES[p] for p in cfg["policies"])}이다. '
        '삭제를 실제로 다시 계산하면서 시간을 측정하고, 각 단계에서 PFN의 저장된 행 ID·순서와 완전히 일치하는지 검사한다.',
        '최종 품질만 필요하다면 마지막 버퍼로 XGB를 한 번 학습해도 된다. '
        '이번 비교는 후속 갱신 시간 합계와 기존 클래스 유지도 평가하므로 모든 단계에서 XGB를 새로 재학습한다. '
        '각 단계에서는 고정 패널만 예측하고 전체 test는 마지막 모델에서만 예측한다. '
        'XGB는 300 trees, depth 8, learning rate 0.05, CUDA histogram, threads 8이며 PFN은 IDS 가중치 fine-tuning 없는 기존 v3 결과다.',
        '## 평가와 비용 범위',
        '최종 전체 test는 CIC 3,042,473개와 ToN 2,271,723개이며 seed 43의 같은 query 순서와 batch 8,192개를 유지한다. '
        '신규 공격군 평균은 마지막 전체 test 혼동행렬에서 CIC 4개·ToN 7개 신규 공격의 F1을 같은 비중으로 평균한다. '
        '기존 성능 유지는 클래스당 최대 2,000개 고정 패널에서 초기 세 클래스의 최종−초기 Macro-F1으로 측정한다. '
        '패널과 전체 test는 서로 다른 평가 모집단이며 패널의 미래 클래스 특징은 오프라인 진단에만 사용한다.',
        '주 비용은 2단계 이후 선택/삭제·모델 구성·fit/cache 시간의 합계다. '
        '초기 준비, 모든 추론, 체크포인트 저장, 검증은 제외하여 기존 PFN과 기준을 맞춘다. '
        '추론 시간은 별도 수치로 보존하며 갱신 시간의 우위를 전체 운영 비용 우위로 해석하지 않는다. '
        '정확히 같은 행을 사용하지만 모델 학습 방법과 사전학습 여부는 각 방법의 특성으로 유지한다.',
        '평균과 표본 표준편차(ddof=1)는 조건별 seed에서 계산한다. '
        'XGB 대비 PFN 차이는 같은 seed·같은 정책·같은 행의 결과끼리 뺀 뒤 집계한다. '
        '부분 완료 시 전체 검증을 마친 동일 seed 집합만 양쪽 곡선에 사용한다. '
        '세 seed는 고정 train/test 분할에서의 반복이며 표준편차는 신뢰구간이나 동등성 검정이 아니다.',
        '## 도표와 관찰 결과']
    if len(comparable):
        for title,path in figs:text.append(f'![{title}; 동일 사례 비교]({path})')
        for ds in cfg['datasets']:
            text.append('### '+ds+' · 전체 표본 수 비교')
            vals=[]
            for n in cfg['new_budgets']:
                for policy in ['accumulate']+cfg['policies']:
                    for model in ['xgb','pfn']:
                        row=summary[(summary.dataset==ds)&(summary.n==n)&(summary.policy==policy)&(summary.model==model)].iloc[0]
                        vals.append([n,POLICY_NAMES[policy],model.upper(),format_stat(row,'macro_f1_pct'),
                                     format_stat(row,'new_attack_f1_pct'),format_stat(row,'update_seconds',3)])
            text.append(table(['n','보유 방식','모델','전체 F1 (%)','신규 F1 (%)','갱신 (초)'],vals))
            for policy in cfg['policies']:
                d=paired_summary[(paired_summary.dataset==ds)&(paired_summary.n==100)&(paired_summary.policy==policy)].iloc[0]
                text.append(f'n=100 {POLICY_NAMES[policy]}에서 PFN−XGB 차이는 전체 F1 {format_stat(d,"macro_f1_pct_minus_xgb")} pp, '
                    f'신규 공격군 F1 {format_stat(d,"new_attack_f1_pct_minus_xgb")} pp, 갱신 시간 {format_stat(d,"update_seconds_minus_xgb",3)}초다. '
                    '성능 차이의 양수는 PFN 우세, 시간 차이의 음수는 PFN이 빠른 방향이다.')
        text.append('## 클래스별 n=100 결과')
        text.append('클래스는 각 데이터셋 전체 test 표본 수 내림차순이다. 모든 n의 클래스별 precision·recall·F1과 정상 오탐률은 CSV에 보존한다.')
        for ds in cfg['datasets']:
            for policy in cfg['policies']:
                part=cs[(cs.dataset==ds)&(cs.n==100)&(cs.policy==policy)]
                names=part[['name','support']].drop_duplicates().sort_values('support',ascending=False)
                vals=[]
                for _,c in names.iterrows():
                    x=part[(part.name==c['name'])&(part.model=='xgb')].iloc[0]
                    f=part[(part.name==c['name'])&(part.model=='pfn')].iloc[0]
                    vals.append([c['name'],int(c.support),format_stat(x,'f1_pct'),format_stat(f,'f1_pct'),
                                 format_stat(x,'precision_pct'),format_stat(f,'precision_pct')])
                text.append('### '+ds+' · '+POLICY_NAMES[policy])
                text.append(table(['클래스','test 수','XGB F1','PFN F1','XGB 정밀도','PFN 정밀도'],vals))
    else:text.append('추가 XGB 결과의 전체 예측 검증이 아직 끝나지 않아 동일 버퍼 모델 비교 수치를 제시하지 않는다.')
    text += ['## 해석과 후속 판단',
        '같은 버퍼에서 XGB와 PFN이 모두 개선된다면 데이터 구성 효과를 먼저 검토한다. '
        'PFN이 비슷한 품질로 더 빠른 갱신을 보이면 가중치 재학습 생략의 적용상 이점으로 해석할 수 있다. '
        'XGB가 같은 버퍼에서 더 좋은 품질과 충분히 낮은 비용을 보이면 해당 조건의 PFN 채택 근거를 재검토한다. '
        '낮은 희소 클래스 F1은 클래스별 정밀도·재현율·오탐을 분석하여 개선 방향을 정하며 전체 평균만으로 가리지 않는다.',
        '버퍼 정책은 클래스별 보유 비율도 바꾼다. '
        'FIFO는 확보 순서에 따른 삭제이며 실제 트래픽 노후화 판정이 아니다. '
        '새 클래스만 들어오는 현재 시나리오에서 정상·기존 공격을 추가 공급하지 않으므로 실제 운영 분포 변화와 동일하다고 주장하지 않는다. '
        '원본 행 ID는 train/test에서 다르지만 같은 특징 벡터는 남을 수 있으며 기존 중복 특징 제외 진단을 유지한다.',
        '## 실행 근거',f'실행 경로는 `{run.relative_to(ROOT)}`이며 설정은 `configs/experiments/exp85_protocol.json`이다. '
        '원본 PFN 결과는 config.json의 source_runs에 연결한다. '
        '실측값은 `docs/lablog/data/exp85_values.csv`, 집계는 `exp85_summary.csv`, 대응 차이는 `exp85_paired_summary.csv`, '
        '단계별 클래스 구성은 `exp85_context_counts.csv`에 있다. '
        '실험 결과와 논문 근거 채택은 구분하며 이 생성기는 원고나 슬라이드를 수정하지 않는다.']
    if state['state']=='failed':text.append('실패 원인: '+state.get('error','FAILURE.json 참조')+'. 소스와 저장된 완료 조건을 검증한 뒤 미완료 조건을 재개한다.')
    target=write_record(ROOT/f'docs/lablog/{cfg["report_date"]}/exp85.md','\n\n'.join(text),
        summary=f'PFN과 같은 버퍼 행의 XGB 재학습; {status} {state["completed_jobs"]}/{cfg["expected_jobs"]}조건, seed 43·44·45, 전체 test·갱신 비용·클래스별 대응 비교',compile_pdf=False)
    info=build_report(target);vp=ROOT/'docs/lablog/_support/validation.json';v=read(vp)
    previous={x['report']:x for x in v.get('reports',[])};previous[info['report']]=info
    v.update(reports=[previous[k] for k in sorted(previous)],structure=check());assert v['structure']['passed'];write(vp,v)
    sp=ROOT/'docs/lablog/_support/sources.json';sources=read(sp)
    entry=dict(path=str(target.relative_to(ROOT)),title='EXP85 · 동일 버퍼 XGB/PFN 비교',sha256=sha(target),
               pdf_persisted=False,last_compiled_pdf_sha256=info['compiled_pdf_sha256'],state=state['state'],run=str(run.relative_to(ROOT)))
    key=next(k for k,val in sources.items() if isinstance(val,list) and any(isinstance(x,dict) and str(x.get('path','')).startswith('docs/lablog/reports/') for x in val))
    sources[key]=[x for x in sources[key] if x.get('path')!=entry['path']]+[entry];write(sp,sources)
    write(run/'aggregate_validation.json',dict(passed=True,state=state['state'],validated_seeds=validated,
        matched_conditions=len(matching),matching=matching,report_compile=info,files={k:sha(dest/f'exp85_{k}.csv') for k in outputs}))
    print(dict(state=state['state'],validated_seeds=validated,report=str(target)),flush=True)


if __name__=='__main__':main()
