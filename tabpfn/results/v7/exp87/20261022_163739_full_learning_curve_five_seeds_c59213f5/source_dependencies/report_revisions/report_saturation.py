"""Validate predictions and report EXP87 without retaining a compiled report PDF."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').exists())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap,repo_root
bootstrap(_root)
ROOT=repo_root(__file__)
import argparse,json,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from exp61_utils import write,sha
from lablog import write_record,build_report
from curve_report import rewrite

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--extension',type=Path);args=parser.parse_args()
    out=args.run.resolve();cfg=json.loads((out/'config.json').read_text());r=json.loads((out/'results.json').read_text())
    jobs=r['jobs'];protocol=cfg['design'];now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat();date=cfg['report_date']
    rows=[];classes=[];checks=[];lookup={}
    for j in jobs:
        ds=j['dataset'];names=j['class_names'];b=names.index('benign');pred=np.load(j['predictions_path'],mmap_mode='r')
        y=np.load(out/'data'/ds/'test_y.npy',mmap_mode='r')
        assert len(pred)==len(y) and not np.any(y==b) and not np.any(pred==b)
        cm=np.bincount(y.astype('int64')*len(names)+pred,minlength=len(names)**2).reshape(len(names),len(names))
        assert cm.tolist()==j['metrics']['confusion']
        assert j['context_rows']==j['n']*(len(names)-1) and j['train_class_counts'][b]==0
        rows.append(dict(dataset=ds,seed=j['seed'],n=j['n'],method=j['method'],macro_f1=j['metrics']['macro_f1'],
            preparation_seconds=j['preparation_seconds'],inference_seconds=j['inference_seconds'],context_rows=j['context_rows'],
            overlap_rows=j['exact_feature_overlap_attack_rows'],unique_train_features=j['training_unique_feature_rows']))
        for c,x in enumerate(j['metrics']['classes']):
            if c==b:continue
            classes.append(dict(dataset=ds,seed=j['seed'],n=j['n'],method=j['method'],class_name=names[c],**{k:x[k] for k in ['support','precision','recall','f1','TP','FP','FN']}))
        lookup[ds,j['seed'],j['n'],j['method']]=j
        checks.append(dict(job=Path(j['predictions_path']).parent.name,confusion_recomputed=True,benign_rows=0))
    for (ds,seed,n,m),j in lookup.items():
        other=lookup.get((ds,seed,n,'pfn' if m=='xgb' else 'xgb'))
        if other:assert j['train_source_ids']==other['train_source_ids']
    text=['# EXP87 · 공격별 표본 수 증가와 성능 포화 구간 탐색','## 목적과 현재 상태',
        f"현재 상태는 {r['state']} / {r['stage']}, 완료 모델 조건은 {len(jobs)}개다. 실행 시작은 {cfg['started_at']}, 기록 시각은 {now}다. 실행은 v7/EXP87에 등록했으며 경로는 `{out.relative_to(ROOT)}`다.",
        '공격당 사례를 2개부터 늘렸을 때 각 모델의 공격 Macro-F1이 어느 범위에서 평탄해지는지, 두 모델의 차이가 얼마나 되는지 구분한다. 2026-10-22 사용자 요청에 따라 정상 학습·정상 test 추론을 제외한다. 수학적 수렴이나 통계적 동등성을 입증하는 실험이 아니라 관측 범위의 학습 곡선을 확인한다.',
        '## 공통 설정과 표본 선택',
        'XGB는 기존 300 trees/depth8/lr0.05/subsample0.8/colsample0.8/min_child_weight1을 유지한다. PFN은 기존 v3 checkpoint, 4 estimators, fit_with_cache, CPU cache이며 가중치 FT와 context subsampling을 하지 않는다. 초기 1000개 context를 사용하지 않으며 각 공격당 정확히 n개씩 제공한다. 같은 seed·n에서 두 모델의 입력 행 ID가 같다.',
        'EXP86에서 사용한 seed별 각 공격의 첫 20행을 그대로 유지하고, 같은 클래스의 남은 정제 train에서 중복 없이 무작위 추출하여 뒤에 붙인다. 따라서 작은 n의 표본은 큰 n에 포함되며, 정상 없이 균형 context를 구성한다. CIC 공격 6종, ToN 공격 9종이므로 전체 context는 각각 6n, 9n행이다.',
        '| 데이터 | 탐색할 공격당 n | 균형 표본 상한 이유 | 고정 공격 test |',
        '|---|---|---|---:|',
        '| CIC2018 | 2/5/10/20/40/80/160/320/450 | web_attacks train 450개 | 390,270 |',
        '| ToN-IoT | 2/5/10/20/40/80/160/320/640/1280/2151 | ransomware train 2,151개 | 797,421 |',
        '표본 상한을 넘기기 위해 희귀 클래스를 복제하거나 클래스별 다른 n을 적용하지 않는다. 같은 고정 test의 공격만을 사용하며, 모든 n에서 평가 모집단과 클래스별 test 표본 수가 같다. 기존 query 순서에서 정상 위치만 제거하고 batch8192로 추론한다. n=2/5/10/20도 이 방식으로 다시 추론하여 정상 포함 query 배치와의 차이가 곡선 중간에 섞이지 않도록 한다.',
        '## 사전에 정한 탐색·검증 규칙',
        '먼저 seed43만 실행한다. 두 모델 각각 최근 세 n의 Macro-F1 최대−최소가 1%p 이내이고 n이 80 이상이면 포화 후보로 판단한다. 다음으로 더 큰 n에서도 최근 세 점 조건을 만족해야 단일 seed의 후보 구간을 확정한다. 모델 한쪽만 평탄하면 탐색을 계속한다.',
        '후보가 정해지면 마지막 세 n을 seed44–47로 실행하고 seed43과 합쳐 평균·표본 표준편차를 계산한다. 각 모델의 5-seed 평균이 해당 세 점에서 1%p 이내에 있는지 확인하며 개별 seed의 범위도 공개한다. 평균 조건이 실패하면 다음 n을 5개 seed 모두에서 추가하고 가장 최근 세 n을 다시 확인한다. 가용 라벨 상한에 도달해 종료했으면 포화 확인 여부와 별도로 상한 도달을 명시한다.',
        'PFN−XGB 차이의 절댓값 1%p 이내는 점수가 가깝다는 기술적 기준이다. 각 모델의 학습 곡선이 평탄하다는 조건과 별개이며 통계적 동등성을 뜻하지 않는다. 탐색에 이미 검토한 고정 test를 사용하므로 표본 수 선택은 탐색적 분석이다. 다른 seed 반복은 표본 선택·모델 변동의 확인이며 독립된 미공개 test 검증은 아니다.',
        '## Feature-disjoint attack의 의미',
        'EXP86의 feature-disjoint attack은 해당 학습/context에 포함한 특징 벡터와 완전히 동일한 공격 test 행을 제외한 보조 집계다. 서로 다른 원본 행 ID라도 숫자 특징 전체가 같을 수 있다. 이는 다른 특징 열을 사용한다는 뜻이나 세션·근접 중복까지 제거했다는 뜻이 아니다. 제외 행이 n·seed마다 달라지므로 EXP87의 주 결과와 포화 판단에는 사용하지 않고 고정 attack-only test를 유지한다. 동일 특징 중첩 수는 결과 JSON/CSV에 진단 자료로 남긴다.',
        '## 현재 탐색 결정']
    for ds,d in r['decisions'].items():
        text.append(f"{ds}: seed43 측정 n={d['pilot_budgets']}, 단일 seed 후보 확인={d['pilot_plateau_confirmed']}, 다중 seed 상태={d['validation_state']}, 라벨 상한 도달={d['cap_reached']}.")
        status=d.get('validation_window') or d.get('pilot_window',{})
        if status.get('complete'):
            text.append(f"최근 확인 구간 {status['budgets']}, 사용 seed {status['seeds']}: XGB 점수 범위 {status['mean_ranges']['xgb']*100:.2f}%p, PFN 점수 범위 {status['mean_ranges']['pfn']*100:.2f}%p. 해당 범위는 최대−최소 차이다.")
        if d.get('validation_state')=='complete':
            text.append('5-seed 평균에서 포화 기준을 만족했다.' if d['final_plateau_confirmed'] else '가용 라벨 상한까지 평가했지만 5-seed 평균의 포화 기준은 만족하지 않았다. 더 큰 균형 n은 현재 train pool에서 구성할 수 없다.')
    if rows:
        f=pd.DataFrame(rows);c=pd.DataFrame(classes);f.to_csv(out/'values.csv',index=False);c.to_csv(out/'per_class.csv',index=False)
        summary=f.groupby(['dataset','n','method']).agg(seeds=('seed','nunique'),mean_f1=('macro_f1','mean'),sd_f1=('macro_f1','std'),
            mean_preparation=('preparation_seconds','mean'),mean_inference=('inference_seconds','mean')).reset_index()
        summary.to_csv(out/'summary.csv',index=False)
        paired=f.pivot(index=['dataset','seed','n'],columns='method',values='macro_f1').dropna()
        if 'pfn' in paired and 'xgb' in paired:
            paired['pfn_minus_xgb']=paired.pfn-paired.xgb;paired.to_csv(out/'paired.csv')
        else:
            paired=pd.DataFrame()
        figures=ROOT/'docs/lablog/figures';colors={'xgb':'#242424','pfn':'#0072B2'}
        for phase in ['pilot','validation']:
            fig,axes=plt.subplots(1,2,figsize=(11,3.8))
            for ax,ds in zip(axes,protocol['datasets']):
                plotted=False
                for m in ['xgb','pfn']:
                    q=f[(f.dataset==ds)&(f.method==m)]
                    if phase=='pilot':q=q[q.seed==43]
                    group=q.groupby('n').macro_f1.agg(['mean','std','count'])
                    if phase=='validation':group=group[group['count']>=2]
                    if group.empty:continue
                    ax.errorbar(group.index,group['mean']*100,yerr=group['std'].fillna(0)*100,color=colors[m],marker='o',capsize=3,label=m.upper());plotted=True
                ax.set_xscale('log',base=2);ax.set_xticks(protocol['budgets'][ds]);ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.tick_params(axis='x',labelrotation=45,labelsize=8);ax.set(ylim=(0,100),title=ds,xlabel='Labeled rows per attack class (log axis)',ylabel='Attack-only Macro-F1 (%)');ax.grid(alpha=.2)
                if plotted:ax.legend()
                else:ax.text(.5,.5,'Awaiting repeated seeds',ha='center',transform=ax.transAxes)
            fig.tight_layout();fig.savefig(figures/f'exp87_{phase}.pdf');plt.close(fig)
            caption='Seed43 탐색; 가로축 log 간격; 후속 확장의 seed43 점도 포함' if phase=='pilot' else '2개 이상 seed가 완료된 조건의 평균±표본 표준편차; 최종 검증은5 seeds'
            text.extend(['## '+('단일 seed 학습 곡선' if phase=='pilot' else '다중 seed 확인'),f'![{caption}](../../figures/exp87_{phase}.pdf)'])
        text.extend(['## 전체 측정 수치','표의 반복 수가 1이면 seed43 탐색 결과이며 표준편차를 계산하지 않는다. 반복 수가 다른 행은 같은 정밀도의 확정 비교가 아니다.',
            '| 데이터 | n | 모델 | seeds | 공격 F1 (%) | 준비 (초) | 공격 test 추론 (초) |','|---|---:|---|---:|---:|---:|---:|'])
        for _,z in summary.iterrows():
            metric=f'{z.mean_f1*100:.2f} ± {z.sd_f1*100:.2f}' if z.seeds>1 else f'{z.mean_f1*100:.2f}'
            text.append(f'| {z.dataset} | {int(z.n)} | {z.method} | {int(z.seeds)} | {metric} | {z.mean_preparation:.3f} | {z.mean_inference:.2f} |')
        text.extend(['## 모델 간 차이와 마지막 측정점',
            '두 모델 점수 차이가 작아지는 것과 두 모델 각각 더 많은 사례에서 개선되지 않는 것은 다른 조건이다. 한 점에서 가까웠다가 다시 벌어질 수 있으므로 전체 곡선과 마지막 확인 구간을 함께 본다.'])
        for ds in protocol['datasets']:
            q=paired.loc[ds] if len(paired) and ds in paired.index.get_level_values(0) else pd.DataFrame()
            if q.empty:continue
            for label,part in [('seed43',q[q.index.get_level_values('seed')==43]),('반복 확인',q)]:
                g=part.groupby('n').pfn_minus_xgb.agg(['mean','count'])
                if label=='반복 확인':g=g[g['count']==5]
                if g.empty:continue
                close=g.index[g['mean'].abs()<=.01].tolist();last=g.iloc[-1]
                text.append(f"{ds} {label}: 측정한 조건 중 평균 점수 차이 절댓값이 1%p 이내인 n={close}. 가장 큰 완료 n={g.index[-1]}의 PFN−XGB 차이는 {last['mean']*100:.2f}%p다.")
        text.extend(['## 마지막 측정점의 클래스별 품질',
            '각 데이터에서 현재 가장 큰 완료 n의 수치이며, 가능한 seed만 평균한다. 낮은 F1이 누락 때문인지 다른 공격의 false positive 유입 때문인지 precision·recall을 함께 본다. 행은 고정 공격 test 표본 수 내림차순이다.',
            '| 데이터 / n | 클래스 | 모델 | seeds | Precision (%) | Recall (%) | F1 (%) |','|---|---|---|---:|---:|---:|---:|'])
        for ds in protocol['datasets']:
            data=c[c.dataset==ds]
            if data.empty:continue
            n=int(data.n.max());data=data[data.n==n]
            order=data.groupby('class_name').support.max().sort_values(ascending=False).index
            for name in order:
                for m in ['xgb','pfn']:
                    z=data[(data.class_name==name)&(data.method==m)]
                    if len(z):text.append(f'| {ds} / {n} | {name} | {m} | {z.seed.nunique()} | {z.precision.mean()*100:.2f} | {z.recall.mean()*100:.2f} | {z.f1.mean()*100:.2f} |')
        text.extend(['## 기존 EXP86과 query 배치 변경 점검',
            '동일한 n=20 학습 행·공격 test에서 정상 query를 제거한 실행과 EXP86의 예측을 비교한다. 모든 EXP87 n은 공격만 입력하는 방식으로 실행했다. 아래 비교는 query 구성·배치 경계 변경에 대한 진단이며 기존 전체 test 추론시간과 새 공격-only 추론시간은 평가 행 수가 다르다.'])
        for j in jobs:
            a=j.get('anchor_comparison')
            if a:text.append(f"{j['dataset']} seed{j['seed']} {j['method']}: 예측 불일치 {a['prediction_disagreement_count']:,}행 ({a['prediction_disagreement_rate']*100:.4f}%), Macro-F1 변화 {a['macro_f1_delta']*100:.4f}%p.")
        text.extend(['## 해석의 범위와 다음 판단',
            'CIC와 ToN은 공격 클래스 수·공격의 정의·특징 분포·test 불균형이 다르다. 같은 n에서 점수가 높은 현상을 현재 설정의 표본 효율 차이로 기록하며 데이터셋 전체의 고유 난이도 차이로 단정하지 않는다. Macro-F1 포화가 모든 희귀 공격의 포화를 뜻하지도 않으므로 클래스별 곡선은 per_class.csv로 추적한다.',
            '준비시간은 모델 생성과 XGB fit 또는 PFN context/cache 준비의 합이며 추론을 제외한다. 순차 갱신 실험이 아니므로 이를 후속 갱신 비용이라고 부르지 않는다. 표본 상한에서 개선이 계속된다면 더 큰 균형 n이 가능한 새로운 데이터 또는 별도 불균형 배분 시나리오가 필요하다. 이번 실행에서 자동으로 oversampling이나 클래스별 편입 제한을 도입하지 않는다.'])
    text.extend(['## 재현·검증 자료',
        f'실행 경로 `{out.relative_to(ROOT)}`의 config/protocol, source_dependencies, data, jobs, events/decisions, CSV를 확인한다. 예측 혼동행렬과 모델 간 입력 ID를 재검증했다. 원고·슬라이드는 자동 변경하지 않는다.'])
    if rows:
        text=rewrite(text,f,c,summary,r,protocol,ROOT,out)
    extension_validation=None
    if args.extension:
        from extension_report import augment
        text,extension_validation=augment(text,args.extension.resolve(),out,ROOT)
    expected=10*sum(map(len,protocol['budgets'].values()))
    summary=f"EXP86 학습곡선 확장; 모든 n에 동일 5 seeds, 평균 수치·모델 차이·클래스별 결과; {r['state']} {len(jobs)}/{expected}조건; 끝점은 균형 표본 상한"
    if extension_validation:summary+=f"; 클래스별 가용 수까지 확대 {extension_validation['state']} {extension_validation['jobs']}/10조건"
    revisions=out/'source_dependencies'/'report_revisions';revisions.mkdir(parents=True,exist_ok=True)
    for file in [Path(__file__),Path(__file__).with_name('curve_report.py')]:
        shutil.copy2(file,revisions/file.name)
    target=write_record(ROOT/'docs/lablog/evidence'/date/'exp87_record','\n\n'.join(text).replace('|\n\n|','|\n|'),summary=summary,compile_pdf=False)
    validation=build_report(target);write(out/'report_validation.json',dict(**validation,checked_at=now,validated_jobs=len(jobs),checks=checks,source_sha256=sha(Path(__file__))))
    if args.extension:
        write(args.extension/'report_validation.json',dict(**validation,checked_at=now,**extension_validation))
    print(json.dumps(dict(report=str(target),state=r['state'],stage=r['stage'],jobs=len(jobs)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
