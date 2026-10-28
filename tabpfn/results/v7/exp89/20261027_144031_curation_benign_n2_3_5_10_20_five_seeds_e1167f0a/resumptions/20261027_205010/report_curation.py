"""Report EXP89 states and validated results; compile PDF only in temporary storage."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap,repo_root
bootstrap(_root)
ROOT=repo_root(__file__)

import argparse
from datetime import datetime
import json
import shutil
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lablog import write_record,build_report,check
from exp61_utils import sha
from curation_benign import read,write,stamp

DS={'cic2018':'CIC2018','toniot':'ToN-IoT'}
METHOD={'xgb':'XGB','pfn':'TabPFN'}
VERSION={'original':'정제 전','clean':'정제 후'}
BUDGETS=[2,3,5,10,20]


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                      ['| '+' | '.join(map(str,r))+' |' for r in rows])


def ms(values,scale=100):
    a=np.asarray(values,dtype=float)*scale
    assert len(a)==5 and np.isfinite(a).all(),a
    return f'{a.mean():.2f} ± {a.std(ddof=1):.2f}'


def subset(frame,**keys):
    mask=np.ones(len(frame),dtype=bool)
    for k,v in keys.items():mask &= frame[k].to_numpy()==v
    return frame[mask]


def plots(run,frames,date):
    dest=ROOT/'docs/lablog/figures'
    dest.mkdir(exist_ok=True)
    paths=[]
    m=frames['metrics']
    colors={'xgb':'#2864A0','pfn':'#C26020'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    # Original fixed attack test: original and clean training are compared on
    # exactly the same rows. Clean-test sensitivity is kept in its own figure.
    for i,ds in enumerate(DS):
        for j,version in enumerate(['original','clean']):
            ax=axes[i,j]
            for method in METHOD:
                for benign in [False,True]:
                    g=subset(m,dataset=ds,train_version=version,method=method,benign_training=benign,
                             test_version='original',test_scope='attack').groupby('n').attack_macro_f1
                    mean=g.mean().reindex(BUDGETS)*100;sd=g.std().reindex(BUDGETS)*100
                    ax.plot(range(5),mean,marker='s' if benign else 'o',ls='--' if benign else '-',color=colors[method],
                            label=f'{METHOD[method]} / '+('with benign' if benign else 'attacks only'))
                    ax.fill_between(range(5),mean-sd,mean+sd,color=colors[method],alpha=.07)
                    for x,y in enumerate(mean):
                        ax.annotate(f'{y:.1f}',(x,y),xytext=(0,7 if benign else -12),textcoords='offset points',
                                    ha='center',fontsize=6,color=colors[method])
            ax.set(title=f'{DS[ds]} / {version} train',xticks=range(5),xticklabels=BUDGETS,
                   xlabel='Rows per supplied class',ylabel='Attack Macro-F1 (%)',ylim=(0,100))
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    path=dest/f'{date}_exp89_original_attack.pdf';fig.savefig(path);paths.append(path);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,ds in enumerate(DS):
        for j,measure in enumerate(['attack_macro_f1','benign_fpr']):
            ax=axes[i,j]
            for method in METHOD:
                for version in ['original','clean']:
                    g=subset(m,dataset=ds,train_version=version,method=method,benign_training=True,
                             test_version='original',test_scope='all').groupby('n')[measure]
                    mean=g.mean().reindex(BUDGETS)*100;sd=g.std().reindex(BUDGETS)*100
                    ax.plot(range(5),mean,marker='o',ls='--' if version=='clean' else '-',color=colors[method],
                            label=f'{METHOD[method]} / {version} train')
                    ax.fill_between(range(5),mean-sd,mean+sd,color=colors[method],alpha=.07)
            ax.set(title=f'{DS[ds]} / with benign training',xticks=range(5),xticklabels=BUDGETS,
                   xlabel='Rows per class',ylabel='Attack Macro-F1 (%)' if j==0 else 'Benign false positive rate (%)',ylim=(0,100))
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    path=dest/f'{date}_exp89_original_full.pdf';fig.savefig(path);paths.append(path);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for i,ds in enumerate(DS):
        for j,method in enumerate(METHOD):
            ax=axes[i,j]
            for benign in [False,True]:
                for tv in ['original','clean']:
                    g=subset(m,dataset=ds,train_version='clean',method=method,benign_training=benign,
                             test_version=tv,test_scope='attack').groupby('n').attack_macro_f1
                    mean=g.mean().reindex(BUDGETS)*100;sd=g.std().reindex(BUDGETS)*100
                    color='#3C805B' if benign else '#7B50A1'
                    ax.plot(range(5),mean,marker='o',ls='--' if tv=='clean' else '-',color=color,
                            label=('with benign' if benign else 'attacks only')+f' / {tv} test')
                    ax.fill_between(range(5),mean-sd,mean+sd,color=color,alpha=.07)
            ax.set(title=f'{DS[ds]} / {METHOD[method]} / clean train',xticks=range(5),xticklabels=BUDGETS,
                   xlabel='Rows per supplied class',ylabel='Attack Macro-F1 (%)',ylim=(0,100))
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=6)
    path=dest/f'{date}_exp89_test_curation.pdf';fig.savefig(path);paths.append(path);plt.close(fig)
    pc=frames['per_class']
    for ds in DS:
        meta=read(run/'data'/ds/'manifest.json')
        names=sorted([x for x in meta['class_names'] if x!='benign'],
                     key=lambda x:meta['original_test_counts'][meta['class_names'].index(x)],reverse=True)
        for scope in ['attack','all']:
            arms=[(v,b) for v in ['original','clean'] for b in [False,True]] if scope=='attack' else [(v,True) for v in ['original','clean']]
            fig,axes=plt.subplots(2,len(arms),figsize=(3.4*len(arms),7 if ds=='cic2018' else 8.5),layout='constrained',squeeze=False)
            for i,method in enumerate(METHOD):
                for j,(version,benign) in enumerate(arms):
                    ax=axes[i,j]
                    g=subset(pc,dataset=ds,train_version=version,method=method,benign_training=benign,
                             test_version='original',test_scope=scope)
                    a=g.pivot_table(index='name',columns='n',values='f1').reindex(index=names,columns=BUDGETS).to_numpy()*100
                    assert np.isfinite(a).all()
                    im=ax.imshow(a,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
                    for y in range(len(names)):
                        for x in range(5):
                            ax.text(x,y,f'{a[y,x]:.1f}',ha='center',va='center',fontsize=7,color='white' if a[y,x]>65 else 'black')
                    ax.set(xticks=range(5),xticklabels=BUDGETS,yticks=range(len(names)),yticklabels=names if j==0 else [],
                           title=f'{METHOD[method]} / {version}\n'+('with benign' if benign else 'attacks only'))
                    if i==1:ax.set_xlabel('Rows per supplied class')
            fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='Class F1 (%)')
            path=dest/f'{date}_exp89_classes_{ds}_{scope}.pdf';fig.savefig(path);paths.append(path);plt.close(fig)
    return paths


def basic_sections(run,cfg):
    design=cfg['design']
    date=cfg['started_at'][:10]
    today=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d')
    parts=[f'''# EXP89: 정상 소수 학습과 충돌 정제의 영향을 분리한 TabPFN·XGB 비교

## 목적과 승인 범위
정상 학습과 충돌 정제가 극소수 공격 분류 성능에 미치는 영향을 분리하는 것이 본 실험의 목적이다. 사용자 승인에 따라 클래스당 n=2·3·5·10·20에서 정제 전후 train, 정상 n개 포함 여부, 두 모델을 교차한다. 평가에는 원래 전체 test를 고정하고 정제 전후 및 정상 포함 여부를 같은 예측의 행 선택으로 구분한다. 본 연구 버전은 v7이며 새로운 연구 버전으로 전환하지 않는다.

실행 시작일은 {date}이고 문서 작성일은 {today}이다. 실험 ID는 EXP89이며 실행 ID는 {run.name}이다. 데이터 2종, train 버전 2개, 정상 학습 조건 2개, 예산 5개, seed 5개, 모델 2개를 합쳐 400개 조건을 수행한다. 완료 여부는 아래 실행 상태와 검증 기록을 따른다.

## 표본 선택과 모델 설정
클래스별 n개는 해당 train pool에서 균등 비복원 추출한 20개의 중첩 prefix다. seed는 43·44·45·46·47이며 같은 조건에서 두 모델에 정확히 같은 행 ID를 제공한다. 정제 후 공격 표본은 EXP86의 기존 선택을 재현하고 n=3만 중간 prefix로 추가한다. 정상 포함 조건은 공격 표본을 그대로 두고 정상 n개를 추가하므로 전체 학습 행 수는 CIC2018의 경우 6n에서 7n, ToN-IoT는 9n에서 10n으로 바뀐다.

정제 전후 train의 후보 모집단은 다르므로 같은 seed라도 선택된 행 ID는 달라질 수 있다. 따라서 train 정제 비교는 정제된 후보에서 표본을 뽑는 정책의 차이를 측정하며 특정 행의 삭제 효과만을 고정하는 비교가 아니다. 정상 추가 비교는 같은 공격 표본을 공유하지만 출력 클래스 수와 전체 context 크기도 함께 증가한다. 정상 2~20개는 정상의 충분한 다양성을 보장하지 않는다.

정제 전 버전은 기존 source의 46개 특징, family 라벨, 원래 outer split ID를 그대로 사용하고 충돌 행 제거만 적용하지 않은 버전이다. 원래 feature 변환과 행별 고정값 처리도 유지하며 전체 pool로 새로운 scaler나 특징 선택을 학습하지 않는다. 정제 후 버전은 train·validation·test 라벨을 함께 이용했던 기존 충돌 제거 결과를 재현하는 대조군이다. 정제 전 버전을 새 수집환경이나 오류 없는 정답 데이터로 간주하지 않는다.

XGBoost (XGB)는 기존 300 trees, depth 8, learning rate 0.05, subsample/colsample 0.8, min_child_weight 1 설정을 유지한다. TabPFN은 기존 v3 체크포인트, 4 estimators, fit_with_cache, CPU cache를 사용하며 context subsampling과 가중치 fine-tuning을 하지 않는다. 매 조건에서 새 모델을 구성하고 동일 seed를 사용한다. 새 하이퍼파라미터 탐색이나 유리한 클래스 선택을 수행하지 않는다.

## 평가 모집단과 지표
원래 전체 test의 모든 행에 대해 seed43의 라벨과 무관한 고정 순서와 batch 8,192로 한 번 예측한다. test 행을 표본 추출하거나 중복 제거하지 않는다. 정제 후 test는 원래 test에 포함된 고정 부분집합이며, 공격만 평가는 실제 라벨이 정상인 행만 제외한다. 공격을 정상으로 예측한 경우는 false negative로 남기고 정상 출력 열을 제거하거나 확률을 다시 정규화하지 않는다.

공격 Macro-F1은 공격 클래스별 F1의 단순 평균이며 모든 비교에서 같은 공격 클래스 목록을 사용한다. 정상 포함 평가의 공격 F1에는 정상에서 유입되는 false positive가 포함된다. 전체 클래스 Macro-F1은 정상까지 포함한 별도 지표이며 공격 Macro-F1과 같은 값으로 비교하지 않는다. 정상 false positive rate (FPR)는 실제 정상 중 공격으로 예측된 비율이며, 공격 탐지 recall과 공격 종류 분류 F1을 구분한다.

정상 없이 학습한 모델의 전체 test 결과는 정상 출력 라벨이 없어 정상 FPR 100%인 구조적 진단이다. 이 조건을 정상 탐지 성능의 주 근거로 사용하지 않는다. 정상 포함 학습 모델의 전체 test 결과와 동일 모델의 공격 부분집합 결과는 같은 예측에서 계산한다. 기존 EXP86/87과는 원래 test 복원으로 query 배치 구성이 달라지므로 별도의 연속성 점검을 수행한다.

평균과 표본 표준편차는 같은 고정 split에서 다섯 seed에 걸친 표본·모델 변동이다. 동일 seed끼리 차이를 계산하며 test 정제 비교에서는 동일 모델을 유지한다. 추가 seed는 새로운 수집환경이나 미검토 test에 대한 일반화 검증을 대신하지 않는다. 모델 실행 완료와 논문 근거 채택은 구분한다.
''']
    if (run/'RUN_META.json').exists():
        for resume in read(run/'RUN_META.json').get('resumptions', []):
            previous = resume['previous_progress']
            parts.append('## 중단 및 재개 이력\n'
                f"마지막 저장 진행 시각은 {previous['at']}이며, 재개 전 완료 조건은 {resume['completed_before']}/400개다. "
                f"사용자 재개 요청에 따라 {resume['resumed_at']}에 동일 실행 경로에서 이어서 실행했다. "
                f"재개 점검 시 운영체제 부팅 시각은 {resume['boot_time']}였고 이전 실행 프로세스는 없었다. "
                '정확한 종료 원인은 기록되지 않아 단정하지 않는다. '
                '라이브러리 버전과 실험 코드·설정의 동일성을 확인했고, 완료 예측의 해시를 검증하여 재사용하며 미완료 조건은 처음부터 재실행한다. '
                '중단 상태와 미완료 예측 사본은 실행 폴더의 resumptions 아래에 보존한다.')
    if (run/'inventory.json').exists():
        inv=pd.DataFrame(read(run/'inventory.json'))
        parts.append('## 원래 분할의 클래스별 표본 수\n표본 수는 가용 전체 train pool과 고정 test의 행 수이며 실제 모델 학습 수 n과 구별한다. 정제 전후 분포 표의 클래스는 정제 후 test 표본 수 내림차순으로 정렬한다.')
        for ds in DS:
            g=inv[inv.dataset==ds].sort_values('clean_test',ascending=False)
            parts.append(f'### {DS[ds]}\n'+table(['클래스','정제 전 train','정제 후 train','정제 전 test','정제 후 test'],
                [[r.class_name,f'{r.original_train:,}',f'{r.clean_train:,}',f'{r.original_test:,}',f'{r.clean_test:,}'] for r in g.itertuples()]))
    return parts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    run=args.run.resolve();cfg=read(run/'config.json')
    date=cfg['started_at'][:10].replace('-','')
    destination=ROOT/'docs/lablog'/date/'exp89_report.md'
    parts=basic_sections(run,cfg)
    evidence=ROOT/'docs/lablog/evidence'/date
    evidence.mkdir(parents=True,exist_ok=True)
    pipeline=run/'report_pipeline.json'
    if args.partial:
        count=len(list((run/'jobs').glob('*/COMPLETE.json')))
        failure=read(run/'FAILURE.json') if (run/'FAILURE.json').exists() else None
        state='실패/중단' if failure else '실행 중'
        text=f'400개 조건 중 {count}개 완료를 {stamp()}에 확인했다. 현재 상태는 {state}이며 전체 결과와 독립 검산은 완료되지 않았다. 부분 평균을 최종 결과로 제시하지 않는다.'
        if failure:
            text+=f"\n\n중단 원인은 `{failure['error']}`다. 완료한 조건의 예측과 ID는 보존하며 원인 수정 후 동일 실행 경로의 검증된 완료 조건을 재사용하여 재개한다."
        parts.insert(1,'## 실행 상태\n'+text)
        parts.append(f'## 실행 인계\n실행 경로는 `{run.relative_to(ROOT)}`다. `progress.json`과 `stdout.log`에서 진행 상태를 확인한다. 남은 작업은 모든 모델 평가, 독립 입력·예측 검산, 표·그림 생성, 한국어 해석과 임시 PDF 컴파일이다. 관리 실행기는 완료 후 analyze_curation.py와 report_curation.py를 순서대로 호출하며 모델 또는 마감 단계의 실패는 FAILURE.json에 남긴다.')
        target=write_record(destination,'\n\n'.join(parts),summary=f'정제 전후·정상 n개 포함/제외, n=2/3/5/10/20, 5 seeds; {state} {count}/400조건; 전체 test 고정',compile_pdf=False)
        validation=build_report(target)
        write(evidence/'exp89_partial_validation.json',dict(at=stamp(),state=state,completed_jobs=count,**validation))
        write(pipeline,dict(at=stamp(),phase='partial_report_compiled',completed_jobs=count,report=str(target)))
        print(str(target),flush=True)
        return
    result=read(run/'results.json')
    validation=read(run/'analysis/validation.json')
    assert result['state']=='complete' and validation['passed'] and validation['jobs']==400
    frames={p.stem:pd.read_csv(p) for p in (run/'analysis').glob('*.csv')}
    # CSV booleans are read as bool by pandas; reject ambiguous strings.
    assert frames['metrics'].benign_training.dtype==bool
    paths=plots(run,frames,date)
    m=frames['metrics'];pc=frames['per_class'];delta=frames['paired_contrasts']
    parts.insert(1,f"## 실행 상태\n400개 모델 조건과 1,600개 평가 view의 집계를 완료했고 저장 예측·입력 ID·지표의 독립 검산을 통과했다. 모델 완료일은 {result['updated_at'][:10]}이며 검산 시각은 {validation['validated_at']}다. 아래 수치는 다섯 seed 평균과 표본 표준편차이며 모델별 우위를 특정 결과만으로 일반화하지 않는다.")
    observations=[]
    for ds in DS:
        for method in METHOD:
            original=subset(m,dataset=ds,method=method,n=20,test_version='original',test_scope='attack',train_version='original')
            a=original[~original.benign_training].attack_macro_f1.mean()*100
            b=original[original.benign_training].attack_macro_f1.mean()*100
            fixed=subset(m,dataset=ds,method=method,n=20,train_version='clean',benign_training=False,test_scope='attack')
            before=fixed[fixed.test_version=='original'].attack_macro_f1.mean()*100
            after=fixed[fixed.test_version=='clean'].attack_macro_f1.mean()*100
            observations.append(f'{DS[ds]} {METHOD[method]}의 n=20에서 정제 전 train·고정된 정제 전 공격 test를 사용할 때 정상 학습 추가 전후 공격 Macro-F1은 {a:.2f}% → {b:.2f}% ({b-a:+.2f} percentage point (pp))다. 정제 후 공격만 학습한 모델을 고정하고 test만 정제 전에서 후로 바꿀 때는 {before:.2f}% → {after:.2f}% ({after-before:+.2f} pp)다. 두 변화는 각각 정상 학습 조건과 평가 모집단 선택의 관찰이며 서로 다른 비교다.')
    parts.insert(2,'## 핵심 관찰\n'+'\n\n'.join(observations))
    parts.append('## 고정된 정제 전 공격 test의 학습곡선\n정제 전 공격 test 전체를 동일하게 유지해 네 학습 조건을 비교한다. 정상 포함 여부는 각 클래스당 n과 같은 수의 정상 학습 행을 추가하는 조건이다. 아래 표의 값은 공격 Macro-F1(%)이며 각 셀은 5개 seed 평균 ± 표본 표준편차다.')
    for ds in DS:
        rows=[]
        for version in ['original','clean']:
            for benign in [False,True]:
                for n in BUDGETS:
                    g=subset(m,dataset=ds,train_version=version,benign_training=benign,n=n,test_version='original',test_scope='attack')
                    x=g[g.method=='xgb'].sort_values('seed').attack_macro_f1.to_numpy()
                    p=g[g.method=='pfn'].sort_values('seed').attack_macro_f1.to_numpy()
                    rows.append([VERSION[version],'포함' if benign else '제외',n,ms(x),ms(p),ms(p-x)])
        parts.append(f'### {DS[ds]}\n'+table(['train 정제','정상 학습','n','XGB F1 (%)','TabPFN F1 (%)','PFN−XGB (pp)'],rows))
    parts.append(f'![정제 전 공격 test 고정: 정상 학습과 train 정제 비교. 음영은 seed 표준편차.](../figures/{date}_exp89_original_attack.pdf)')
    parts.append('## 정상 포함 학습 모델의 전체 test 평가\n정상 포함 모델의 공격 F1에는 정상에서 유입된 오분류가 반영된다. 아래는 정제 전 전체 test를 사용하며 공격 F1과 전체 클래스 F1을 별도 열로 구분한다. FPR은 낮을수록 정상 오탐이 적고 공격 탐지 recall은 공격을 정상으로 놓치지 않는 비율이다. 공격 종류를 잘못 분류한 경우는 탐지 recall에는 성공으로 남을 수 있으므로 공격 F1을 함께 본다.')
    for ds in DS:
        rows=[]
        for version in ['original','clean']:
            for n in BUDGETS:
                for method in METHOD:
                    g=subset(m,dataset=ds,train_version=version,benign_training=True,n=n,method=method,test_version='original',test_scope='all')
                    rows.append([VERSION[version],n,METHOD[method],ms(g.attack_macro_f1),ms(g.all_macro_f1),ms(g.benign_fpr),ms(g.attack_detection_recall)])
        parts.append(f'### {DS[ds]}\n'+table(['train 정제','n','모델','공격 F1 (%)','전체 F1 (%)','정상 FPR (%)','탐지 recall (%)'],rows))
    parts.append(f'![정제 전 전체 test: 정상 포함 학습 모델의 공격 F1과 정상 FPR.](../figures/{date}_exp89_original_full.pdf)')
    parts.append('## n=20의 원인 분리 대조\n각 차이는 동일 seed를 대응시켜 계산한다. 정상 학습 추가와 train 정제 비교는 정제 전 공격 test를 고정한다. test 정제 비교는 정제 후 train의 같은 모델을 유지하며, 정상 평가 행 추가는 정상 포함 학습 모델의 동일 예측을 사용한다. 차이의 단위는 pp이며 양수는 해당 변경 후 공격 Macro-F1 증가를 뜻한다.')
    rows=[]
    for ds in DS:
        for method in METHOD:
            for contrast,label,filters in [
                ('benign_training','정상 학습 추가',dict(train_version='original',test_version='original',test_scope='attack')),
                ('train_curation','train 정제',dict(benign_training=True,test_version='original',test_scope='attack')),
                ('test_curation','test 정제',dict(benign_training=True,train_version='clean',test_scope='attack')),
                ('benign_test_inclusion','정상 test 추가',dict(benign_training=True,train_version='original',test_version='original'))]:
                g=subset(delta,dataset=ds,method=method,n=20,contrast=contrast,measure='attack_macro_f1',**filters)
                rows.append([DS[ds],METHOD[method],label,ms(g.delta)])
    parts.append(table(['데이터','모델','변경','대응 차이 (pp)'],rows))
    parts.append(f'![같은 정제 후 학습 모델의 평가 모집단만 변경한 결과.](../figures/{date}_exp89_test_curation.pdf)')
    parts.append('## 클래스별 분류와 오분류\n클래스별 그림은 정제 전 test의 평가 표본 수 내림차순으로 정렬한다. 정상 미포함 test와 정상 포함 test의 공격 F1을 분리해 제시하며 색과 수치는 5개 seed 평균이다. 정상 test 그림은 정상 학습을 포함한 모델만 표시한다. 전체 클래스·예산의 precision, recall, TP, FP, FN은 per_class.csv에 보존한다.')
    for ds in DS:
        parts.append(f'![{DS[ds]}: 정제 전 공격 test의 클래스별 F1.](../figures/{date}_exp89_classes_{ds}_attack.pdf)')
        parts.append(f'![{DS[ds]}: 정제 전 전체 test의 공격 클래스별 F1. 정상 학습 포함.](../figures/{date}_exp89_classes_{ds}_all.pdf)')
    parts.append('### BF와 mitm의 n=20 정밀도·재현율\nBF와 mitm의 높은 점수 또는 낮은 F1을 누락과 다른 클래스의 오분류 유입으로 구분한다. 아래는 정제 전 test를 평가하며 같은 학습 모델의 공격만/정상 포함 집계를 나란히 제시한다. 정상 포함 집계는 정상 학습 조건에서만 주 비교로 사용한다.')
    for ds,name in [('cic2018','brute_force'),('toniot','mitm')]:
        rows=[]
        for version in ['original','clean']:
            for benign in [False,True]:
                for method in METHOD:
                    for scope in (['attack','all'] if benign else ['attack']):
                        g=subset(pc,dataset=ds,name=name,train_version=version,benign_training=benign,method=method,n=20,
                                 test_version='original',test_scope=scope)
                        rows.append([VERSION[version],'포함' if benign else '제외',METHOD[method],
                            '공격만' if scope=='attack' else '정상 포함',f'{g.precision.mean()*100:.2f}',
                            f'{g.recall.mean()*100:.2f}',f'{g.f1.mean()*100:.2f}'])
        parts.append(f'### {DS[ds]} / {name}\n'+table(['train 정제','정상 학습','모델','test','Precision (%)','Recall (%)','F1 (%)'],rows))
    parts.append('## 기존 결과 연속성과 구현 검증\n정제 후 공격 학습 ID는 EXP86과 정확히 일치하며 n=3은 그 중첩 표본의 새 측정점이다. 원래 전체 test를 복원해 query 배치 구성이 달라졌으므로 기존 예측을 그대로 재사용하지 않고 모든 조건을 같은 query 규칙으로 실행했다. 아래는 정제 후에 남는 test 행에서 EXP86 예측과의 차이이며 모델별 40조건을 요약한다.')
    continuity=frames['continuity']
    rows=[]
    for method in METHOD:
        g=continuity[continuity.method==method]
        rows.append([METHOD[method],len(g),int(g.retained_prediction_disagreements.max()),
                     f'{g.attack_macro_f1_delta.abs().max()*100:.6f}'])
    parts.append(table(['모델','대조 조건','최대 예측 불일치 행','최대 F1 절대 차이 (pp)'],rows))
    parts.append('독립 검산은 원본 특징·라벨과 test 전체의 일치, 원래 분할 유지, train/test 행 ID 분리, 정상 추가 시 공격 ID 보존, 모델 간 입력 일치, n별 중첩을 확인했다. 저장 예측의 SHA-256과 sklearn 혼동행렬로 400개 조건의 네 평가 view를 다시 계산했다. 공격을 정상으로 분류한 false negative를 제외하지 않았고 모든 클래스의 precision·recall·F1과 TP·FP·FN을 점검했다. TabPFN은 fit 후와 예측 후의 가중치 digest가 기존 체크포인트와 같은지 검사했다.')
    parts.append(f"실제 트리 분할 수가 0인 XGB 조건은 {len(validation['zero_split_jobs'])}개다. 해당 목록은 validation.json에 보존하며 이런 조건이 있으면 기존 고정 설정의 제약과 모델 차이를 구분해 해석한다.")
    parts.append('## 준비·추론 비용\n모델 준비는 새 모델 생성과 fit/cache 준비를 합한 값이고 가중치 갱신 또는 순차 추가 비용이 아니다. 추론은 원래 전체 test의 모든 행에 대한 비용이다. 정제 후 공격만 평가한 EXP87 비용과 직접 비교하지 않으며 파일 저장·독립 검산·보고서 생성 비용을 추론에 합산하지 않는다. 아래는 n=20의 5개 seed 평균이며 단위는 초다.')
    rows=[]
    for ds in DS:
        for version in ['original','clean']:
            for benign in [False,True]:
                for method in METHOD:
                    g=subset(frames['cost'],dataset=ds,train_version=version,benign_training=benign,method=method,n=20)
                    rows.append([DS[ds],VERSION[version],'포함' if benign else '제외',METHOD[method],
                                 f'{g.preparation_seconds.mean():.3f}',f'{g.inference_seconds.mean():.3f}'])
    parts.append(table(['데이터','train 정제','정상 학습','모델','준비 (초)','전체 test 추론 (초)'],rows))
    parts.append('''## 해석 범위와 다음 개선
정상 학습의 효과는 동일한 공격 test에서 정상 추가 전후의 대응 차이로 판단한다. 전체 test에서 공격 F1이 달라지면 정상에서 유입된 false positive를 별도로 확인한다. 정상 표본이 2~20개인 조건에서 FPR이 높게 남는 경우 정상의 대표성과 정상·공격 판정 기준을 개발 데이터에서 개선하는 후속 대조가 필요하다. 정상 포함 여부의 차이를 정상 데이터가 충분한 운영 조건의 성능으로 일반화하지 않는다.

test 정제에 따른 점수 변화는 같은 모델에서 평가 모집단을 바꾼 결과다. train 정제의 효과는 동일 원래 test에서 비교하므로 평가 행 삭제에 따른 차이와 분리된다. 두 효과가 크더라도 원래 라벨 충돌이 모두 라벨 오류인지 또는 46개 특징의 표현 손실인지는 이 실험만으로 구분되지 않는다. 그 구분에는 원래 flow·세션의 관찰 근거와 라벨 품질 검토가 필요하다.

TabPFN의 이점은 전체 n=2·3·5·10·20에서 동일 표본을 받는 XGB와의 대응 차이로 검토한다. 특정 한 예산이나 클래스의 가장 높은 수치만으로 보편적 우위를 주장하지 않는다. 동일 test의 반복 seed는 표본 선택 변동을 확인하며 host·session이 분리된 외부 일반화 검증은 남아 있다. 모델 우위가 조건에 따라 바뀌면 그 조건과 비용을 밝혀 적용 범위를 정하고, 본 실행의 완료만으로 원고의 성능 주장을 채택하지 않는다.

## 재현 자료
''')
    parts.extend(f'`{p.relative_to(ROOT)}`' for p in [run,ROOT/'configs/experiments/exp89_protocol.json',
                 ROOT/'scripts/v7/exp89/curation_benign.py',ROOT/'scripts/v7/exp89/analyze_curation.py',
                 ROOT/'scripts/v7/exp89/report_curation.py',run/'analysis'])
    parts.append('원시 예측·학습 ID·XGB 모델·체크포인트 해시·환경과 실행 소스는 결과 경로에 보존한다. 수치 CSV와 그림은 lablog의 data·figures에 연결하며 보고서 PDF는 임시 컴파일로만 검증한다. 영어 원고와 발표자료는 변경하지 않는다.')
    data=ROOT/'docs/lablog/data'
    data.mkdir(exist_ok=True)
    for p in (run/'analysis').glob('*.csv'):shutil.copy2(p,data/f'{date}_exp89_{p.name}')
    target=write_record(destination,'\n\n'.join(parts),summary='정제 전후·정상 n개 포함/제외, n=2/3/5/10/20, 5 seeds; 400/400조건·1,600 view 독립 검산 완료; 원래 전체 test에서 영향 분리',compile_pdf=False)
    compiled=build_report(target)
    structure=check()
    assert structure['passed'],structure
    write(evidence/'exp89_validation.json',dict(at=stamp(),run=str(run.relative_to(ROOT)),**compiled,
                                               structure=structure,independent_validation=str((run/'analysis/validation.json').relative_to(ROOT))))
    write(evidence/'exp89_sources.json',dict(run=str(run.relative_to(ROOT)),protocol='configs/experiments/exp89_protocol.json',
        report_sha256=sha(target),figures={str(p.relative_to(ROOT)):sha(p) for p in paths},
        conclusions_adoption='exploratory_not_manuscript_adopted',generated_at=stamp()))
    finalization=run/'source_dependencies/finalization'
    finalization.mkdir(parents=True,exist_ok=True)
    shutil.copy2(__file__,finalization/Path(__file__).name)
    write(pipeline,dict(at=stamp(),phase='complete',report=str(target),compiled=True,pdf_persisted=False,scientific_adoption='exploratory'))
    print(json.dumps(dict(report=str(target),compiled=compiled,structure=structure),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
