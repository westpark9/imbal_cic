#!/usr/bin/env python3
"""Verified Korean research report; evaluated candidates are not auto-adopted."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)
ROOT=_root
import argparse
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from lablog import write_record,build_report,check
from exp61_utils import sha
from run_shared_experts import read,write,stamp,score,conditions,full_condition
from shared_experts import METHODS

LABEL={'xgb':'XGB fixed','xgb_tuned':'XGB OOF','pfn':'PFN4','pfn16':'PFN16',
       'pfn_log':'Log PFN','mean_xgb_pfn':'XGB+PFN mean','mean_views':'View mean',
       'hier_pfn':'PFN hierarchy','hier_pfn_xgb':'PFN gate+XGB','hier_xgb':'XGB hierarchy',
       'mean_hierarchy':'Global+hierarchy','mean_bank':'Bank mean','convex_oof':'OOF convex',
       'gate_oof':'OOF gate','anchored_gate':'Anchored gate','pfn_bank_mean':'PFN bank mean'}
DS={'cic2018':'CIC2018','toniot':'ToN-IoT'}
KEY=['dataset','regime','n','seed','phase']
CANDIDATES=['mean_views','hier_pfn','mean_hierarchy','mean_bank','convex_oof','gate_oof','anchored_gate','pfn_bank_mean']


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def ms(x,scale=100):
    x=np.asarray(x,dtype=float)*scale
    return f'{x.mean():.2f} ± {x.std(ddof=1):.2f}' if len(x)>1 else f'{x.mean():.2f}'


def frame(run):
    metrics=[];classes=[];selections=[]
    for job in sorted((run/'jobs').iterdir()):
        if not job.is_dir():continue
        sampling=read(job/'sampling.json')
        if (job/'selection.json').exists():
            s=read(job/'selection.json')
            selections.append(dict(dataset=sampling['dataset'],regime=sampling['regime'],n=sampling['n'],
                seed=sampling['seed'],chosen_xgb=s['chosen_xgb'],**{f'weight_{i}':v for i,v in enumerate(s['convex_weights'])}))
        for phase in ['panel','full']:
            d=job/phase
            if not (d/'COMPLETE.json').exists():continue
            for item in read(d/'metrics.json'):
                basic={k:item[k] for k in KEY+['method','train_rows','evaluated_rows','inference_seconds',
                       'fit_seconds','oof_seconds','combiner_fit_seconds','preparation_seconds']}
                basic.update({k:v for k,v in item['normal_inclusive'].items() if k not in ['classes','confusion']})
                basic.update(attack_only_macro_f1=item['attack_only']['attack_macro_f1'],
                             helpful=item['corrections']['helpful'],harmful=item['corrections']['harmful'],
                             changed=item['corrections']['changed'])
                metrics.append(basic)
                for c in item['normal_inclusive']['classes']:
                    classes.append({**{k:item[k] for k in KEY+['method']},**c})
    return pd.DataFrame(metrics),pd.DataFrame(classes),pd.DataFrame(selections)


def validate(run,cfg,df):
    validations=[]
    for ds,r,n,s in conditions(cfg):
        job=run/'jobs'/f'{ds}_{r}_n{n}_s{s}'
        md=read(run/'data'/ds/'manifest.json');C=len(md['class_names']);source=Path(md['source_exp89'])
        x=np.load(job/'train_X.npy');y=np.load(job/'train_y.npy');ids=np.load(job/'train_ids.npy')
        group=np.load(job/'groups.npy');fold=np.load(job/'folds.npy')
        assert len(np.unique(ids))==len(ids)
        for g in np.unique(group):assert len(np.unique(fold[group==g]))==1
        assert not np.isin(ids,np.load(source/'test_ids.npy',mmap_mode='r')).any()
        state=read(job/'selection.json');assert state['test_access'] is False
        assert state['train_ids_sha256']==sha(job/'train_ids.npy')
        expected=(750 if r=='fixed750' else n)+(C-1)*n
        assert len(y)==expected and len(x)==expected
        for phase in ['panel']+(['full'] if full_condition(cfg,r,n) else []):
            d=job/phase;marker=read(d/'COMPLETE.json')
            assert marker['metrics_sha256']==sha(d/'metrics.json')
            assert marker['selection_sha256']==sha(job/'selection.json')
            actual=read(d/'metrics.json');assert {z['method'] for z in actual}==set(METHODS)
            truth=np.load(run/'data'/ds/'panel_y.npy') if phase=='panel' else np.load(source/'test_y.npy',mmap_mode='r')
            w=np.load(run/'data'/ds/'panel_weights.npy') if phase=='panel' else None
            for z in actual:
                p=d/f"pred_{z['method']}.npy";assert sha(p)==z['predictions_sha256']
                pred=np.load(p,mmap_mode='r');assert len(pred)==len(truth)
                cm=confusion_matrix(truth,pred,labels=np.arange(C),sample_weight=w)
                np.testing.assert_allclose(cm,z['normal_inclusive']['confusion'],rtol=1e-9,atol=1e-4)
                checked=score(cm,md['class_names'])
                for k in ['attack_macro_f1','all_macro_f1','benign_fpr']:
                    assert abs(checked[k]-z['normal_inclusive'][k])<1e-9
            validations.append(dict(condition=job.name,phase=phase,methods=len(actual),rows=len(truth)))
    expect=len(conditions(cfg))*len(METHODS)+cfg['design']['full_test']['expected_conditions']*len(METHODS)
    assert len(df)==expect,(len(df),expect)
    result=dict(passed=True,at=stamp(),evaluations=len(df),conditions=len(conditions(cfg)),details=validations,
        checks=['saved predictions and selection hashes','same label budget','train/test ID disjointness',
                'feature groups isolated across OOF folds','independent confusion matrices and aggregate metrics'],
        limitations=['outer split is not feature-group-disjoint','three seeds are not new independent data environments',
                     'panel estimates are not exact full-test metrics','OOF fitting has half/two-thirds the final context'])
    write(run/'analysis/validation.json',result)
    return result


def plots(df,date):
    dest=ROOT/'docs/lablog/figures';dest.mkdir(exist_ok=True);paths=[]
    selected=['xgb_tuned','pfn','pfn16','mean_views','hier_pfn','convex_oof','anchored_gate']
    palette=dict(zip(selected,['#777777','#24649d','#24374b','#008b75','#c58e22','#8f59a9','#c44040']))
    panel=df[df.phase=='panel']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,ds in enumerate(DS):
        for j,r in enumerate(['matched','fixed750']):
            ax=axes[i,j]
            for method in selected:
                s=panel[(panel.dataset==ds)&(panel.regime==r)&(panel.method==method)].groupby('n').attack_macro_f1
                mean=s.mean()*100;sd=s.std()*100
                if mean.empty:continue
                ax.plot(mean.index,mean,marker='o',label=LABEL[method],color=palette[method],lw=1.4)
                ax.fill_between(mean.index,mean-sd,mean+sd,color=palette[method],alpha=.08)
            ax.set(title=f'{DS[ds]} / benign '+('n' if r=='matched' else '750'),xlabel='Labels per attack class',
                   ylabel='Weighted panel attack Macro-F1 (%)',xticks=[2,3,5,10,20])
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    path=dest/f'{date}_exp90_panel.pdf';fig.savefig(path);plt.close(fig);paths.append(path)
    full=df[df.phase=='full']
    if not full.empty:
        fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
        for i,ds in enumerate(DS):
            base=full[(full.dataset==ds)&(full.method=='pfn')].set_index(['n','seed'])
            a=[]
            for method in METHODS:
                g=full[(full.dataset==ds)&(full.method==method)].set_index(['n','seed'])
                a.append([(g.loc[n].attack_macro_f1-base.loc[n].attack_macro_f1).mean()*100 for n in [5,20]])
            a=np.array(a);v=max(abs(a).max(),1)
            ax=axes[i];im=ax.imshow(a,cmap='RdBu',vmin=-v,vmax=v,aspect='auto')
            for row in range(len(METHODS)):
                for col in range(2):ax.text(col,row,f'{a[row,col]:+.2f}',ha='center',va='center',fontsize=8,
                                         color='white' if abs(a[row,col])>.65*v else 'black')
            ax.set(xticks=[0,1],xticklabels=[5,20],yticks=range(len(METHODS)),yticklabels=[LABEL[m] for m in METHODS],
                   title=f'{DS[ds]} / benign 750 / full test',xlabel='Labels per attack class')
            fig.colorbar(im,ax=ax,shrink=.7,label='Attack Macro-F1 difference from PFN4 (pp)')
        path=dest/f'{date}_exp90_full_gain.pdf';fig.savefig(path);plt.close(fig);paths.append(path)
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for ax,ds in zip(axes,DS):
            for method in selected:
                g=full[(full.dataset==ds)&(full.method==method)].groupby('n')[['attack_macro_f1','benign_fpr']].mean()*100
                ax.plot(g.benign_fpr,g.attack_macro_f1,'o-',label=LABEL[method],color=palette[method])
                for n,v in g.iterrows():ax.annotate(str(n),(v.benign_fpr,v.attack_macro_f1),fontsize=7)
            ax.set(title=f'{DS[ds]} / full test',xlabel='Benign false positive rate (%)',ylabel='Attack Macro-F1 (%)')
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        path=dest/f'{date}_exp90_false_alarm.pdf';fig.savefig(path);plt.close(fig);paths.append(path)
    return paths


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--partial',action='store_true')
    args=ap.parse_args();run=args.run.resolve();cfg=read(run/'config.json');date=cfg['started_at'][:10].replace('-','')
    df,pc,selection=frame(run);analysis=run/'analysis';analysis.mkdir(exist_ok=True)
    df.to_csv(analysis/'metrics.csv',index=False);pc.to_csv(analysis/'per_class.csv',index=False);selection.to_csv(analysis/'selection.csv',index=False)
    final=not args.partial
    validation=validate(run,cfg,df) if final else dict(passed=False,status='running_or_partial',at=stamp())
    parts=[f'''# EXP90: 소수 라벨을 공유하는 TabPFN 전문가 구조 탐색

## 핵심 질문과 수행 상태
TabPFN을 포함한 전문가 구조가 동일한 라벨 예산의 단독 모델과 단순 평균을 넘어 공격 분류를 개선하는지가 본 실험의 핵심 질문이다. 기존 EXP89의 TabPFN·XGB 비교만으로 제안 구조의 효과를 입증할 수 없다는 사용자 지적을 출발점으로 삼는다. 연구 버전은 승인된 v7을 유지하고 새 모델 후보를 EXP90에서 탐색한다. 실행 시작은 {cfg['started_at']}이며 문서 작성 시각은 {stamp()}이다.

실행 ID는 {run.name}이다. 현재 상태는 {'전체 조건 실행 및 독립 검증 완료' if final else '진행 중 또는 부분 완료이며 아래 완료 조건만 해석 가능'}이다. 구성과 계수는 각 조건의 test 예측 전에 고정하고 성능을 본 뒤 후보를 수정하지 않는다. 이 기록의 비교와 후보 순위는 탐색 결과이며 영어 논문이나 발표자료에 자동 채택하지 않는다.

## 기존 구조의 병목과 새 가설
기존 TAILGUARD 구현은 라벨을 Global·expert·route·tune·cal 역할로 나눴다. 이전 EXP82의 ToN 점검에서는 695개 Global의 52.89%와 전체 1,700개 PFN의 66.61%가 비교됐고, 당시 제안 출력은 Global과 같았다. 이 수치는 정제 후 데이터와 당시 표본 구성의 결과여서 EXP90 점수와 직접 차감하지 않는다. 이번 설계는 라벨 분할로 Global의 정보를 줄이지 않고 모든 확보 사례로 Global을 다시 맞춘다.

전문가는 같은 라벨을 재사용하면서 입력 표현 또는 분류 과제를 바꾼다. 입력을 달리한 전문가가 상보적인 오류를 내는지, 정상/공격과 공격 종류를 분리하면 정상 오탐을 줄이는지, 선택기가 실제 수정 이익을 남기는지를 각각 검증한다. 기존 residual 군집·S/V 구조를 그대로 재현한 실험은 아니며 제안 모델의 교체 가능한 구조 후보를 평가한다. 기존 제안 대비 개선을 확정하려면 향후 동일 표본·동일 예산에서 직접 재실행해야 한다.

## 문헌과 설계의 연결
BoostPFN은 PFN context를 약한 학습기로 보고 잔차에 따라 후속 context의 추출 가중치를 바꿨다. 이번에는 큰 pool에서 잔차 표본을 추가하지 않고 이미 받은 소수 라벨을 공유한다. 따라서 BoostPFN 재현이나 신규 부스팅 알고리즘으로 부르지 않는다. 단순 평균보다 학습된 결합이 나은지를 별도 비교하는 설계 원리만 참고했다.
https://arxiv.org/abs/2503.01256

LoCalPFN은 근접 사례 검색과 가중치 fine-tuning을 결합했다. 이번에는 매 query마다 context를 검색하거나 backbone을 fine-tuning하지 않고 고정된 전문가들을 재사용한다. 검색·학습을 생략한 동등한 LoCalPFN 구현이라는 의미는 아니다. 속도 이점도 측정 전에는 주장하지 않는다.
https://arxiv.org/abs/2406.05207

TabFM+는 여러 입력 표현과 앙상블·보정을 결합했다. 이번 log view는 그 방향을 참고한 자체 실험이며 TabFM의 모델이나 32-view 절차를 재현하지 않는다. DistPFN은 학습 context prior와 test posterior의 차이를 보정했다. 이번 실험은 실제 test prior를 선택기에 전달하지 않으며 batch posterior 기반 보정도 수행하지 않는다.
https://arxiv.org/abs/2609.37959
https://arxiv.org/abs/2605.04363

## 데이터·라벨 예산·평가 범위
원래 outer split의 CIC2018·ToN-IoT를 사용하고 전역 충돌 정제를 적용하지 않는다. EXP89와 같은 46개 float32 특징, family 라벨, nan-to-num 처리와 공격 학습 행 ID를 유지한다. 여기서 정제 전은 충돌 행 제거만 끈 버전이며 PCAP나 원래 CICFlowMeter 특징 버전으로 바꿨다는 뜻이 아니다. 같은 벡터의 다른 라벨도 이번 train과 test에 남아 있다.

공격 클래스당 n은 2·3·5·10·20이고 seed는 43·44·45다. 정상 n개 조건은 EXP89와 동일하며 정상 750개 조건은 기존 정상 20개에 같은 train pool의 730개를 추가하고 n에 따라 바꾸지 않는다. 같은 조건의 모든 모델은 동일한 행 ID와 라벨 예산을 받으며 별도 검증 라벨을 추가하지 않는다. 신규 공격만 적고 기존 공격 125개를 고정했던 순차 실험과는 다른 최종 다중분류 시나리오다.

선택기 학습에는 정확히 같은 특징 벡터를 하나의 group으로 묶은 out-of-fold (OOF) 예측을 사용한다. n=2·3은 2-fold, 그 외는 3-fold이며 fold 경계의 동일 벡터 누출을 막는다. 특정 클래스가 하나의 group에만 있으면 해당 fold 학습에서 클래스가 사라질 수 있고 그 클래스 확률을 0으로 정렬한다. 이 경우를 조용히 제거하거나 추가 라벨로 보완하지 않고 아래에 기록한다.

고정 패널은 정상 최대 100,000개와 공격 클래스당 최대 20,000개를 무작위 추출하고 같은 데이터셋의 모든 조건에 공유한다. 각 클래스의 전체 test 수/패널 수를 가중치로 삼아 혼동행렬을 추정하지만 패널 F1은 전체 test의 정확한 값이 아니다. 전체 평가는 사전에 정한 정상 750개·n=5/20·3 seed·2 데이터셋의 12조건에 대해 수행하며 패널 성능에 따라 대상 조건을 고르지 않는다. 전체 평가는 같은 split의 정밀한 계산이고 새로운 독립 test 검증은 아니다.

## 후보 구조와 학습 규칙
PFN4 Global은 확보한 모든 학습 행을 context로 쓰며 backbone 가중치를 갱신하지 않는다. Log PFN은 같은 행에서 부호를 보존한 log1p 입력을 사용하되 protocol, flag, TTL 및 일부 유형 코드 필드는 변환하지 않는다. 정상/공격 PFN gate는 전체 라벨을 이진으로 바꾸고 공격 PFN specialist는 정상 행을 뺀 공격 종류만 학습한다. 계층형 출력은 정상 확률과 공격 확률×공격 종류의 조건부 확률로 결합한다.

학습형 결합 bank는 OOF-selected XGB, PFN4, Log PFN, PFN hierarchy의 네 확률이다. OOF convex는 클래스 균형 negative log likelihood (NLL)와 균등 가중치 기준 L2=0.01로 비음수 합 1 가중치를 맞춘다. OOF gate는 각 전문가의 confidence·entropy·margin·정상 확률·Global과의 불일치 및 Global 예측 클래스로 query별 softmax 가중치를 학습하고 L2=0.1을 적용한다. Anchored gate는 OOF gate와 전체 Global을 50:50으로 섞어 Global을 명시적으로 유지한다.

XGB OOF는 depth 8/3과 min-child-weight 1/0.1의 네 설정 중 클래스 균형 OOF NLL로 선택하고 나머지는 300 trees, learning rate 0.05, subsample/colsample 0.8을 유지한다. XGB fixed는 기존 depth 8, min-child-weight 1 설정이다. PFN4는 기존 v3 checkpoint와 4 estimators, CPU context cache, fit-with-cache를 사용하며 subsampling하지 않는다. PFN16은 내부 estimator 수만 16으로 늘린 연산량 비교 기준이며 실제 context 크기가 다른 외부 bank와 비용이 완전히 같다고 가정하지 않는다.

모든 결합 계수는 OOF 예측에서만 학습하고 test 라벨·실제 test prior는 입력하지 않는다. 현재 gate는 이미 계산한 네 전문가 확률을 사용하므로 expert를 부르기 전 선별하는 비용 절감 구조가 아니다. test-time 전문가 계산을 모두 지불하며 sparse routing이나 S/V 속도 개선을 주장하지 않는다. 가중치 fine-tuning 없이 바뀌는 것은 context, 입력 표현, XGB 및 결합 계수다.
''']
    rows=[]
    missing=[]
    for ds in cfg['design']['datasets']:
        md=read(run/'data'/ds/'manifest.json');C=len(md['class_names'])
        rows.append([DS[ds],C-1,f"{md['original_test_rows']:,}",f"{md['panel_rows']:,}",
                     f'{C}n',f'750+{C-1}n'])
    parts.append(table(['데이터','공격 수','전체 test','패널','정상 n 학습 수','정상 750 학습 수'],rows))
    for job in sorted((run/'jobs').iterdir()):
        s=read(job/'sampling.json')
        absent=sum(bool(x['absent_train_classes']) for x in s['fold_audit'])
        if absent:missing.append([s['dataset'],s['regime'],s['n'],s['seed'],s['unique_feature_groups'],absent])
    parts.append(f'클래스가 빠진 OOF 학습 fold가 있는 조건은 {len(missing)}개다. 이는 fold 예측의 신뢰도를 제한하며 최종 Global에는 모든 클래스가 포함된다.')
    if missing:parts.append(table(['데이터','정상 조건','n','seed','고유 group','클래스 누락 fold'],missing))
    parts.append('## 지표와 비용의 의미\n공격 Macro-F1은 정상 포함 평가에서 각 공격 클래스 F1의 평균이며 정상에서 유입된 false positive를 포함한다. 정상 false positive rate (FPR)와 공격 탐지 recall을 함께 제시한다. 정상 정답 행을 제외한 공격-only 지표는 같은 예측에서 계산한 보조 진단이며 정상 출력 열을 지우거나 확률을 재정규화하지 않는다. 평균과 표준편차는 고정 split에서 세 표본 seed의 변동이다.\n\nOOF 준비, 최종 fit/cache 준비, 구성요소별 추론 비용을 분리한다. 학습형 결합에는 필요한 모든 OOF 학습과 결합 최적화 비용을 더한다. 추론 시간 표는 필요한 모든 구성요소의 합이며 여러 방법을 한꺼번에 계산한 공통 결합 처리 시간은 별도 원시 기록에 남긴다. sparse 호출이 없는 현재 구현의 비용을 Global 단독 비용으로 표시하지 않는다.')
    if not df.empty:
        paths=plots(df,date)
        for p in paths:parts.append('!['+p.stem+']('+str(p)+')')
        for phase in ['full','panel']:
            z=df[df.phase==phase]
            if z.empty:continue
            parts.append('## '+('전체 test 결과: 사전 지정 12조건' if phase=='full' else '가중 패널 결과: 60조건 탐색'))
            if phase=='panel':parts.append('아래 수치는 층화 표본의 가중 혼동행렬에서 계산한 추정치다. 전체 test 결과와 섞어 평균하지 않는다.')
            for (ds,r),g in z.groupby(['dataset','regime'],sort=False):
                ns=sorted(g.n.unique());rows=[]
                for method in METHODS:
                    h=g[g.method==method]
                    rows.append([LABEL[method]]+[ms(h[h.n==n].attack_macro_f1) for n in ns])
                parts.append(f'### {DS[ds]} · 정상 '+('n개' if r=='matched' else '750개')+' · 공격 Macro-F1 (%)')
                parts.append(table(['방법']+[f'n={n}' for n in ns],rows))
        full=df[df.phase=='full']
        if not full.empty:
            parts.append('## 후보별 추가 이익과 정상 오탐\n동일 seed·같은 test에서 PFN4를 뺀 차이를 계산한다. 평균 1위 후보를 사후에 골라 제안 모델로 확정하지 않으며 모든 후보의 양·음의 차이를 보존한다. 복잡한 gate는 단순 View mean, Bank mean 및 PFN16보다도 나아야 구조를 추가할 이유가 생긴다.')
            base=full[full.method=='pfn'].set_index(KEY)
            comparisons=[]
            for method in CANDIDATES:
                g=full[full.method==method].set_index(KEY)
                d=g.attack_macro_f1-base.attack_macro_f1;fp=g.benign_fpr-base.benign_fpr
                block=d.groupby(['dataset','n']).mean()
                comparisons.append(dict(method=method,mean_delta=d.mean(),std_delta=d.std(),wins=int((d>0).sum()),
                    cases=len(d),worst_block=block.min(),best_block=block.max(),mean_fpr_delta=fp.mean()))
            comp=pd.DataFrame(comparisons);comp.to_csv(analysis/'paired_candidates.csv',index=False)
            parts.append(table(['후보','평균 ΔF1 (pp)','개선 seed 조건','최저 집단 Δ','최고 집단 Δ','평균 ΔFPR (pp)'],
                [[LABEL[r.method],f'{r.mean_delta*100:+.2f}',f'{r.wins}/{r.cases}',f'{r.worst_block*100:+.2f}',
                  f'{r.best_block*100:+.2f}',f'{r.mean_fpr_delta*100:+.2f}'] for r in comp.itertuples()]))
            promising=comp[(comp.mean_delta>0)&(comp.worst_block>=0)&(comp.mean_fpr_delta<=0)]
            if promising.empty:
                parts.append('사전 지정 전체 조건에서 평균 F1 개선·모든 집단 비감소·평균 FPR 비증가를 동시에 충족한 후보는 없다. 구성요소별 이익을 확인한 뒤 손상 클래스와 정상 오탐을 제한하는 후속 검증이 필요하다. 단일 조건의 최고점을 일반적인 구조 우위로 확대하지 않는다.')
            else:
                parts.append('평균 F1 개선·모든 집단 비감소·평균 FPR 비증가를 함께 보인 탐색 후보는 '+', '.join(LABEL[m] for m in promising.method)+'이다. 이 기준은 유의성 검정이나 독립 데이터 검증을 대신하지 않는다. 단순 평균과 PFN16 비교까지 포함해 후속 실험의 우선순위를 정해야 한다.')
            parts.append('## 정상 오탐과 공격 탐지 recall: 전체 test')
            for (ds,n),g in full.groupby(['dataset','n']):
                parts.append(f'### {DS[ds]} · n={n}\n'+table(['방법','FPR (%)','공격 탐지 recall (%)','공격-only F1 (%)'],
                    [[LABEL[m],ms(g[g.method==m].benign_fpr),ms(g[g.method==m].attack_detection_recall),
                      ms(g[g.method==m].attack_only_macro_f1)] for m in METHODS]))
            parts.append('## 클래스별 결과와 수정·손상\nF1 증가가 특정 쉬운 공격에만 집중됐는지와 희귀 공격 precision 개선 여부를 확인한다. 클래스별 수치는 정상 포함 전체 test의 평균이며 클래스 순서는 test 표본 수 내림차순이다. 전체 PFN의 잘못을 맞춘 횟수와 맞던 예측을 망친 횟수는 별도 CSV에 보존한다. 이 횟수는 표본 수가 큰 클래스에 지배될 수 있어 Macro-F1과 동일한 이익 척도가 아니다.')
            display=['xgb_tuned','pfn','pfn16','mean_views','hier_pfn','convex_oof','anchored_gate']
            for (ds,n),g in pc[pc.phase=='full'].groupby(['dataset','n']):
                order=g.groupby('name').support.mean().sort_values(ascending=False).index
                rows=[]
                for name in order:
                    h=g[g.name==name]
                    rows.append([name,f'{h.support.iloc[0]:,.0f}']+[ms(h[h.method==m].f1) for m in display])
                parts.append(f'### {DS[ds]} · n={n} · 클래스 F1 (%)\n'+table(['클래스','test 수']+[LABEL[m] for m in display],rows))
            parts.append('## 준비·추론 비용: 전체 test\n아래 시간은 같은 실행 장치에서 측정한 초 단위 평균이다. 초기 데이터 추출과 보고서·독립 검증 시간은 포함하지 않는다. 준비는 필요한 OOF 학습과 최종 fit/cache를 포함하며 단순 Global보다 학습형 결합의 준비가 비쌀 수 있다. 추론은 모델 구성요소의 합으로 결합 처리 및 파일 저장 비용을 제외한 값이다.')
            for (ds,n),g in full.groupby(['dataset','n']):
                parts.append(f'### {DS[ds]} · n={n}\n'+table(['방법','OOF (s)','최종 fit (s)','총 준비 (s)','구성요소 추론 (s)'],
                    [[LABEL[m]]+[ms(g[g.method==m][k],1) for k in ['oof_seconds','fit_seconds','preparation_seconds','inference_seconds']] for m in METHODS]))
    parts.append('''## 해석의 범위와 다음 개선안
Global의 라벨을 유지하는 설계는 기존 분할 구조의 정보 손실을 제거하지만 분류 성능 향상을 보장하지 않는다. 계층형이 개선되면 정상 gate와 공격 종류 specialist의 분리 효과를 all-XGB 및 PFN gate+XGB 대조로 확인해야 한다. View mean이 학습형 gate와 비슷하거나 더 좋으면 선택기를 복잡하게 만드는 대신 입력 표현의 상보성을 우선 검증해야 한다. PFN16이 같은 수준이면 단순 estimator 증가 대비 새 구조의 추가 가치를 다시 평가해야 한다.

OOF 선택기가 개선되지 않으면 극소수 공격의 검증 예측이 불안정한 것이 한 설명 후보지만 결과만으로 원인을 확정하지 않는다. 다음 단계는 기존 충분한 정상·공격으로 선택기를 먼저 개발하고 신규 공격 라벨만 소량 제공하는 분리 실험, 정상 오탐 제한을 명시한 선택 목적, 동일 벡터 그룹 기준 외부 split 및 별도 IDS 데이터 검증이다. 이런 후속 설계에서 추가 개발 라벨을 공짜로 취급하지 않고 별도로 계산해야 한다. 현재의 query별 결합은 모든 전문가 결과가 필요하므로, 성능 이익이 남는 경우에만 사전 호출기를 증류해 실제 비용 절감을 검증한다.

원래 test는 여러 선행 실험에서 이미 관찰한 벤치마크이고 원본 행 ID만 train과 분리돼 있다. 동일 벡터가 outer train/test에 남아 있을 수 있으며 group OOF만으로 외부 누출 가능성이 제거되지 않는다. 동일 특징의 다른 라벨은 이 특징만 보는 결정적 모델로 완벽하게 동시에 분류할 수 없으므로, 어려운 충돌 사례를 삭제한 점수를 일반 운영 성능으로 해석하지 않는다. 세 seed의 평균 차이는 실제 수집환경 일반화나 통계적 유의성을 입증하지 않는다.

이 결과로 영어 원고의 TAILGUARD 구조·제목·주장을 바꾸지 않는다. 다음 제안은 후보의 실제 이익, 정상 오탐, 희귀 공격 precision, 단순 대조군 및 계산량을 함께 본 뒤 구체화한다. 모델 실행 성공과 논문 근거 채택은 구별한다.
''')
    if (run/'FAILURE.json').exists() and args.partial:parts.append('## 중단 원인과 복구\n'+read(run/'FAILURE.json')['error']+'\n완료된 구성요소 확률과 OOF·선택기 파일을 보존했다. 원인 수정 후 소스 변경 이력을 기록하고 완료 파일 해시를 검사한 뒤 재개해야 한다.')
    parts.append('## 재현 근거\n'+table(['항목','경로'],[[k,str(v.relative_to(ROOT))] for k,v in [
        ('프로토콜',ROOT/'configs/experiments/exp90_protocol.json'),('실행',run),
        ('전체 수치',analysis/'metrics.csv'),('클래스 수치',analysis/'per_class.csv'),
        ('선택기 계수',analysis/'selection.csv'),('검증',analysis/'validation.json')]]))
    if not final:write(analysis/'validation.json',validation)
    summary=('라벨 공유 전문가 60 패널·12 전체 조건/16방법 실행·검증 완료; 단독·평균·계층·OOF gate·PFN16 대조' if final else
             f'라벨 공유 전문가 실험 진행/부분 완료; 현재 {len(df)}개 방법 평가 기록')
    target=write_record(ROOT/'docs/lablog'/date/'exp90_report.md','\n\n'.join(parts),summary=summary,compile_pdf=False)
    compiled=build_report(target)
    structure=check();assert structure['passed'],structure
    evidence=ROOT/'docs/lablog/evidence'/date;evidence.mkdir(exist_ok=True)
    write(evidence/'exp90_validation.json',dict(validation=validation,compile=compiled,structure=structure))
    write(run/'report_pipeline.json',dict(at=stamp(),status='complete' if final else 'partial',report=str(target),
        compile=compiled,validation=validation.get('passed',False)))
    print(json.dumps(dict(report=str(target),status='complete' if final else 'partial',evaluations=len(df)),ensure_ascii=False))


if __name__=='__main__':main()
