"""Korean standalone report and exportable plots for validated EXP88 results."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').exists())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import repo_root
from lablog import write_record,build_report,check
from exp61_utils import write,sha
ROOT=repo_root(__file__)

import argparse,json,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS=['row_random','unique_random','cluster20','cluster40']
LABEL={'row_random':'행 무작위','unique_random':'고유벡터 무작위','cluster20':'군집 20 대표','cluster40':'군집 40 대표'}
EN={'row_random':'Random rows','unique_random':'Random unique vectors','cluster20':'20-cluster representatives','cluster40':'40-cluster sensitivity'}
COLORS={'row_random':'#2864A0','unique_random':'#B97214','cluster20':'#16866C','cluster40':'#A55E98'}
DS={'cic2018':'CIC2018','toniot':'ToN-IoT'}
MODEL={'xgb':'XGB','pfn':'TabPFN'}
BUDGETS=[2,5,10,20]

def read(path):return json.loads(Path(path).read_text())

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']+['---:']*(len(headers)-1))+' |']+
                      ['| '+' | '.join(map(str,r))+' |' for r in rows])

def mean_sd(values,scale=100):
    a=np.asarray(values)*scale
    return f'{a.mean():.2f} ± {a.std(ddof=1):.2f}'

def figures(run,frames,date):
    dest=ROOT/'docs/lablog/figures';dest.mkdir(exist_ok=True)
    main=frames['metrics'];coverage=frames['coverage_class_macro'];classes=frames['per_class']
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'savefig.dpi':160})
    paths=[]
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for i,ds in enumerate(DS):
        for k,method in enumerate(MODEL):
            ax=axes[i,k]
            for arm in ARMS:
                g=main[(main.dataset==ds)&(main.method==method)&(main.arm==arm)].groupby('n').macro_f1
                mean=g.mean().reindex(BUDGETS)*100;sd=g.std().reindex(BUDGETS)*100
                ax.plot(range(4),mean,marker='o',color=COLORS[arm],ls='--' if arm=='cluster40' else '-',label=EN[arm])
                if arm!='cluster40':ax.fill_between(range(4),mean-sd,mean+sd,color=COLORS[arm],alpha=.09)
            ax.set(title=f'{DS[ds]} / {MODEL[method]}',xticks=range(4),xticklabels=BUDGETS,
                   xlabel='Training rows per attack class',ylabel='Attack Macro-F1 (%)',ylim=(0,100))
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,loc='lower right')
    p=dest/f'{date}_exp88_learning.pdf';fig.savefig(p);paths.append(p);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    for i,ds in enumerate(DS):
        for j,(measure,label) in enumerate([('unique_vectors','Distinct selected vectors'),
                ('distance_median','Mean class median distance'),('distance_p90','Mean class 90th-percentile distance')]):
            ax=axes[i,j]
            for arm in ARMS:
                g=coverage[(coverage.dataset==ds)&(coverage.arm==arm)].groupby('n')[measure]
                mean=g.mean().reindex(BUDGETS);sd=g.std().reindex(BUDGETS)
                ax.plot(range(4),mean,marker='o',color=COLORS[arm],ls='--' if arm=='cluster40' else '-',label=EN[arm])
                ax.fill_between(range(4),mean-sd,mean+sd,color=COLORS[arm],alpha=.08)
            ax.set(title=f'{DS[ds]} / {label}',xticks=range(4),xticklabels=BUDGETS,
                   xlabel='Training rows per attack class');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    p=dest/f'{date}_exp88_coverage.pdf';fig.savefig(p);paths.append(p);plt.close(fig)
    for ds in DS:
        meta=read(run/'data'/ds/'manifest.json')
        names=sorted([n for n in meta['class_names'] if n!='benign'],
                     key=lambda n:meta['test_counts'][meta['class_names'].index(n)],reverse=True)
        fig,axes=plt.subplots(2,4,figsize=(13,6.5 if ds=='cic2018' else 8),layout='constrained')
        for i,method in enumerate(MODEL):
            for j,arm in enumerate(ARMS):
                ax=axes[i,j];selected=classes[(classes.dataset==ds)&(classes.method==method)&(classes.arm==arm)]
                a=selected.pivot_table(index='name',columns='n',values='f1').reindex(index=names,columns=BUDGETS).to_numpy()*100
                im=ax.imshow(a,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
                for y in range(len(names)):
                    for x in range(4):ax.text(x,y,f'{a[y,x]:.1f}',ha='center',va='center',fontsize=7,color='white' if a[y,x]>65 else 'black')
                ax.set(xticks=range(4),xticklabels=BUDGETS,yticks=range(len(names)),
                       yticklabels=names if j==0 else [],title=f'{MODEL[method]} / {EN[arm]}')
                if i==1:ax.set_xlabel('Rows per class')
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.75,label='Class F1 (%)')
        p=dest/f'{date}_exp88_classes_{ds}.pdf';fig.savefig(p);paths.append(p);plt.close(fig)
    return paths

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    run=args.run.resolve();analysis=run/'analysis';validation=read(analysis/'validation.json');assert validation['passed']
    cfg=read(run/'config.json');result=read(run/'results.json');assert result['state']=='complete'
    date=cfg['started_at'][:10].replace('-','');today=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d')
    evidence=ROOT/'docs/lablog/evidence'/date;evidence.mkdir(parents=True,exist_ok=True)
    frames={p.stem:pd.read_csv(p) for p in analysis.glob('*.csv')}
    paths=figures(run,frames,date)
    data=ROOT/'docs/lablog/data';data.mkdir(exist_ok=True)
    for p in analysis.glob('*.csv'):shutil.copy2(p,data/f'{date}_exp88_{p.name}')
    main=frames['metrics'];cover=frames['coverage_class_macro'];classes=frames['per_class']
    paired=frames['model_paired_deltas'];deltas=frames['selection_paired_deltas']
    parts=[]
    parts.append(f'''# EXP88: 극소수 표본의 대표성과 CIC2018 성능 정체 진단

## 목적과 실험 범위
표본의 대표성이 CIC2018의 극소수 학습 성능을 제한하는지 확인하는 것이 본 실험의 목적이다. 클래스당 2·5·10·20개에서 행 무작위, 고유벡터 무작위, 군집 대표 선택을 비교한다. 충분히 다양한 사례를 추가했는데도 F1이 변하지 않는 경우와, 선택 방식만 바꿔도 F1이 개선되는 경우를 구별한다. 이 실험은 가설 a의 진단이며, 큰 표본에서도 개선되지 않는 원인인 가설 b나 수집환경 단서에 관한 가설 c를 직접 검증하지 않는다.

실행 시작일은 {cfg['started_at'][:10]}이고 완료일은 {result['updated_at'][:10]}이며 보고서 작성일은 {today}이다. 승인된 연구 버전 v7에서 EXP88로 등록했다. {run.name} 실행은 총 320개 모델 조건을 완료했고 독립적인 지표 재계산을 통과했다. 실행 완료와 논문 근거 채택은 구별하며 현재 결과는 탐색적 진단이다.

## 데이터와 모델 설정
정제된 기존 분할과 46개 flow 특징을 그대로 사용한다. 서로 같은 특징 벡터에 다른 공격 family 라벨이 붙은 행은 기존 전역 충돌 제거 규칙에 따라 제외되고, 같은 라벨의 중복은 유지된다. 학습과 평가에서 정상 트래픽은 제외하므로 이 결과는 공격 유형 분류이며 정상/공격 탐지 성능이 아니다. 기존 scenario별 시간 분할은 유지하지만 하나의 전역 시간 경계 또는 독립 수집환경 분할로 해석하지 않는다.

두 모델은 모든 조건에서 정확히 같은 학습 행과 원래 46개 특징을 받는다. XGBoost (XGB)는 기존 설정인 300 trees, depth 8, learning rate 0.05, subsample/colsample 0.8을 사용한다. TabPFN은 기존 v3 체크포인트, 4 estimators, fit_with_cache, CPU cache를 사용하며 가중치를 갱신하지 않는다. 모델 설명에 사용하는 체크포인트와 구현 버전은 실행의 config.json, checkpoint.json, environment.json에 보존한다. seed는 43·44·45·46·47이며 각 seed 안의 2·5·10·20개는 중첩 표본이다.

평가는 기존 EXP87과 같은 전체 공격 test인 CIC2018 390,270행, ToN-IoT 797,421행에서 수행한다. query 순서와 batch 8,192도 동일하게 유지한다. 새로운 참조 표본을 train에서 분리하고 표본 추출 seed 체계를 고정했으므로 EXP87의 학습 행을 그대로 재사용하지 않는다. 본 실험의 선택 방식 간 비교를 주 근거로 사용하고 EXP87의 80·320개 결과는 큰 예산의 참고 결과로만 해석한다.

## 후보 집합과 독립 참조 표본
참조 표본은 클래스별 원래 clean train 행의 10%에서 최대 2,000개를 고정 난수로 선택한다. 참조 ID는 모든 선택 후보, 스케일러 적합, 군집 적합에서 제외한다. 참조와 후보에 같은 특징 벡터의 다른 행이 남을 수 있으므로 행 ID 분리와 특징 벡터 분리를 구별하며, 거리 0의 비율과 참조 고유벡터 가중 진단도 보존한다. 표의 클래스 순서는 해당 데이터셋의 고정 test 표본 수 내림차순이다.
''')
    for ds in DS:
        inv=frames['inventory'];g=inv[inv.dataset==ds].sort_values('test_rows',ascending=False)
        parts.append(f'### {DS[ds]} 표본 수\n'+table(['클래스','원래 train','선택 후보','후보 고유벡터','참조','test'],
            [[r.class_name,f'{r.original_train_rows:,}',f'{r.candidate_rows:,}',f'{r.candidate_unique_vectors:,}',f'{r.reference_rows:,}',f'{r.test_rows:,}'] for r in g.itertuples()]))
    parts.append('''## 선택 방식과 거리의 정의
행 무작위는 후보 행에서 균등하게 비복원 추출한다. 고유벡터 무작위는 46개 float32 값이 모두 같은 행을 하나로 묶고 서로 다른 벡터를 균등하게 비복원 추출한다. 따라서 고유벡터 무작위는 중복 제거와 함께 원래 행 빈도에 따른 가중도 바꾼다. 이 비교의 차이를 중복 효과만으로 단정하지 않는다.

군집 대표 선택은 클래스별 특징 공간을 20개 군집으로 나누고 각 군집 중심에서 가장 가까운 실제 후보 행을 선택한다. 첫 대표는 seed로 정하고, 이후에는 이미 뽑은 대표와의 최소 거리가 가장 큰 대표를 하나씩 추가한다. 대표 20개의 앞 2·5·10·20개를 사용하며 40개 군집에도 같은 규칙을 적용해 민감도를 확인한다. 이 순서는 드문 외곽 영역을 먼저 고를 수 있으므로 작은 예산에서 무작위보다 밀집 영역을 더 잘 대표한다고 보장하지 않는다.

군집 수는 실제 공격 패턴의 정답 개수가 아니다. 군집 적합 seed는 클래스와 군집 수별로 고정하고 모델 seed 및 첫 대표 선택 seed만 다섯 번 바꾼다. 군집 20 조건의 n=20은 다섯 seed 모두 같은 대표 집합이며 순서와 모델 난수만 달라진다. 따라서 이 지점의 표준편차는 독립적인 대표 집합 다섯 개의 불확실성을 나타내지 않는다.

거리 계산에만 숫자 특징의 signed log1p, 후보 기반 중앙값·사분위 범위 스케일링, [-6,6] 고정 clipping을 적용한다. 사분위 범위의 하한은 0.1이며 스케일러 적합에는 클래스당 최대 5,000개의 후보를 쓰고 클래스 가중을 동일하게 둔다. PROTOCOL, L7_PROTO, TCP_FLAGS, CLIENT_TCP_FLAGS, SERVER_TCP_FLAGS, ICMP_TYPE, ICMP_IPV4_TYPE, DNS_QUERY_TYPE, FTP_COMMAND_RET_CODE는 범주형으로 처리한다. 각 범주는 one-hot을 sqrt(2)로 나누어 불일치가 제곱거리 1을 기여하도록 하고, 전체 유클리드 거리는 sqrt(46)으로 나눈다. 이 변환은 모델 입력을 바꾸지 않는다.

군집 적합은 클래스별 최대 100,000개 후보 행을 고정 추출하고, 변환 후 같은 벡터를 합친 빈도를 sample weight로 넣는 Lloyd KMeans를 사용한다. n_init=3, max_iter=300이며 대표 검색과 군집 배정에는 해당 클래스의 전체 후보를 사용한다. 모든 클래스에서 실제 점유 군집 수가 지정한 20 또는 40개인지 검사했다. 기존 특징의 척도·clipping·범주형 지정에 의존하는 운영상 정의이므로 실제 공격 동작 또는 독립 환경의 다양성과 동일시하지 않는다.

전체 후보의 클래스 라벨과 특징을 알고 선택하는 진단이므로 이 절차를 오직 n개의 라벨만 확보한 배포 가능한 획득 방법으로 주장하지 않는다. 선택용 전체 후보와 전역 정제에 필요한 라벨 정보는 모델에 들어가는 소수 학습 라벨 예산과 별도 전제다.

## 고유벡터와 커버리지 관찰
고유벡터 수는 선택된 n개 안에서 정확히 같은 46개 특징을 합친 개수다. 커버리지는 선택에 쓰지 않은 같은 클래스 참조 각 행에서 가장 가까운 선택 표본까지의 거리로 측정한다. 그 거리의 중앙값과 90백분위수를 클래스별로 계산하고 클래스에 동일 가중을 주어 요약한다. 90백분위수는 멀리 남은 참조 영역의 변화를 보여주며 새로운 공격 패턴의 개수를 세는 지표가 아니다.
''')
    parts.append(f'![고유벡터 수와 독립 참조 거리. 음영은 seed 간 표준편차. 거리의 절대값은 데이터셋별 스케일링에 의존한다.](../../figures/{date}_exp88_coverage.pdf)')
    for ds in DS:
        rows=[]
        for arm in ARMS:
            for n in BUDGETS:
                g=cover[(cover.dataset==ds)&(cover.arm==arm)&(cover.n==n)]
                rows.append([LABEL[arm],n,f'{g.unique_vectors.mean():.2f}',f'{g.distance_median.mean():.3f}',f'{g.distance_p90.mean():.3f}',f'{g.zero_distance_rate.mean()*100:.2f}'])
        parts.append(f'### {DS[ds]} 커버리지 요약\n'+table(['선택','n','고유벡터','거리 중앙값','거리 P90','거리 0 (%)'],rows))
    parts.append('''표의 거리 0 비율은 변환된 특징 공간의 최근접 거리가 수치 허용오차 1e-7 미만인 비율이다. clipping으로 인해 원래 벡터가 다른 행도 일치할 수 있으며, 정확한 원래 벡터 중복 여부는 별도의 고유벡터 수와 test hash 진단으로 확인한다. 군집 대표 선택의 군집 점유 수는 선택 규칙의 직접적인 결과이므로 독립적인 성능 증거로 사용하지 않는다. 참조 거리와 test F1을 함께 사용한다.

## 전체 test Macro-F1
Macro-F1은 공격 클래스별 F1의 단순 평균이다. 표는 다섯 seed의 평균 ± 표본 표준편차이며 단위는 %다. 서로 다른 선택 방식과 모델의 차이는 같은 seed를 대응시켜 percentage point (pp)로 계산한다. 오차막대는 표본·모델 난수에 대한 변동이며 새로운 수집환경에 대한 신뢰구간이 아니다.
''')
    parts.append(f'![동일 학습 예산과 전체 공격 test의 학습곡선. 실선은 주 비교, 점선은 군집 수 40 민감도 검사.](../../figures/{date}_exp88_learning.pdf)')
    for ds in DS:
        rows=[]
        for arm in ARMS:
            for n in BUDGETS:
                g=main[(main.dataset==ds)&(main.arm==arm)&(main.n==n)]
                delta=paired[(paired.dataset==ds)&(paired.arm==arm)&(paired.n==n)].pfn_minus_xgb_pp
                rows.append([LABEL[arm],n,mean_sd(g[g.method=='xgb'].macro_f1),mean_sd(g[g.method=='pfn'].macro_f1),mean_sd(delta,1)])
        parts.append(f'### {DS[ds]} 전체 공격 test\n'+table(['선택','n','XGB F1 (%)','TabPFN F1 (%)','TabPFN−XGB (pp)'],rows))
    # Remaining interpretation is built from the final, independently checked data.
    parts.extend(interpretation_sections(run,frames,date))
    lead=['## 핵심 관찰']
    for ds in DS:
        g=main[(main.dataset==ds)&(main.n==20)].groupby(['arm','method']).macro_f1.mean()*100
        dg=cover[(cover.dataset==ds)&(cover.n==20)].groupby('arm').distance_p90.mean()
        lead.append(f'{DS[ds]}의 n=20에서 행 무작위 → 군집 20 대표의 F1은 XGB '
            f"{g['row_random','xgb']:.2f}% → {g['cluster20','xgb']:.2f}% "
            f"({g['cluster20','xgb']-g['row_random','xgb']:+.2f} pp), TabPFN "
            f"{g['row_random','pfn']:.2f}% → {g['cluster20','pfn']:.2f}% "
            f"({g['cluster20','pfn']-g['row_random','pfn']:+.2f} pp)다. "
            f'클래스 평균 참조 거리 P90은 {dg.row_random:.3f}에서 {dg.cluster20:.3f}로 바뀐다. '
            '이 수치는 커버리지와 분류 성능의 관계를 직접 대조하는 핵심 관찰이다.')
    lead.append('CIC2018의 무작위 소수 표본에서 보인 정체에는 표본 대표성이 관여한다. '
        '군집 20과 40 모두 같은 모델 학습 예산에서 무작위보다 높은 평균 F1을 보이므로, 기존 곡선만으로 모든 유용한 패턴을 이미 학습했거나 현재 특징의 성능 상한에 도달했다고 판단할 수 없다. '
        '이것은 후보 전체의 라벨과 특징을 이용한 진단 결과이며 실제로 클래스당 20개 라벨만 확보한 획득 절차의 성과는 아니다.')
    lead.append('모델의 상대 이점은 데이터셋과 선택 설정에 따라 다르다. '
        'n=20의 군집 대표 두 설정에서 CIC2018은 XGB가, ToN-IoT는 TabPFN이 더 높은 평균 F1을 보인다. '
        '따라서 이번 결과는 TabPFN의 보편적 소수 표본 우위보다, 표본 대표성이 개선됐을 때 데이터셋별로 다른 모델 반응을 뒷받침한다.')
    lead.append('새 실험은 같은 예산의 선택 방식을 비교하므로 선택한 표본의 대표성에 대한 근거를 제공한다. '
        '수집환경의 독립성, 패킷 표현의 이점, 특징의 정보 상한은 이번 결과로 확정하지 않는다. '
        '아래 표준편차·클래스별 결과·군집 수 민감도·중복 보조 평가를 함께 읽어야 한다.')
    text='\n\n'.join(parts).replace('## 목적과 실험 범위','\n\n'.join(lead)+'\n\n## 목적과 실험 범위',1)
    text=text.replace('의 F1은 XGB ','의 Macro-F1(공격 클래스별 F1의 평균)은 XGBoost (XGB) ',1)
    text=text.replace(' pp),',' percentage point (pp)),',1)
    text=text.replace('참조 거리 P90은','참조 거리의 90백분위수 (P90)는',1)
    summary='대표성 진단 320/320조건·독립 검산 완료; CIC2018 n=20 무작위→군집대표 F1 XGB 79.75→87.56%, TabPFN 81.66→82.59%; 군집 수·정확한 중복 제외 민감도 확인'
    target=write_record(evidence/'exp88.md',text,summary=summary,compile_pdf=False)
    compiled=build_report(target);structure=check();assert structure['passed']
    validation.update(report_pending=False,report=str(target.relative_to(ROOT)),compile=compiled,structure=structure)
    write(analysis/'validation.json',validation);write(evidence/'exp88_validation.json',validation)
    outputs=[target,*paths,*data.glob(f'{date}_exp88_*.csv')]
    info=dict(run=str(run.relative_to(ROOT)),research_version='v7',experiment='exp88',execution_start=cfg['started_at'],
        execution_end=result['updated_at'],report_written=today,conditions=320,evidence_status='exploratory_not_manuscript_adopted',
        failed_preparation_run='tabpfn/results/v7/exp88/20261027_121953_sampling_diversity_2_5_10_20_five_seeds_bb195c2e',
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in outputs},validation=str((evidence/'exp88_validation.json').relative_to(ROOT)))
    write(evidence/'exp88_manifest.json',info)
    for file in [Path(__file__),Path(__file__).with_name('analyze_selection.py'),Path(__file__).with_name('finish_selection.py')]:
        dst=run/'source_dependencies'/file.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dst)
    print(json.dumps(dict(report=str(target),compile=compiled,structure=structure),ensure_ascii=False))


def interpretation_sections(run,frames,date):
    main=frames['metrics'];deltas=frames['selection_paired_deltas'];paired=frames['model_paired_deltas']
    classes=frames['per_class'];cover=frames['coverage_class_macro'];parts=[]
    history_path=ROOT/'tabpfn/results/v7/exp87/20261022_163739_full_learning_curve_five_seeds_c59213f5/results.json'
    historical=read(history_path)['jobs'];history_rows=[]
    for ds in DS:
        for n in [20,80,320]:
            h=[j for j in historical if j['dataset']==ds and j['n']==n]
            assert len(h)==10
            history_rows.append([DS[ds],n,*[mean_sd([j['metrics']['macro_f1'] for j in h if j['method']==method]) for method in MODEL]])
    parts.append('## 기존 큰 예산 곡선과의 연결\n아래 표는 2026-10-22 EXP87의 기존 다섯 seed 결과를 읽기 전용으로 인용한 참고 자료다. 이번 EXP88과 모델 설정 및 고정 test는 같지만 학습 후보에서 참조 행을 분리했는지와 선택된 학습 행은 다르다. 따라서 EXP87과 EXP88의 수치 차이를 참조 분리 또는 선택 방식 하나의 효과로 해석하지 않는다. 큰 예산에서의 증가 폭은 기존 근거로 제시하고 이번 새 비교의 범위는 2~20개로 유지한다.')
    parts.append(table(['기존 EXP87','n','XGB F1 (%)','TabPFN F1 (%)'],history_rows))
    parts.append(f'기존 수치 출처: `{history_path.relative_to(ROOT)}`')
    parts.append('## 선택 방식의 대응 차이와 해석\n선택 방식의 효과는 각 seed의 무작위 기준에서 빼서 계산한다. 평균 차이만으로 일관성을 판단하지 않도록 다섯 seed의 차이 범위와 양의 차이를 보인 seed 수를 함께 제시한다. 군집 수 40 결과는 20개 군집과 같은 효과가 유지되는지 확인하는 민감도 검사이며 유리한 군집 수를 test에서 골라 최종 방법으로 확정하지 않는다.')
    for ds in DS:
        rows=[]
        for n in BUDGETS:
            for arm in ARMS[1:]:
                for method in ['xgb','pfn']:
                    g=deltas[(deltas.dataset==ds)&(deltas.n==n)&(deltas.arm==arm)&(deltas.method==method)].delta_pp
                    rows.append([n,LABEL[arm],MODEL[method],mean_sd(g,1),f'{g.min():+.2f} ~ {g.max():+.2f}',f'{int((g>0).sum())}/5'])
        parts.append(f'### {DS[ds]} 무작위 대비 변화\n'+table(['n','선택','모델','대응 차이 (pp)','차이 범위 (pp)','개선 seed'],rows))
        g=main[(main.dataset==ds)&(main.n==20)]
        paragraphs=[]
        for method in MODEL:
            mean=g[g.method==method].groupby('arm').macro_f1.mean()*100
            dg=cover[(cover.dataset==ds)&(cover.n==20)].groupby('arm').distance_p90.mean()
            change=mean.cluster20-mean.row_random
            paragraphs.append(f'{MODEL[method]}의 n=20 F1은 행 무작위 {mean.row_random:.2f}%, 고유벡터 무작위 {mean.unique_random:.2f}%, 군집 20 대표 {mean.cluster20:.2f}%, 군집 40 대표 {mean.cluster40:.2f}%다. '
                f'군집 20 대표와 행 무작위의 평균 차이는 {change:+.2f} pp다.')
        paragraphs.append(f'같은 n=20에서 클래스 평균 거리 P90은 행 무작위 {dg.row_random:.3f}, 군집 20 대표 {dg.cluster20:.3f}, 군집 40 대표 {dg.cluster40:.3f}이다. '
            '따라서 특징 공간의 커버리지 변화와 F1 변화는 별개 수치로 확인해야 하며, 고유벡터 수가 늘었다는 사실만으로 분류에 유용한 정보가 같은 비율로 늘었다고 해석할 수 없다.')
        parts.extend(paragraphs)
    parts.append('## 클래스별 성능과 쉬운 패턴의 범위\n클래스별 F1은 같은 test의 다른 클래스에서 유입된 오탐도 포함한다. 각 그림은 해당 데이터셋의 test 표본 수 내림차순이며 셀은 다섯 seed 평균 F1 (%)이다. 한 클래스의 F1 개선은 그 클래스의 대표 선택뿐 아니라 다른 클래스의 학습 표본 변화로 인한 오탐 감소에서도 발생할 수 있다. 현재 실험은 모든 클래스의 선택 방식을 함께 바꾸므로 개별 클래스 대표성의 독립 인과 효과를 분리하지 않는다.')
    for ds in DS:
        parts.append(f'![{DS[ds]} 클래스별 F1. 열은 2·5·10·20개, 행은 test 표본 수 내림차순.](../../figures/{date}_exp88_classes_{ds}.pdf)')
        meta=read(run/'data'/ds/'manifest.json');names=sorted([x for x in meta['class_names'] if x!='benign'],key=lambda x:meta['test_counts'][meta['class_names'].index(x)],reverse=True)
        rows=[]
        for name in names:
            g=classes[(classes.dataset==ds)&(classes.name==name)&(classes.n==20)]
            vals=g.groupby(['arm','method']).f1.mean()*100
            rows.append([name,*[f'{vals[arm,method]:.2f}' for method in MODEL for arm in ['row_random','cluster20']]])
        parts.append(table(['클래스','XGB 무작위','XGB 군집20','TabPFN 무작위','TabPFN 군집20'],rows))
        rows=[]
        raw_coverage=frames['coverage']
        for name in names:
            cg=raw_coverage[(raw_coverage.dataset==ds)&(raw_coverage.class_name==name)&(raw_coverage.n==20)]
            distance=cg.groupby('arm').reference_distance_p90.mean()
            fg=classes[(classes.dataset==ds)&(classes.name==name)&(classes.n==20)].groupby(['arm','method']).f1.mean()*100
            rows.append([name,f'{distance.row_random:.3f}',f'{distance.cluster20:.3f}',
                f"{fg['cluster20','xgb']-fg['row_random','xgb']:+.2f}",
                f"{fg['cluster20','pfn']-fg['row_random','pfn']:+.2f}"])
        parts.append(table(['클래스','무작위 거리 P90','군집20 거리 P90','XGB F1 변화 (pp)','TabPFN F1 변화 (pp)'],rows))
    parts.append('## CIC2018 web_attacks의 정밀도와 재현율\nweb_attacks의 낮은 F1은 실제 web 공격의 미탐과 다른 공격을 web으로 분류하는 오탐을 나누어 확인한다. 아래는 n=20에서 다섯 seed 평균이며 평가 web 표본 수는 127개다. 정밀도·재현율은 %이고 TP·FP·FN은 seed 평균 건수다. 평균 F1과 평균 정밀도·재현율을 다시 조합한 F1은 일반적으로 같지 않으므로 F1은 앞 표의 직접 평균을 사용한다.')
    rows=[]
    for arm in ARMS:
        for method in MODEL:
            g=classes[(classes.dataset=='cic2018')&(classes.arm==arm)&(classes.method==method)&
                      (classes.n==20)&(classes.name=='web_attacks')]
            rows.append([LABEL[arm],MODEL[method],f'{100*g.precision.mean():.2f}',f'{100*g.recall.mean():.2f}',
                         f'{g.TP.mean():.1f}',f'{g.FP.mean():.1f}',f'{g.FN.mean():.1f}'])
    parts.append(table(['선택','모델','정밀도 (%)','재현율 (%)','TP','FP','FN'],rows))
    web=classes[(classes.dataset=='cic2018')&(classes.n==20)&(classes.name=='web_attacks')]
    for method in MODEL:
        g=web[web.method==method].groupby('arm').agg(f1=('f1','mean'),fp=('FP','mean'),recall=('recall','mean'))
        parts.append(f'{MODEL[method]}의 web F1은 행 무작위 {100*g.loc["row_random","f1"]:.2f}%에서 '
            f'군집 20 대표 {100*g.loc["cluster20","f1"]:.2f}%로 변한다. '
            f'같은 비교의 평균 오탐은 {g.loc["row_random","fp"]:.1f}개에서 {g.loc["cluster20","fp"]:.1f}개로, '
            f'평균 재현율은 {100*g.loc["row_random","recall"]:.2f}%에서 {100*g.loc["cluster20","recall"]:.2f}%로 변한다.')
    parts.append('이 표는 web의 특징 영역을 더 잘 덮더라도 다른 클래스에서 유입되는 오탐이 줄지 않으면 F1이 개선되지 않을 수 있음을 점검한다. 모든 클래스의 선택을 함께 바꾼 실험이므로 오탐의 변화가 web 대표 또는 상대 클래스 대표 중 어디에서 생겼는지는 별도 대조가 필요하다. 후속 실험은 web 표본 수와 나머지 클래스의 표본 수를 독립적으로 조절하고, 학습 클래스 비율 변화와 새로운 음성 사례의 추가 효과를 구별해야 한다.')
    parts.append('## 두 모델의 차이와 오류 보완 가능성\n모델 차이는 전체 Macro-F1의 대응 차이로 먼저 평가한다. 아래 n=20 오류 표는 한 모델만 맞힌 사례를 클래스별 비율로 구한 뒤 클래스를 동일 가중하여 요약한다. 이 비율은 Macro-F1 차이와 같지 않으며 올바른 모델을 사전에 선택하는 규칙의 성능도 아니다. 두 모델의 오류가 다르다는 관찰과 실제 결합 모델의 개선은 구별한다.')
    complement=frames['complementarity'];rows=[]
    for ds in DS:
        for arm in ARMS:
            g=complement[(complement.dataset==ds)&(complement.arm==arm)&(complement.n==20)]
            pa=(g.pfn_only_correct/g.support).groupby(g.seed).mean()*100
            xa=(g.xgb_only_correct/g.support).groupby(g.seed).mean()*100
            both=(g.both_wrong/g.support).groupby(g.seed).mean()*100
            rows.append([DS[ds],LABEL[arm],mean_sd(pa,1),mean_sd(xa,1),mean_sd(both,1)])
    parts.append(table(['데이터','선택','PFN만 정답 (%)','XGB만 정답 (%)','둘 다 오답 (%)'],rows))
    parts.append('## 정확한 중복에 대한 보조 평가\n전체 test를 주 평가로 유지하면서 두 가지 보조 평가를 같은 저장 예측에 적용한다. 첫째는 모든 방식·seed의 n=20 학습 벡터 합집합과 정확히 일치하는 test 행을 제거한 고정 test다. 이 mask는 모든 예산과 모델에 공통이며 예산마다 평가 대상이 달라지는 문제를 피한다. 둘째는 test의 고유 특징 벡터마다 한 행을 남긴 평가다. 두 평가 모두 원래 test의 빈도와 클래스 구성에 영향을 주므로 주 점수를 대체하지 않는다.')
    validation=read(run/'analysis/validation.json');rows=[]
    for v in validation['dataset_checks']:
        rows.append([DS[v['dataset']],f"{v['full_test_rows']:,}",f"{v['disjoint_test_rows']:,}",f"{v['unique_test_rows']:,}"])
    parts.append(table(['데이터','원래 test','학습 합집합과 불일치','test 고유벡터'],rows))
    secondary=frames['secondary_metrics'];rows=[]
    for ds in DS:
        for arm in ARMS:
            for method in MODEL:
                g=secondary[(secondary.dataset==ds)&(secondary.arm==arm)&(secondary.method==method)&(secondary.n==20)]
                original=main[(main.dataset==ds)&(main.arm==arm)&(main.method==method)&(main.n==20)].macro_f1
                rows.append([DS[ds],LABEL[arm],MODEL[method],mean_sd(original),
                             mean_sd(g[g.evaluation=='context_union_disjoint'].macro_f1),mean_sd(g[g.evaluation=='unique_test_vector'].macro_f1)])
    parts.append(table(['데이터','선택','모델','원래 F1 (%)','불일치 F1 (%)','고유 test F1 (%)'],rows))
    disjoint=secondary[(secondary.dataset=='cic2018')&(secondary.n==20)&(secondary.evaluation=='context_union_disjoint')]
    dg=disjoint.groupby(['arm','method']).macro_f1.mean()*100
    parts.append(f'CIC2018의 고정 불일치 test에서도 행 무작위 → 군집 20 대표의 F1은 '
        f"XGB {dg['row_random','xgb']:.2f}% → {dg['cluster20','xgb']:.2f}%, "
        f"TabPFN {dg['row_random','pfn']:.2f}% → {dg['cluster20','pfn']:.2f}%다. "
        '대표 선택의 이득은 선택된 학습 벡터와 test의 정확한 일치만으로 설명되지 않는다. '
        '다만 이 결과는 근접 중복 또는 수집환경 공유의 영향을 배제하지 않는다.')
    parts.append('보조 평가의 exact hash는 기존 전처리에서 충돌 검사를 거친 원래 46개 특징의 벡터 식별자다. 근접 중복, 같은 세션, 동일 호스트나 수집환경의 공유를 제거하는 검사가 아니다. 고정 불일치 mask의 클래스별 남은 표본 수는 validation.json에 보존한다.')
    parts.append('## 준비·추론 비용\n표본 선택·커버리지 계산의 일회성 준비 비용과 모델의 construct+fit/cache 및 전체 test 추론 비용은 분리한다. 모든 모델은 새로 생성하며 TabPFN 가중치 fine-tuning은 수행하지 않는다. 아래 비용은 n=20에서 seed 평균이고 단위는 초다. GPU는 같은 장비를 순차 사용했으며 파일 저장·추가 검증 시간은 모델 추론 비용에 합산하지 않는다.')
    rows=[]
    for ds in DS:
        for arm in ARMS:
            for method in MODEL:
                g=main[(main.dataset==ds)&(main.arm==arm)&(main.method==method)&(main.n==20)]
                rows.append([DS[ds],LABEL[arm],MODEL[method],f'{g.preparation_seconds.mean():.3f}',f'{g.inference_seconds.mean():.3f}'])
    parts.append(table(['데이터','선택','모델','모델 준비 (초)','전체 test 추론 (초)'],rows))
    prepared=read(run/'prepared.json');parts.append(f'두 데이터셋의 후보 분리·스케일링·군집 적합·대표 검색·커버리지 계산과 source 검증은 합계 {prepared["seconds"]:.1f}초다. 이 비용은 전체 후보 정보를 사용하는 선택 준비 비용이며 소수 학습 행에 대한 모델 fit 비용과 구별한다.')
    parts.append('''## 구현 검증과 중단 실행의 복구
첫 준비 실행은 CIC2018 bot에서 MiniBatchKMeans의 k=20 중 실제 점유 군집이 19개여서 사전 검사에서 중단됐다. 이 실행에서는 모델 fit이나 test 예측을 수행하지 않았다. 중복이 많은 행 분포를 다루도록 적합용 변환 벡터를 합치고 빈도를 sample weight로 보존하는 Lloyd KMeans로 수정했다. 수정 후 15개 클래스 모두 지정한 20·40개 군집을 확보했고 그 뒤 320개 모델 조건을 완료했다. 첫 실행의 FAILURE.json, 소스 스냅샷과 등록 상태는 보존한다.

독립 분석기는 저장된 모든 예측의 체크섬과 confusion matrix 및 Macro-F1을 재계산했다. 원본 pickle의 특징·family와 선택 행을 다시 대조하고, clean train 소속·참조/test와의 ID 분리·클래스별 예산·중첩 선택·모델 간 동일 입력을 검사했다. 커버리지의 최근접 거리가 예산 증가에 따라 증가하지 않는지도 확인했다. TabPFN은 각 fit 이후와 예측 이후의 가중치 digest가 기존 체크포인트 기준과 같은지 검사했다. 자동 검사는 구현의 일관성을 확인하며 데이터 수집 편향이 없다는 증거를 제공하지 않는다.

## 판단 범위와 다음 실험
가설 a는 클래스와 선택 방식별로 판단해야 한다. 작은 예산에서 높은 F1을 보이는 클래스가 있더라도 모든 공격의 주요 패턴이 충분히 학습됐다는 뜻은 아니다. 선택 방식의 변경으로 F1이 개선되면 같은 라벨 예산에서도 대표성 개선 여지가 있다는 근거로 해석한다. 참조 거리가 줄어도 F1이 개선되지 않으면 현재 거리 정의와 분류 경계의 불일치, 클래스 간 혼동, 모델의 제한을 후속 검토한다. 이 경우에도 특징의 정보 상한을 곧바로 확정하지 않는다.

현재 새 비교는 2~20개 범위이므로 20개 이후의 정체 자체를 새 실험으로 재검증한 것은 아니다. 기존 EXP87의 더 큰 예산 곡선과 함께 해석하되, 선택 방식의 효과가 20개 이후에도 유지되는지는 별도 예산 확장이 필요하다. 군집 수 20과 40의 결과가 달라지면 현재 다양성 정의에 대한 민감도를 우선 인정하고, 후보 밀집도와 분류 경계를 고려한 대표 선택을 후속 대조한다. 현재 farthest-first는 대표 사이 거리를 벌리는 규칙이며 자연 빈도의 평균 참조 거리를 최소화하는 규칙이 아니다. 작은 예산에서 이 규칙이 불리하더라도 다양성 선택 일반이 무효라고 해석하지 않으며, 군집 20의 n=20에서는 모든 대표를 사용하므로 부분집합 선택 문제와 구별한다. 특정 군집 수나 거리 전처리를 test 점수로 반복 조정해 유리한 결과를 선택하지 않는다.

가설 b는 어려운 클래스 자체의 표본과 다른 클래스의 표본을 분리해 늘리는 실험으로 이어간다. 예를 들어 web_attacks의 표본을 고정한 채 다른 클래스의 표본을 늘려 오탐·정밀도 변화가 있는지 확인한다. 가설 c는 클래스별 호스트·상대 호스트·세션·시간 구간의 반복 수집 여부를 먼저 조사하고 가능한 경우에만 환경을 분리해 검증한다. 패킷 기반 표현 비교는 동일 flow 라벨 예산·분할·관찰 구간 아래에서 별도로 수행해야 하며, 이번 거리 진단만으로 표현 변경을 정당화하지 않는다.

TabPFN의 이점은 선택 방식별 대응 F1 차이와 오류 보완의 일관성으로 판단한다. 오류 보완이 관찰되더라도 실제 라우팅 또는 결합 규칙은 별도 개발 데이터에서 설계하고 평가해야 한다. 특정 모델이 이기도록 선택 방식이나 test 대상을 바꾸는 대신, 동일 입력에서 얻는 장점과 비용이 어디에 있는지를 연구 주장에 연결한다. 현재 결과를 원고나 발표자료에 자동 반영하지 않았다.

## 재현 경로
''')
    paths=[run,ROOT/'configs/experiments/exp88_protocol.json',ROOT/'scripts/v7/exp88/selection_diagnostic.py',
           ROOT/'scripts/v7/exp88/analyze_selection.py',ROOT/'scripts/v7/exp88/report_selection.py',run/'analysis',
           ROOT/'tabpfn/results/v7/exp88/20261027_121953_sampling_diversity_2_5_10_20_five_seeds_bb195c2e']
    parts.extend(f'`{p.relative_to(ROOT)}`' for p in paths)
    parts.append('原始 예측·학습 ID·군집 대표·참조 거리와 모델 설정은 실행 폴더에 보존한다. 보고서 표의 CSV는 docs/lablog/data, 검증·출처는 docs/lablog/evidence의 해당 실행 날짜 아래에 보존한다. 보고서 PDF는 임시 컴파일로만 검증하고 프로젝트에 상시 저장하지 않는다.'.replace('原始','원시'))
    return parts

if __name__=='__main__':main()
