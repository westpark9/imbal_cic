"""Presentation shared by EXP87 reruns: EXP86-style curves with explicit values."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

COLORS={'xgb':'#222222','pfn':'#0072B2'}
LABELS={'xgb':'XGB','pfn':'TabPFN'}
NAMES={'cic2018':'CIC2018','toniot':'ToN-IoT'}
SEEDS={43,44,45,46,47}


def rewrite(text, frame, classes, summary, results, protocol, root, out):
    figures=root/'docs/lablog/figures'
    sections={};heading=None
    for line in text[1:]:
        if line.startswith('## '):
            heading=line[3:];sections[heading]=[]
        elif heading:sections[heading].append(line)
    complete={};values=[]
    for ds in protocol['datasets']:
        data=frame[frame.dataset==ds];grid=protocol['budgets'][ds]
        complete[ds]=[n for n in grid if all(set(data[(data.n==n)&(data.method==m)].seed)==SEEDS for m in LABELS)]
        stats={m:data[(data.method==m)&data.n.isin(complete[ds])].groupby('n').macro_f1.agg(['mean','std']) for m in LABELS}
        fig,(ax,gapax)=plt.subplots(2,1,figsize=(11.4,6.1),sharex=True,gridspec_kw={'height_ratios':[3.4,1]})
        for m,z in stats.items():
            ax.errorbar(z.index,z['mean']*100,yerr=z['std']*100,color=COLORS[m],marker='o',capsize=3,label=LABELS[m],linewidth=1.8)
            other=stats['pfn' if m=='xgb' else 'xgb']
            for n,row in z.iterrows():
                above=row['mean']>=other.loc[n,'mean']
                y=(row['mean']+(row['std'] if above else -row['std']))*100
                ax.annotate(f"{row['mean']*100:.2f}",(n,y),xytext=(0,7 if above else -8),textcoords='offset points',
                    ha='center',va='bottom' if above else 'top',fontsize=8.8,color=COLORS[m],
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.7))
                values.append(dict(dataset=ds,n=int(n),method=m,mean=float(row['mean']),sd=float(row['std']),seeds=5))
        gap=(stats['pfn']['mean']-stats['xgb']['mean'])*100
        gapax.plot(gap.index,gap.values,'D-',color='#795548',markersize=4,linewidth=1)
        for n,y in gap.items():
            gapax.annotate(f'{y:+.2f}',(n,y),xytext=(0,7 if y>=0 else -8),textcoords='offset points',
                ha='center',va='bottom' if y>=0 else 'top',fontsize=8.5)
        if len(gap):
            padding=max(4.5,float(gap.max()-gap.min())*.4)
            gapax.set_ylim(min(-1,float(gap.min())-padding),max(1,float(gap.max())+padding))
        pending=sorted(set(grid)-set(complete[ds]))
        if pending:ax.text(.02,.04,'5-seed results pending at n = '+', '.join(map(str,pending)),transform=ax.transAxes,fontsize=9)
        ax.set(title=f'{NAMES[ds]} — attack-only learning curve (5 seeds)',ylabel='Macro-F1 (%)',ylim=(0,108))
        ax.set_yticks(range(0,101,20));ax.grid(alpha=.2);ax.legend(loc='lower right')
        gapax.axhline(0,color='#888888',linewidth=.8);gapax.grid(alpha=.2)
        gapax.set(ylabel='PFN − XGB\n(pp)',xlabel='Labeled rows per attack class, n (log scale)')
        gapax.set_xscale('log',base=2);gapax.set_xticks(grid);gapax.xaxis.set_major_formatter(ScalarFormatter())
        gapax.set_xlim(grid[0]/1.18,grid[-1]*1.18);gapax.tick_params(axis='x',labelrotation=35,labelsize=9)
        fig.tight_layout();fig.savefig(figures/f'exp87_quality_{ds}.pdf');plt.close(fig)
        # Keep the same class heatmap representation used in EXP86.
        data=classes[classes.dataset==ds]
        order=data.groupby('class_name').support.max().sort_values(ascending=False).index
        fig,axes=plt.subplots(1,2,figsize=(12,4.5 if ds=='toniot' else 3.6),sharey=True)
        for ax,m in zip(axes,LABELS):
            z=data[(data.method==m)&data.n.isin(complete[ds])].pivot_table(index='class_name',columns='n',values='f1',aggfunc='mean').reindex(index=order,columns=grid)*100
            ax.imshow(np.ma.masked_invalid(z.to_numpy()),vmin=0,vmax=100,cmap='Blues',aspect='auto')
            ax.set(xticks=range(len(grid)),xticklabels=grid,yticks=range(len(order)),yticklabels=order,title=f'{NAMES[ds]}: {LABELS[m]} F1 (%)',xlabel='Rows per attack class')
            ax.tick_params(axis='x',labelrotation=45,labelsize=8)
            for i in range(len(z)):
                for k in range(len(grid)):
                    v=z.iloc[i,k]
                    if np.isfinite(v):ax.text(k,i,f'{v:.1f}',ha='center',va='center',color='white' if v>60 else 'black',fontsize=8)
        fig.tight_layout();fig.savefig(figures/f'exp87_per_class_{ds}.pdf');plt.close(fig)
    (out/'figure_values.json').write_text(json.dumps(dict(points=values,complete_budgets=complete,expected_seeds=sorted(SEEDS)),indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,6))
    for col,ds in enumerate(protocol['datasets']):
        for row,(metric,label) in enumerate([('preparation_seconds','Preparation (seconds)'),('inference_seconds','Attack-test inference (seconds)')]):
            ax=axes[row,col]
            for m in LABELS:
                z=frame[(frame.dataset==ds)&(frame.method==m)&frame.n.isin(complete[ds])].groupby('n')[metric].agg(['mean','std'])
                ax.errorbar(z.index,z['mean'],yerr=z['std'],color=COLORS[m],marker='o',capsize=3,label=LABELS[m])
            ax.set_xscale('log',base=2);ax.set(title=NAMES[ds],xlabel='Rows per attack class (log scale)',ylabel=label)
            ax.set_xticks(protocol['budgets'][ds]);ax.xaxis.set_major_formatter(ScalarFormatter());ax.tick_params(axis='x',labelrotation=45,labelsize=7)
            ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(figures/'exp87_cost.pdf');plt.close(fig)
    expected=10*sum(map(len,protocol['budgets'].values()))
    result=['# EXP87 · 극소수 공격 사례의 TabPFN·XGB 분류 비교 — EXP86 학습곡선 확장',
        '## 목적과 실행 상태',
        'EXP87은 EXP86의 공격-only 학습곡선에서 공격당 학습 사례 수를 20개보다 크게 늘린 후속 실험이다. 같은 모델·같은 공격 test·같은 seed를 유지하여 표본 수에 따른 Macro-F1과 클래스별 품질 변화를 확인한다. XGB n=1의 분할 불가 조건을 제외하여 2개부터 시작하며, 정상 포함 평가는 사용자 요청에 따라 제외한다.',
        f"현재 상태는 {results['state']}, 완료 조건은 {len(frame)}/{expected}개다. 전체 x축에서 seed 43–47의 5회 평균·표본 표준편차를 비교한다. 5개 seed가 모두 끝나지 않은 조건은 주 곡선에서 제외하고 그림에 대기 중인 n을 명시한다.",
        '## 공통 설정과 평가 의미',
        *sections['공통 설정과 표본 선택'][:2],
        '평가는 EXP86 Phase A와 동일한 고정 공격 test의 Macro-F1이며 각 공격의 F1을 같은 가중치로 평균한다. 각 seed에서 계산한 점수를 평균하며 seed 간 혼동행렬을 먼저 합치지 않는다. 정상 test는 추론에서도 제외하고 attack-only query 순서와 batch8192를 모든 n에서 동일하게 사용한다. 특징 전처리와 모델 설정은 EXP86을 유지한다.',
        '## 실제 제공·평가 표본 수',
        '| 데이터 | 공격 수 | 공격당 n | 전체 학습/context | 공격 test |','|---|---:|---|---|---:|',
        '| CIC2018 | 6 | 2/5/10/20/40/80/160/320/450 | 6n; 최대 2,700행 | 390,270 |',
        '| ToN-IoT | 9 | 2/5/10/20/40/80/160/320/640/1280/2151 | 9n; 최대 19,359행 | 797,421 |',
        '가로축 끝점은 성능 포화가 아니라 균등 표본 선택의 데이터 상한이다. CIC2018의 web_attacks 학습 사례는 450개, ToN-IoT의 ransomware는 2,151개여서 모든 공격에 같은 n을 주는 현재 조건에서는 더 늘릴 수 없다.',
        '## Attack-only Macro-F1 (%)',
        'EXP86과 같이 XGB는 검정색, TabPFN은 파란색이며 점은 5-seed 평균, 오차 막대는 표본 표준편차다. 각 점에 평균을 소수 둘째 자리까지 표시하고 아래 패널에는 TabPFN−XGB의 차이(%p)를 표시한다. 2개부터 수천 개까지 한 곡선에서 읽을 수 있도록 x축은 로그 간격이며 실제 n을 모두 표시한다. 모든 조건의 평가 집합과 평균에 포함한 seed는 동일하다.']
    for ds in protocol['datasets']:
        result.append(f'![{NAMES[ds]}: 5 seeds 평균±표본 표준편차; 점 옆 수치는 평균, 아래 패널은 모델 간 차이](../../figures/exp87_quality_{ds}.pdf)')
    result.extend(['## 수치 요약',
        '두 모델 모두 동일 5개 seed가 완료된 조건의 평균±표본 표준편차다. 모든 원시 결과와 부분 완료 조건은 values.csv와 summary.csv에 보존한다. 표준편차는 선택 표본·모델 seed 변동이며 신뢰구간이나 통계적 동등성을 뜻하지 않는다.',
        '| 데이터 | n | XGB F1 (%) | TabPFN F1 (%) | PFN−XGB (%p) |','|---|---:|---:|---:|---:|'])
    for ds in protocol['datasets']:
        for n in complete[ds]:
            z=summary[(summary.dataset==ds)&(summary.n==n)].set_index('method');a=z.loc['xgb'];b=z.loc['pfn']
            result.append(f'| {NAMES[ds]} | {n:,} | {a.mean_f1*100:.2f} ± {a.sd_f1*100:.2f} | {b.mean_f1*100:.2f} ± {b.sd_f1*100:.2f} | {(b.mean_f1-a.mean_f1)*100:+.2f} |')
    result.extend(['## 클래스별 결과','EXP86의 클래스별 F1 그림을 같은 n 범위까지 확장한다. 각 셀은 5-seed 평균이며 클래스는 고정 공격 test 표본 수 내림차순이다. 반복이 남은 열은 공란으로 표시한다.'])
    for ds in protocol['datasets']:
        result.append(f'![{NAMES[ds]}: 클래스별 F1의 5-seed 평균; 고정 공격 test 표본 수 내림차순](../../figures/exp87_per_class_{ds}.pdf)')
    result.extend(['## 비용',
        'EXP86과 같이 초기 준비와 추론을 분리한다. 준비는 새 모델 생성과 XGB fit 또는 PFN context/cache 준비의 합이며 순차 갱신 비용이 아니다. 이번 추론시간은 공격 test만 처리한 시간이므로 EXP86의 정상 포함 전체 test 추론시간과 직접 비교하지 않는다.',
        '![초기 준비와 공격 test 추론시간; 동일 5 seeds 평균±표본 표준편차](../../figures/exp87_cost.pdf)',
        '## 근거에 따른 해석과 추가 확대 가능성',
        '현재 결과는 수렴을 확정하지 않는다. 이전 보고서는 수렴 절차를 앞세우고 단일 seed 곡선과 마지막 세 조건의 평균 곡선을 분리하여 EXP86에서 이어지는 학습곡선을 읽기 어렵게 했다. 이번 개정에서는 전체 범위를 동일 5-seed 평균으로 연결하고 수렴 규칙과 실행 이력을 뒤로 옮겼다.'])
    for ds in protocol['datasets']:
        compared=summary[(summary.dataset==ds)&summary.n.isin(complete[ds])].pivot(index='n',columns='method',values='mean_f1')
        gaps=(compared['pfn']-compared['xgb'])*100 if {'pfn','xgb'}<=set(compared.columns) else compared.iloc[:0,0:0].sum(axis=1)
        close=[int(n) for n in gaps.index[gaps.abs()<=1]]
        result.append(f'{NAMES[ds]}에서 평균 차이가 1%p 이내인 측정 조건은 n={close}다. 이는 점수 차이를 읽기 위한 기술적 기준이며 통계적 동등성이나 그 뒤의 모든 n에서 비슷함을 의미하지 않는다.')
        ends=complete[ds][-2:]
        if len(ends)==2:
            for m in LABELS:
                z=summary[(summary.dataset==ds)&(summary.method==m)].set_index('n');a,b=[z.loc[n,'mean_f1']*100 for n in ends]
                result.append(f'{NAMES[ds]} {LABELS[m]}의 마지막 두 완료 조건은 n={ends[0]:,}→{ends[1]:,}, 평균 F1 {a:.2f}→{b:.2f}% (변화 {b-a:+.2f}%p)다.')
    result.extend([
        '특히 ToN-IoT에서 마지막 구간의 개선이 계속되어 더 큰 학습 집합을 시험할 근거가 있다. CIC2018에서는 전체 평균 변화가 작더라도 클래스별 품질과 seed 변동을 함께 확인해야 한다. 서로 다른 공격 정의·클래스 수·특징 분포를 가진 두 데이터셋의 고유 난이도를 점수만으로 단정하지 않는다.',
        '균등한 공격당 n을 더 늘리려면 가장 작은 공격 클래스의 새로운 독립 학습 사례가 필요하다. 현재 데이터만으로는 희귀 공격을 모두 유지하고 나머지 공격의 사례를 늘리는 대안을 구성할 수 있다. 후자는 클래스 비율도 달라지므로 기존 균형 곡선과 구분하여 보고해야 하며, 같은 행의 복제를 새로운 사례로 세지 않는다.',
        '후속 제안은 CIC2018에서 목표 n=800/1600/3200, ToN-IoT에서 목표 n=3200/6400으로 놓고 클래스별 실제 사용 수를 min(n, 가용 학습 사례 수)로 설정하는 것이다. seed43에서 먼저 확인하는 안이며 아직 실행하지 않았다. 구체적인 클래스별 수와 전체 행 수는 configs/experiments/proposals/20261022_exp87_beyond_balanced_cap.json에 있다.',
        '## 마지막 측정점의 precision·recall·F1',*sections['마지막 측정점의 클래스별 품질'],
        '## 재현 점검과 실행 이력',
        '처음에는 seed43으로 전체 범위를 탐색하고 마지막 세 n만 seed44–47로 확인했다. 사용자 피드백에 따라 중간 구간까지 나머지 seed를 채워 모든 n에서 같은 반복 수로 비교한다. 완료한 기존 88조건은 원본 결과를 보존하여 재사용하고 빠진 112조건만 추가한다.',
        *sections['목적과 현재 상태'][:1],
        *sections['현재 탐색 결정'],
        '최초 탐색은 최근 세 n의 점수 범위가 두 모델 모두 1%p 이내인지 확인하는 방식이었다. 실제 종료 원인은 데이터 상한이며 수렴 확정이 아니다. 모델 간 차이가 1%p 이내인 것과 각 모델이 더 이상 개선되지 않는 것은 다르다.',
        '## EXP86과 query 배치 변경 점검',*sections['기존 EXP86과 query 배치 변경 점검'],
        '## 보조 평가의 범위',*sections['Feature-disjoint attack의 의미'],
        '이미 검토한 고정 test의 탐색적 학습곡선이며 추가 seed는 표본 선택·모델 변동을 확인한다. 이 실험만으로 새로운 네트워크 환경에서의 일반화나 전체 IDS 운영 성능을 입증하지 않는다.',
        '## 재현·검증 자료',*sections['재현·검증 자료'],
        '그림에 표시한 평균·표준편차와 완료 n은 실행 폴더의 figure_values.json에 저장했다. 비용과 클래스별 결과도 동일 seed 집합만 사용한다.'])
    continuity=out/'exp86_continuity_check.json'
    if continuity.exists():
        check=json.loads(continuity.read_text())
        index=result.index('## EXP86과 query 배치 변경 점검')+1
        result.insert(index,f"추가로 n=2/5/10/20의 두 모델·5 seeds 전체 {check['compared_jobs']}조건을 EXP86과 대조했다. 모든 학습 행 ID가 동일하며, 정상 query를 제외한 배치 변경 후 Macro-F1 절대 차이의 최댓값은 {check['max_f1_difference_pp']:.5f}%p다. 해당 구간의 연속성을 exp86_continuity_check.json에 보존한다.")
    return result
