#!/usr/bin/env python3
"""Build classwise and label-budget figures from verified measurements."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap,repo_root
bootstrap(_root)
ROOT=repo_root(__file__)
import argparse,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from exp61_utils import write,sha

METHODS=['xgb','pfn_pretrained','pfn_ft_accumulate']
LABELS=['XGB · 누적 재학습','PFN · FT 없음 · 누적 context','PFN · 초기 FT · 누적 context']
COLORS=['#778899','#36A2C8','#00306C']


def save(fig,dest,name):
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{ext}',dpi=200,facecolor='white')
    plt.close(fig)


def build(run):
    data=json.loads((run/'results.json').read_text());verify=json.loads((run/'verification.json').read_text())
    assert data['state']=='complete' and verify['passed']
    cfg=json.loads((run/'config.json').read_text());source=Path(cfg['source'])
    dest=ROOT/'docs/research/20261016';figdir=dest/'figures';figdir.mkdir(parents=True,exist_ok=True)
    font_manager.fontManager.addfont('/home/user/.local/share/fonts/Pretendard/Pretendard-Regular.otf')
    plt.rcParams.update({'font.family':'Pretendard','font.size':12,'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    jobs={(j['dataset'],j['method'],j['new_per_class']):j for j in data['jobs']}
    new_classes={};rows=[]
    for ds in cfg['datasets']:
        meta=json.loads((source/'data'/ds/'manifest.json').read_text());names=meta['class_names']
        y=np.load(Path(meta['source'])/'train_y.npy',mmap_mode='r')
        plans=json.loads((source/'data'/ds/'plans.json').read_text())
        initial=set(y[plans[0]['incoming']]);new_classes[ds]={names[c] for c in set(y[plans[-1]['released']])-initial}
        for n in [100,50,20,10]:
            for method in METHODS:
                j=jobs[ds,method,n]
                value=np.mean([c['f1'] for c in j['metrics']['classes'] if c['name'] in new_classes[ds]])
                rows.append(dict(dataset=ds,new_per_class=n,method=method,label_budget=j['label_budget'],
                    macro_f1=100*j['metrics']['macro_f1'],new_class_mean_f1=100*float(value),
                    update_seconds=j['update_seconds'],benign_fpr=100*j['metrics']['benign_false_positive_rate']))
    with (dest/'exp78_budget_values.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(3,2,figsize=(14.5,11.5));fig.subplots_adjust(left=.08,right=.98,bottom=.09,top=.87,hspace=.53,wspace=.23)
    for col,(ds,title) in enumerate([('cic2018','CIC2018'),('toniot','ToN')]):
        for row,(key,label) in enumerate([('macro_f1','최종 전체 클래스 Macro-F1 (%)'),('new_class_mean_f1','신규 도입 클래스 F1 평균 (%)'),('update_seconds','후속 갱신 합계 (초)')]):
            ax=axes[row,col];ax.set_title(title+' · '+label,loc='left',fontsize=14,color='#00306C',pad=12)
            for k,method in enumerate(METHODS):
                vals=[next(r[key] for r in rows if r['dataset']==ds and r['method']==method and r['new_per_class']==n) for n in [10,20,50,100]]
                ax.plot(range(4),vals,marker=['o','s','^'][k],color=COLORS[k],label=LABELS[k],linewidth=2)
                # Separate numeric annotations vertically across the three methods.
                offset=[-15,2,12][k]
                for x,v in enumerate(vals):ax.annotate(f'{v:.2f}',(x,v),xytext=(0,offset),textcoords='offset points',ha='center',fontsize=9,color=COLORS[k])
            ax.set_xticks(range(4),['10','20','50','100']);ax.set_xlabel('신규 공격 클래스당 라벨 수')
            ax.set_xlim(-.25,3.25);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
            if row<2:ax.set_ylim(0,105)
            else:ax.set_ylim(0,max(r['update_seconds'] for r in rows)*1.3)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False)
    fig.suptitle('신규 공격 라벨 수에 따른 최종 성능과 갱신 비용',y=.992,fontsize=20,color='#00306C')
    fig.text(.5,.022,'초기 1,000개 고정 · 동일 사례의 중첩 부분집합 · 최종 전체 test · seed 43 한 번 · 비용: 초기 준비·모든 추론 제외',ha='center',fontsize=11)
    save(fig,figdir,'exp78_label_budget')

    fig,axes=plt.subplots(1,2,figsize=(17,10));fig.subplots_adjust(left=.14,right=.985,top=.88,bottom=.09,wspace=.40)
    classrows=[]
    for ax,(ds,title) in zip(axes,[('cic2018','CIC2018'),('toniot','ToN')]):
        classes=sorted(jobs[ds,'xgb',100]['metrics']['classes'],key=lambda c:-c['support'])
        yy=np.arange(len(classes))
        for k,method in enumerate(METHODS):
            vals_by={c['name']:c for c in jobs[ds,method,100]['metrics']['classes']}
            vals=[100*vals_by[c['name']]['f1'] for c in classes];pos=yy+(k-1)*.23
            ax.barh(pos,vals,height=.2,color=COLORS[k],label=LABELS[k])
            for p,c,v in zip(pos,classes,vals):
                ax.text(v+.7,p,f'{v:.2f}',va='center',fontsize=9,color=COLORS[k])
                classrows.append(dict(dataset=ds,method=method,class_name=c['name'],test_rows=c['support'],f1=v))
        labels=[c['name'].replace('_',' ').title()+f"\n평가 {c['support']:,}개" for c in classes]
        ax.set_yticks(yy,labels,fontsize=11);ax.invert_yaxis();ax.set_xlim(0,116);ax.set_xticks([0,25,50,75,100])
        ax.set_title(title,loc='left',fontsize=18,color='#00306C',pad=14);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.set_xlabel('최종 전체 test 클래스별 F1 (%)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False)
    fig.suptitle('세 모델의 최종 클래스별 성능',y=.992,fontsize=21,color='#00306C')
    fig.text(.5,.025,'신규 공격당 100개 기준 · 초기 공격당 125개·Benign 750개 · 데이터셋별 평가 표본 수 내림차순 · seed 43',ha='center',fontsize=12)
    save(fig,figdir,'exp78_classwise_three_models')
    with (dest/'exp78_classwise_values.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(classrows[0]));w.writeheader();w.writerows(classrows)
    dist={j['dataset']:j for j in data['dist_jobs']}
    text=['# EXP78 신규 공격 라벨 예산과 FT 없는 DistPFN','',
        '랩미팅 보고를 위해 seed 43 한 번으로 비교했다. 초기 1,000개(Benign 750개, 기존 공격 두 클래스 각각 125개)는 유지하고, 이후 도입 공격당 라벨 수만 100·50·20·10개로 줄였다. 동일 클래스의 작은 학습 집합은 큰 집합에 포함되며 세 모델은 같은 사례를 사용한다. 초기 FT 조건은 EXP77의 동일한 10 epoch checkpoint를 공유하고 이후 가중치를 바꾸지 않는다. 최종 모델로 기존 전체 test를 평가했으며 단계별 예측을 합산하지 않았다.','',
        'XGB는 각 단계 누적 라벨로 새 모델을 재학습하고, PFN은 누적 context와 캐시를 갱신한다. 표시 비용은 초기 단계를 제외한 후속 갱신 시간의 합계로 통일했다. 초기 준비와 모든 추론은 그래프에서 제외하되 원시 측정은 보존했다. 100개 조건과 초기 FT checkpoint는 EXP77에서 재사용했고 50·20·10개 조건만 새로 실행했다.','',
        '## FT 없는 DistPFN','',
        'DistPFN은 사전학습 가중치를 유지한 누적 PFN의 예측 확률에 현재 query 배치의 평균 예측 분포와 context 클래스 비율을 이용한 보정을 적용했다. 초기 FT와 추가 가중치 학습, query 정답 사용은 없다. 저장 모델 복원 후 보정 전 예측은 두 데이터셋의 전체 test에서 EXP77과 모두 일치했다. 동일한 context 준비를 공유하므로 갱신 비용은 FT 없는 PFN의 기존 실측값을 재사용했다. 확률 보정은 추론 시 연산이므로 갱신 비용에 포함하지 않는다.','',
        '| 데이터셋 | Macro-F1 (%) | 정상 FPR (%) | 후속 갱신 (초) |','|---|---:|---:|---:|']
    for ds,name in [('cic2018','CIC2018'),('toniot','ToN')]:
        j=dist[ds];text.append(f"| {name} | {100*j['metrics']['macro_f1']:.2f} | {100*j['metrics']['benign_false_positive_rate']:.3f} | {j['update_seconds']:.2f} |")
    text+=['','## 라벨 예산 결과','',
        '각 클래스 표본 수는 신규 학습 라벨 수이며, 평가 표본 수는 예산에 따라 바뀌지 않는다. 신규 공격 F1 평균은 마지막 상태에서 새로 도입한 공격 클래스별 F1의 단순 평균이고 시간 단계 평균이 아니다.','',
        '| 데이터셋 | 신규 클래스당 라벨 | 누적 라벨 | 모델 | Macro-F1 (%) | 신규 공격 F1 평균 (%) | 후속 갱신 (초) |','|---|---:|---:|---|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['dataset']} | {r['new_per_class']} | {r['label_budget']:,} | {r['method']} | {r['macro_f1']:.2f} | {r['new_class_mean_f1']:.2f} | {r['update_seconds']:.2f} |")
    text+=['','## 관찰 결과','']
    for ds,label in [('cic2018','CIC2018'),('toniot','ToN')]:
        fragments=[]
        for method,name in zip(METHODS,['XGB','FT 없는 PFN','초기 FT PFN']):
            a=next(r for r in rows if r['dataset']==ds and r['method']==method and r['new_per_class']==100)
            b=next(r for r in rows if r['dataset']==ds and r['method']==method and r['new_per_class']==10)
            fragments.append(f"{name} {a['macro_f1']:.2f}% → {b['macro_f1']:.2f}%")
        text += [f"{label}에서 신규 공격당 라벨을 100개에서 10개로 줄였을 때 최종 Macro-F1은 "+', '.join(fragments)+
            '로 바뀌었다. 초기 두 공격과 Benign 학습 사례는 같으며, 이 변화는 신규 context 감소 후 전체 클래스의 마지막 예측에서 관찰한 값이다.','']
    text+=['## 해석과 후속 판단','',
        '이 비교는 초기 적응 데이터가 같은 상황에서 신규 공격의 라벨 확보량이 줄어드는 영향을 확인한다. 초기 공격까지 같은 수로 줄이는 전체 데이터 효율 실험과 구분한다. 초기 FT의 결과와 신규 context의 효과를 분리할 수 있으며, 어느 방법이 일반적으로 우세한지는 단일 seed만으로 확정하지 않는다. 평균 점수의 변화는 클래스별 F1과 정상 오탐률을 함께 보고 해석한다.','',
        '초기 1,000개를 고정하고 신규 라벨만 줄이면 신규 공격의 context 내 비율도 함께 낮아진다. 따라서 관찰된 점수 변화를 표본 다양성 감소만의 효과로 해석하지 않는다. 표본 수 자체의 영향을 더 분리할 필요가 생기면 context 클래스 비율 또는 확률 보정 조건을 통제한 대조군을 후속으로 둔다.','',
        '갱신 시간은 단일 실행의 관찰값이다. 모델을 예측할 때 드는 비용을 포함한 총 운영비 절감의 근거로 확대하지 않는다. DistPFN은 현재 배치의 예측 분포를 이용하므로, 배치 구성 변경에 따른 영향은 별도 검증 대상이다. 기존 EXP77의 FT 있는 DistPFN 결과와 다른 context 정책 결과는 그대로 보존한다.','',
        '## 산출물과 검증','',
        '- [라벨 예산 그래프](figures/exp78_label_budget.pdf)',
        '- [클래스별 F1 그래프](figures/exp78_classwise_three_models.pdf)',
        '- [라벨 예산 수치](exp78_budget_values.csv)',
        '- [클래스별 수치](exp78_classwise_values.csv)',
        f'- 실행: `{run.relative_to(ROOT)}`',
        f'- [결과]({str(Path("../../../")/run.relative_to(ROOT)/"results.json")})',
        f'- [독립 검증]({str(Path("../../../")/run.relative_to(ROOT)/"verification.json")})',
        '- 슬라이드 본문 7장 유지. 6장 세 모델 비교, 7장 FT 없는 DistPFN과 같은 갱신 비용 기준 반영. HTML과 원고·Overleaf는 수정하지 않음.','']
    (dest/'exp78_results.md').write_text('\n'.join(text))
    write(dest/'exp78_evidence.json',dict(run=str(run.relative_to(ROOT)),seed=43,seed_repetitions=1,
        state='verified_exploratory',protocol='configs/experiments/exp78_protocol.json',budget_rows=rows,
        dist_results=dist,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [run/'results.json',run/'verification.json',run/'dist_verification.json']}))
    print(dest/'exp78_results.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);build(p.parse_args().run)
