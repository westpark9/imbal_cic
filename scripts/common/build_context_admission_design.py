#!/usr/bin/env python3
"""Reproduce the design's count audit and figure; never fits a model."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from experiment_paths import repo_root

ROOT=repo_root()

def main():
    config=json.loads((ROOT/'configs/experiments/proposals/20261020_context_admission.json').read_text())
    assert config['status']=='design_only_not_executed'
    frame=pd.read_csv(ROOT/'docs/lablog/data/exp83_budget_values.csv')
    old=frame[(frame.new_per_class==400)&frame.method.isin(['pfn_accumulate','pfn_fifo','pfn_random','pfn_similar'])]
    old.to_csv(ROOT/'docs/lablog/data/20261020_context_design_observations.csv',index=False)
    font_path=Path('/home/user/.local/share/fonts/Pretendard/Pretendard-Regular.otf')
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams['axes.unicode_minus']=False
    ns=config['available_labels_per_new_class']
    fig,axs=plt.subplots(1,2,figsize=(11.5,3.4),layout='constrained')
    for ax,(title,k) in zip(axs,[('CIC2018',1),('ToN-IoT',2)]):
        ax.plot(range(6),[k*n/10 for n in ns],'o-',label='기존: 신규 사례 전부 편입',color='#80909F')
        ax.plot(range(6),[min(k*n,100)/10 for n in ns],'o-',label='설계: 단계당 최대 100개 편입',color='#007BC6')
        ax.set_xticks(range(6),ns);ax.set_ylim(0,85);ax.set_yticks([0,10,20,40,60,80]);ax.grid(axis='y',alpha=.2)
        ax.set_title(title+' · 한 단계 최대 교체 비율',loc='left');ax.set_xlabel('신규 공격당 확보 라벨 수');ax.set_ylabel('1,000개 버퍼 대비 교체율 (%)')
        ax.spines[['top','right']].set_visible(False)
    axs[0].legend(frameon=False,fontsize=9,loc='upper left')
    for ext in ['pdf','png']:fig.savefig(ROOT/f'docs/lablog/figures/20261020_admission_design.{ext}',dpi=180)
    plt.close(fig)
    checks=[]
    for ds,arrivals in [('cic2018',[1,1,1,1]),('toniot',[2,1,1,2,1])]:
        for n in ns:
            for q in [50,100,200]:
                counts=[750,125,125];ever=counts.copy();rows=[]
                for stage,k in enumerate(arrivals,2):
                    floor=[min(500,ever[0])]+[min(25,v) for v in ever[1:]]
                    total=min(k*n,q,1000-sum(floor));quotas=[total//k+(i<total%k) for i in range(k)]
                    assert min(quotas)>=1 and max(quotas)<=n
                    need=total
                    # Any feasible removal order suffices for this count-only check.
                    for c in range(len(counts)):
                        remove=min(need,counts[c]-floor[c]);counts[c]-=remove;need-=remove
                    assert need==0
                    counts+=quotas;ever+=quotas
                    assert sum(counts)==1000 and all(counts[i]>=floor[i] for i in range(len(floor)))
                    rows.append(dict(stage=stage,incoming=k*n,admitted=total,per_new_class=quotas,
                                     protected_old_total=sum(floor),context_rows=sum(counts)))
                checks.append(dict(dataset=ds,n=n,q=q,stages=rows))
    dest=ROOT/'docs/lablog/evidence/20261020/context_admission_feasibility.json'
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(dict(passed=True,scope='Count constraints only, not model performance or real-row selection validation',checks=checks),ensure_ascii=False,indent=2)+'\n')
    print(f'{len(checks)} feasible count schedules; no model experiment executed')

if __name__=='__main__':main()
