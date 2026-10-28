"""Report the separately identified, class-capped larger-context experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from curve_report import COLORS,LABELS,NAMES


def augment(text, extension, base, root):
    out=Path(extension);r=json.loads((out/'results.json').read_text());cfg=json.loads((out/'config.json').read_text())
    plan=cfg['extension_plan'];source=Path(cfg['source_run'])
    baseline=json.loads((base/'results.json').read_text())['jobs'];rows=[];checks=[]
    for ds,conditions in plan['conditions'].items():
        initial=[j for j in baseline if j['dataset']==ds and j['seed']==43]
        cap=max(j['n'] for j in initial)
        selected=[dict(j,source='balanced_anchor') for j in initial if j['n']==cap]
        selected += [dict(j,source='extension') for j in r['jobs'] if j['dataset']==ds]
        for j in selected:
            if j['source']=='extension':
                pred=np.load(j['predictions_path'],mmap_mode='r');y=np.load(out/'data'/ds/'test_y.npy',mmap_mode='r')
                names=j['class_names'];b=names.index('benign');assert not np.any(y==b) and not np.any(pred==b)
                cm=np.bincount(y.astype('int64')*len(names)+pred,minlength=len(names)**2).reshape(len(names),len(names))
                assert cm.tolist()==j['metrics']['confusion']
                expected=next(c['class_counts'] for c in conditions if c['target_n']==j['n'])
                assert {names[c]:n for c,n in enumerate(j['train_class_counts']) if c!=b}==expected
                anchor=next(a for a in initial if a['n']==cap and a['method']==j['method'])
                assert set(anchor['train_source_ids'])<=set(j['train_source_ids'])
                other=next((a for a in selected if a['n']==j['n'] and a['method']!=j['method']),None)
                if other:assert other['train_source_ids']==j['train_source_ids']
                checks.append(dict(dataset=ds,n=j['n'],method=j['method'],confusion_recomputed=True,balanced_anchor_preserved=True))
            rows.append(dict(dataset=ds,n=j['n'],method=j['method'],seed=43,context_rows=j['context_rows'],source=j['source'],
                             macro_f1=j['metrics']['macro_f1'],preparation_seconds=j['preparation_seconds'],inference_seconds=j['inference_seconds']))
    frame=pd.DataFrame(rows);frame.to_csv(out/'values.csv',index=False)
    sections=['## 추가 확대: 희귀 공격은 전부 유지하고 나머지 공격의 사례 수 증가',
        f"사용자가 추가 확대안을 승인하여 seed43으로 실행한다. 현재 상태는 {r['state']} / {r['stage']}, 추가 조건은 {len(r['jobs'])}/10개 완료다. 위의 5-seed 균형 곡선과 달리 아래는 seed43의 별도 조건이다.",
        '각 클래스의 사용 수는 min(목표 n, 해당 클래스의 가용 train 수)다. 기존 균형 조건에서 사용한 모든 사례를 유지하면서 나머지 서로 다른 행을 추가하고 같은 고정 공격 test를 평가한다. 정상·복제 행·가중치 fine-tuning은 추가하지 않는다. 표본 수와 클래스 비율이 함께 변하므로 개선이 나타나더라도 순수한 표본 수 효과로 단정하지 않는다.']
    for ds,conditions in plan['conditions'].items():
        meta=json.loads((source/'data'/ds/'manifest.json').read_text());names=meta['class_names'];cap=meta['cap'];b=meta['benign']
        order=sorted((c for c in range(len(names)) if c!=b),key=lambda c:-meta['test_counts'][c])
        headings=[str(c['target_n']) for c in conditions]
        sections.extend([f"{NAMES[ds]}의 실제 클래스별 제공 수다. 첫 열은 기존 균형 조건의 seed43 연결점(n={cap:,})이며, 행은 고정 공격 test 표본 수 내림차순이다.",
            '| 공격 클래스 | 기존 '+str(cap)+' | '+' | '.join(headings)+' |',
            '|---|'+'---:|'*(len(conditions)+1)])
        for c in order:
            sections.append('| '+names[c]+f' | {cap:,} | '+' | '.join(f"{p['class_counts'][names[c]]:,}" for p in conditions)+' |')
        sections.append(f"| 전체 학습/context | {cap*len(order):,} | "+' | '.join(f"{p['total_rows']:,}" for p in conditions)+' |')
        q=frame[frame.dataset==ds];fig,ax=plt.subplots(figsize=(9,4.1))
        for m in LABELS:
            z=q[q.method==m].sort_values('n');other=q[q.method!=m].set_index('n')
            ax.plot(z.n,z.macro_f1*100,marker='o',color=COLORS[m],label=LABELS[m])
            for _,v in z.iterrows():
                above=v.n not in other.index or v.macro_f1>=other.loc[v.n,'macro_f1']
                ax.annotate(f'{v.macro_f1*100:.2f}',(v.n,v.macro_f1*100),xytext=(0,8 if above else -9),textcoords='offset points',
                            ha='center',va='bottom' if above else 'top',color=COLORS[m],fontsize=9)
        ax.set(ylim=(0,108),title=f'{NAMES[ds]} — class-capped expansion (seed 43)',ylabel='Attack-only Macro-F1 (%)',xlabel='Target rows per class, n; rare classes capped at available rows')
        ax.set_xticks([cap]+[p['target_n'] for p in conditions]);ax.set_yticks(range(0,101,20));ax.grid(alpha=.2);ax.legend(loc='lower right')
        fig.tight_layout();fig.savefig(root/f'docs/lablog/figures/exp87_extension_{ds}.pdf');plt.close(fig)
        sections.append(f'![{NAMES[ds]}: seed43의 기존 균형 끝점과 추가 확대 조건; 각 점에 F1 표시](../../figures/exp87_extension_{ds}.pdf)')
    sections.extend(['| 데이터 | 목표 n | 총 사례 | XGB F1 (%) | PFN F1 (%) | PFN−XGB (%p) |','|---|---:|---:|---:|---:|---:|'])
    for (ds,n),q in frame.groupby(['dataset','n']):
        if len(q)!=2:continue
        a=q.set_index('method');x=a.loc['xgb','macro_f1']*100;p=a.loc['pfn','macro_f1']*100
        sections.append(f"| {NAMES[ds]} | {n:,} | {int(a.iloc[0].context_rows):,} | {x:.2f} | {p:.2f} | {p-x:+.2f} |")
    for ds,q in frame.groupby('dataset'):
        for m in LABELS:
            z=q[q.method==m].sort_values('n')
            if len(z)<2:continue
            a=z.iloc[0];b=z.iloc[-1]
            sections.append(f'{NAMES[ds]} {LABELS[m]}은 기존 균형 끝점 대비 현재 최대 목표 n={int(b.n):,}에서 {a.macro_f1*100:.2f}→{b.macro_f1*100:.2f}% ({(b.macro_f1-a.macro_f1)*100:+.2f}%p)다. 현재는 seed43 관찰이며 다른 seed의 재현 여부는 아직 확인하지 않았다.')
    sections.extend(['CIC2018 web_attacks와 ToN-IoT ransomware는 기존 끝점에서 이미 가용 학습 사례를 모두 사용했으므로, 이 두 클래스의 변화는 새로운 해당 공격 행을 추가한 효과가 아니다. ToN-IoT mitm은 2,151개에서 최대 3,285개로 실제 사례도 추가한다. 아래는 기존 균형 끝점과 가장 큰 완료 조건의 seed43 precision·recall·F1이며, 다른 공격 사례의 증가 및 클래스 비율 변화와 함께 해석한다.',
        '| 데이터 / 클래스 | 모델 | 목표 n | Precision (%) | Recall (%) | F1 (%) |',
        '|---|---|---:|---:|---:|---:|'])
    for ds,cn in [('cic2018','web_attacks'),('toniot','mitm'),('toniot','ransomware'),('toniot','dos')]:
        for m in LABELS:
            prior=[j for j in baseline if j['dataset']==ds and j['seed']==43 and j['method']==m]
            newer=[j for j in r['jobs'] if j['dataset']==ds and j['method']==m]
            selected=[max(prior,key=lambda j:j['n'])]
            if newer:selected.append(max(newer,key=lambda j:j['n']))
            for j in selected:
                c=j['metrics']['classes'][j['class_names'].index(cn)]
                sections.append(f"| {NAMES[ds]} / {cn} | {LABELS[m]} | {j['n']:,} | {c['precision']*100:.2f} | {c['recall']*100:.2f} | {c['f1']*100:.2f} |")
    prior=[j for j in baseline if j['dataset']=='toniot' and j['method']=='pfn' and j['seed']==43]
    newer=[j for j in r['jobs'] if j['dataset']=='toniot' and j['method']=='pfn']
    if newer:
        a=max(prior,key=lambda j:j['n']);b=max(newer,key=lambda j:j['n'])
        names=a['class_names'];dos=names.index('dos');xss=names.index('xss')
        x=a['metrics']['classes'][dos];y=b['metrics']['classes'][dos]
        sections.append(f"ToN-IoT TabPFN의 DoS precision은 {x['precision']*100:.2f}→{y['precision']*100:.2f}%, recall은 {x['recall']*100:.2f}→{y['recall']*100:.2f}%다. 같은 고정 test에서 XSS를 DoS로 잘못 분류한 행은 {a['metrics']['confusion'][xss][dos]:,}→{b['metrics']['confusion'][xss][dos]:,}개다. 표본 확대 후 품질 변화를 공격 누락과 다른 공격의 오분류 유입으로 나누어 확인해야 한다.")
    sections.append('다음 확인은 끝점·직전 조건의 다른 seed 재현성과, 크게 늘어난 오분류 쌍에 대한 context 구성 분석이다. 단일 seed 결과만으로 최적 context 크기나 모델의 최종 성능 상한을 정하지 않는다. 표본 수와 클래스 비율의 영향을 분리하려면 별도 배분 대조가 필요하며 이번 결과만으로 원인을 확정하지 않는다.')
    sections.append(f'추가 실행의 입력·예측·비용과 검증 근거는 `{out.relative_to(root)}`에 보존한다. 비용은 values.csv의 preparation_seconds와 inference_seconds로 분리했다.')
    text=[line for line in text if not line.startswith('후속 제안은 CIC2018')]
    insert=text.index('## 재현·검증 자료')
    return text[:insert]+sections+text[insert:],dict(state=r['state'],jobs=len(r['jobs']),checks=checks)
