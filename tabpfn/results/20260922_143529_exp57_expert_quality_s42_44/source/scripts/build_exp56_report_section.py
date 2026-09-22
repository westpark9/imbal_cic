#!/usr/bin/env python3
"""Render EXP56 comparisons and expert-quality diagnostics from frozen CSVs."""
import argparse
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'lablog/html_report/assets'
DATASETS = [('cic2018', 'CIC-IDS2018'), ('toniot', 'ToN-IoT')]
POLICIES = ['global', 'fixed_region', 'experts', 'scorer_only', 'scorer_verifier']
LABELS = ['1. Global', '2. 고정 영역 배정', '3. 각 expert 전체 test', '4. Scorer-only', '5. Scorer + verifier']
BASES = {
    'cic2018': '20260915_180000_nfv3_cic2018_exp47_clean_revalidation_s42/baseline/20260915_191945_nfv3_cic2018_exp31_c0alloc',
    'toniot': '20260916_021730_nfv3_toniot_exp48_clean_revalidation_s42/baseline/20260916_030340_nfv3_toniot_exp31_c0alloc',
}

class Markup(str):
    pass


def esc(v):
    return str(v) if isinstance(v, Markup) else html.escape(str(v))


def table(headers, rows, kind=''):
    return ('<div class="wrap"><table data-kind="'+kind+'"><thead><tr>'
            + ''.join(f'<th scope="col">{esc(v)}</th>' for v in headers) + '</tr></thead><tbody>'
            + ''.join('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in row)+'</tr>' for row in rows)
            + '</tbody></table></div>')


def f(v):
    return f'{float(v):.4f}'


def metric(v, best):
    s = f(v)
    if round(float(v), 4) == round(float(best), 4):
        s = '<strong>'+s+'</strong>'
    return Markup(f'<span data-value="{float(v):.15g}">{s}</span>')


def five_cells(frame, column):
    experts = frame[frame.condition == 3][column]
    maximum = max(float(frame.loc[p, column]) for p in POLICIES if p != 'experts')
    maximum = max(maximum, experts.max())
    values = []
    for p in POLICIES:
        if p == 'experts':
            best_id = experts.idxmax().replace('_always', '').replace('expert', 'e')
            values.append(Markup(f'{f(experts.min())}–{metric(experts.max(), maximum)} <small>({best_id})</small>'))
        else:
            values.append(metric(frame.loc[p, column], maximum))
    return values


def save_figure(fig, name, caption):
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS/(name+'.svg'), bbox_inches='tight', metadata={'Date': None})
    fig.savefig(ASSETS/(name+'.png'), bbox_inches='tight', dpi=160)
    plt.close(fig)
    svg = '\n'.join(line.rstrip() for line in (ASSETS/(name+'.svg')).read_text().splitlines())+'\n'
    (ASSETS/(name+'.svg')).write_text(svg)
    svg = svg[svg.index('<svg'):]
    # Matplotlib IDs must be unique when several figures share one document.
    ids = re.findall(r'\bid="([^"]+)"', svg)
    for old in sorted(set(ids), key=len, reverse=True):
        svg = svg.replace(f'id="{old}"', f'id="{name}-{old}"').replace(f'#{old}"', f'#{name}-{old}"').replace(f'#{old})', f'#{name}-{old})')
    svg = svg.replace('<svg ', '<svg role="img" aria-label="'+html.escape(caption, quote=True)+'" ', 1)
    return '<figure class="exp56-figure">'+svg+'<figcaption>'+html.escape(caption)+'</figcaption></figure>'


def plots(data):
    font_path = ASSETS / 'report_notosans_kr_regular.otf'
    font = FontProperties(fname=str(font_path) if font_path.exists() else '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.spines.left': False,
                         'axes.edgecolor': '#cbd5df', 'text.color': '#233344', 'axes.labelcolor': '#233344',
                         'xtick.color': '#536171', 'ytick.color': '#233344', 'svg.fonttype': 'path',
                         'svg.hashsalt': 'exp56-20260922'})
    # Register the available CJK collection explicitly (also works without fontconfig).
    from matplotlib import font_manager
    font_manager.fontManager.addfont(font.get_file())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.1), layout='constrained')
    for ax, (ds, title) in zip(axes, DATASETS):
        s = data[ds]['summary']
        for i, p in enumerate(POLICIES):
            if p == 'experts':
                e = s[s.condition == 3].macro_f1
                ax.hlines(i, e.min(), e.max(), color='#929fad', linewidth=4)
                ax.scatter(e, np.full(len(e), i), s=25, color='#60758b', zorder=3)
                label = f'{e.min():.4f}–{e.max():.4f}'
                x = e.max()
            else:
                x = s.loc[p, 'macro_f1']
                ax.barh(i, x, color=['#718096', '#529d9e', '', '#8098bd', '#295f9e'][i], height=.52)
                label = f(x)
            ax.text(x+.014, i, label, va='center', fontsize=10)
        ax.set(yticks=range(5), yticklabels=LABELS, xlim=(0, 1.05), xlabel='Macro-F1', title=title)
        ax.invert_yaxis();ax.grid(axis='x', alpha=.15);ax.set_axisbelow(True)
    macro = save_figure(fig, 'exp56_five_conditions', '그림 1. 같은 test에서 다섯 조건 비교. 3번의 점은 expert별 독립 실행이고 선은 최솟값–최댓값이다.')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), layout='constrained')
    for ax, (ds, title) in zip(axes, DATASETS):
        d=data[ds]; values=[]; labels=[]
        for _, r in d['3a_specialization'].sort_values('expert').iterrows():
            q=d['4c_test_roc_metrics'];q=q[(q['class']==r['class']) & (q.test_fpr_budget==.001)].set_index('model')
            v=q.test_descriptive_recall_at_fpr
            values.append(100*(v[f'expert{int(r.expert)}']-v['global']))
            labels.append(f'e{int(r.expert)}  {r["class"]}')
        ax.barh(range(len(values)), values, color=['#247e8e' if x>0 else '#b46452' for x in values], height=.56)
        for i,v in enumerate(values):
            ax.text(v+(1.5 if v>=0 else -1.5),i,f'{v:+.2f}',ha='left' if v>=0 else 'right',va='center',fontsize=9)
        ax.axvline(0,color='#758292',linewidth=1)
        ax.set(yticks=range(len(values)),yticklabels=labels,xlim=(-40,85),xlabel='담당 클래스 R@FPR≤0.1% 차이 (expert − global, %p)',title=title)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    quality = save_figure(fig, 'exp56_discrimination', '그림 2. 같은 오탐 한도에서 담당 클래스 recall의 global 대비 차이. 오른쪽은 개선, 왼쪽은 하락이다. Expert 번호 순서를 유지했다.')
    return macro, quality


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,default=Path('docs/research/20260922'))
    args=parser.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    data={}
    for ds,_ in DATASETS:
        p=args.results_root/ds
        d={n:pd.read_csv(p/(n+'.csv')) for n in ['1a_summary','1b_per_class','3a_specialization','4a_fixed_fpr','4b_average_precision','4c_test_roc_metrics','3b_constant_control_summary','3d_region_constant_comparison','5a_target_prediction_sources']}
        d['meta']=json.loads((p/'COMPLETE.json').read_text());d['audit']=json.loads((p/'audit.json').read_text());d['summary']=d['1a_summary'].set_index('policy')
        d['classes']=d['1b_per_class'].query('policy == "global"').sort_values('support',ascending=False)['class'].tolist()
        data[ds]=d
    macro_plot,quality_plot=plots(data)
    parts=['<section id="exp56-five-conditions">','<h2>1. 비교 기준과 평가 설계</h2>',
        '<p>Expert가 입력을 보고 클래스를 구분하는지 평가하려면, <b>정답은 채점에만 사용하고 expert 선택이나 오답 복구에 제공하지 않아야 한다.</b> 아래 다섯 조건은 같은 bank와 test를 사용하며, 이전 2번의 정답 클래스 배정을 정답을 쓰지 않는 고정 영역 배정으로 교체했다.</p>',
        table(['조건','예측 규칙','무엇을 확인하는가'],[
            [LABELS[0],'모든 행에 global의 원래 예측','공통 기준선'],
            [LABELS[1],'특징 표현·global 확률의 최근접 centroid로 expert를 배정하고 원래 답을 사용','정해진 입력 영역과 expert의 결합 성능; S/V 없음'],
            [LABELS[2],'각 expert를 전체 test의 모든 클래스에 독립 적용','해당 클래스를 맞히는 능력과 다른 클래스를 오인하는 정도'],
            [LABELS[3],'기존 S+V의 scorer 후보·호출 조건 유지, verifier만 제거','동일 호출 조건의 scorer-only 결과'],
            [LABELS[4],'기존 scorer·호출 조건·verifier 사용','제안 모델의 최종 성능']],'definitions'),
        '<p>진단은 세 질문으로 나눈다. <b>① Global보다 오탐을 억제하면서 더 잘 구분하는가? ② 사전에 정한 입력 영역에서 global·다른 expert·상수 예측보다 나은가? ③ 그 차이가 context 생성 방식의 효과이며 독립 평가에서도 재현되는가?</b> 각 자료가 답하는 질문과 추가 대조를 §4–6에 연결한다.</p>',
        '<p><b>생성 조건: 1 ≤ K &lt; C.</b> K는 global을 제외한 expert 수, C는 benign을 포함한 모델 클래스 수다. 생성 코드에서 K≥C 후보를 제외하도록 적용했다. '
        '정답 클래스 배정의 문제는 K를 줄이는 것만으로 해결되지 않으므로 그 배정 규칙도 주 평가에서 제거했다. 고정 영역 배정은 관측 가능한 규칙의 baseline이며 oracle이나 성능 상한으로 부르지 않는다.</p>',
        '<p class="note">현재 수치는 기존 bank를 이용한 재진단이다. CIC K=8·C=7은 새 제약을 충족하지 않으므로 새 설계의 결과로 주장하지 않는다. ToN K=7·C=10은 개수 조건을 충족한다. 이번에 bank 재학습은 하지 않았다.</p>',
        '<h2>2. 평가 데이터와 세 가지 수치 차이</h2>',
        '<h3>평가 데이터: 두 데이터 모두 모순 제거 고정 분할</h3>',
        '<p><b>CIC EXP47·ToN EXP48의 모순 제거 데이터가 맞다.</b> 같은 46개 feature 벡터에 서로 다른 모델 클래스 라벨이 붙은 경우 그 벡터의 행을 모두 제거했다. 같은 라벨의 중복은 보존했고, 원래 train/validation/test 소속을 유지했다. 전체 데이터의 라벨을 이용한 벤치마크 정리 실험이며 실제 운영 전처리로 주장하지 않는다.</p>',
        table(['데이터','Train','Validation','Test','잔여 모순 벡터 그룹'],[[title]+[f'{data[ds]["audit"]["split_rows"][key]:,}' for key in ['train','val','test']]+['0'] for ds,title in DATASETS],'dataset'),
        '<p class="note">Clean manifest의 hash가 frozen cache identity와 일치하고, test 행 수도 일치함을 확인했다. 이미 관찰한 개발 holdout이므로 독립 final test가 아니다. 0911 원본 데이터의 과거 점수와 직접 비교하지 않는다.</p>',
        '<h3>CIC에서 global = S-only = S+V인 이유</h3>',
        '<p>반올림 때문에 같은 것이 아니라 <b>세 예측 배열이 모든 행에서 동일</b>하다. 저장된 calibration 탐색의 개입 후보 65개가 모두 benign 오탐 증가와 tail-F1 하락 제약을 위반했다. 따라서 global 유지가 선택돼 호출 임계값이 비활성화됐고 expert 호출은 0건이다. '
        'S-only도 같은 호출 조건을 유지하면서 V만 제거하므로 그대로 global이다. 이 결과는 선택된 정책이 개입하지 않았다는 뜻이며 scorer와 verifier가 같은 역량이라는 근거가 아니다.</p>',
        '<h3>CIC infiltration: 이전 정답 배정에서만 F1 0.9140이 나온 이유</h3>']
    hist=pd.DataFrame(data['cic2018']['audit']['historical_target']).set_index('policy')
    rows=[]
    for key,label in [('global','Global'),('expert6_always','e6 전체 test'),('assignment_oracle','제외한 정답 클래스 배정')]:
        r=hist.loc[key];rows.append([label]+[f'{int(r[c]):,}' for c in ['TP','FP','FN']]+[f(r.precision),f(r.recall),f(r.f1)])
    parts += [table(['예측 규칙','TP','FP','FN','Precision','Recall','F1'],rows,'infiltration-explanation'),
        '<p>정답 배정은 실제 infiltration 행만 e6에 주고 다른 클래스는 다른 예측기에 보낸다. 그래서 <b>TP 7,556과 FN 1,186은 e6 그대로인데, e6 전체 적용에서 생기던 FP 806,387 대신 선택된 다른 예측기의 FP 236만 남는다.</b> '
        'F1 = 2×7,556 / (2×7,556 + 236 + 1,186) = 0.9140이다. 정답을 아는 배정 규칙이 음성을 걸러준 결과이며 e6의 분류 성능이 0.9140이라는 뜻이 아니다. 이 이유로 해당 값을 주 비교에서 제거했다.</p>',
        '<h3>ToN scanning: 개별 expert보다 S+V의 F1이 높은 이유</h3>']
    pc=data['toniot']['1b_per_class'];scan=pc[pc['class']=='scanning'].set_index('policy');rows=[]
    for key,label in [('global','Global'),('expert2_always','e2 전체 test'),('expert4_always','e4 전체 test: 개별 expert 최고 F1'),('scorer_only','S-only'),('scorer_verifier','S+V')]:
        r=scan.loc[key];rows.append([label]+[f'{int(r[c]):,}' for c in ['TP','FP','FN']]+[f(r.precision),f(r.recall),f(r.f1)])
    parts += [table(['예측 규칙','TP','FP','FN','Precision','Recall','F1'],rows,'scanning-explanation'),
        '<p>S+V는 expert 하나를 모든 행에 적용하지 않는다. 입력에 따라 expert를 선택하고 호출·승인 조건을 통과한 답만 사용한다. '
        '<b>Scanning TP 4,245개를 확보하면서 FP를 172개로 제한</b>하므로 F1 = 2×4,245 / (2×4,245 + 172 + 1,784) = 0.8128이다. 전체 test에 e4를 적용한 경우와 TP는 비슷하지만 FP는 7,715개다. 따라서 합성 정책의 F1은 각 expert 전면 적용 F1의 최댓값보다 높을 수 있다.</p>',
        '<p>실제 예측 출처도 확인했다. S+V scanning의 TP는 global 452개 + 선택된 expert들 3,793개이며, FP는 global 50개 + expert들 122개다. 이 예측을 구성할 때 test 정답은 사용하지 않았고, 정답 배정의 음성 제거와는 평가 규칙이 다르다.</p>',
        '<details><summary>Scanning 예측을 실제로 제공한 모델별 TP·FP</summary>',
        table(['예측 출처','TP','FP'],[[r.source_model,f'{int(r.TP):,}',f'{int(r.FP):,}'] for _,r in data['toniot']['5a_target_prediction_sources'].query('policy == "scorer_verifier"').iterrows()],'scanning-sources'),'</details>',
        '<h2>3. 수정된 다섯 조건의 성능</h2>',
        '<p class="note">2번은 정답을 쓰지 않는 고정 영역 배정으로 재계산했다. 1·3·4·5번은 기존 예측·점수와 일치한다. 굵은 값은 다섯 조건의 최고값이며 소수 넷째 자리 표시가 같으면 함께 강조한다. 3번은 K개 독립 실행의 최소–최대이고, 괄호는 최대값 expert다.</p>',
        table(['데이터']+LABELS,[[title]+five_cells(data[ds]['summary'],'macro_f1') for ds,title in DATASETS],'five-overall'),macro_plot]
    for ds,title in DATASETS:
        d=data[ds];pc=d['1b_per_class'];rows=[]
        for name in d['classes']:
            q=pc[pc['class']==name].set_index('policy');rows.append([name,f'{int(q.loc["global","support"]):,}']+five_cells(q,'f1'))
        parts += [f'<h3>{title} — 클래스별 F1</h3>',table(['클래스','Test 수','1. Global','2. 고정 영역','3. Expert 범위','4. S-only','5. S+V'],rows,'five-class-'+ds)]
        ids=sorted(d['3a_specialization'].expert.astype(int));rows=[]
        for name in d['classes']:
            q=pc[pc['class']==name].set_index('policy');pol=['global','fixed_region']+[f'expert{k}_always' for k in ids]+['scorer_only','scorer_verifier'];best=q.loc[pol,'f1'].max()
            rows.append([name,f'{int(q.loc["global","support"]):,}']+[metric(q.loc[p,'f1'],best) for p in pol])
        parts += ['<details><summary>각 expert를 모두 펼친 클래스 × 예측기 F1</summary>',table(['클래스','Test 수','1. G','2. 고정 영역']+[f'3. e{k}' for k in ids]+['4. S','5. S+V'],rows,'full-class-'+ds),'</details>']
    parts += ['<h2>4. 어떤 자료가 expert의 어떤 역량을 검증하는가?</h2>',
        '<p>핵심은 <b>그 클래스 이름을 많이 출력하는가</b>가 아니라 <b>그 클래스인 샘플과 아닌 샘플을 구분하는가</b>다. 아래 자료는 동일한 성능을 중복 나열한 것이 아니라 서로 다른 실패 원인을 구분하는 검사다.</p>',
        table(['자료','검증하려는 질문','좋은 결과의 의미 / 해석 한계'],[
            ['전체 test의 precision·recall·F1·FP','원래 예측 규칙으로 양성과 음성을 구분하는가?','Recall과 precision이 함께 좋아야 한다. 전역 macro-F1만으로 특화 영역의 능력을 판정하지 않는다.'],
            ['AP (average precision)','양성에 음성보다 높은 점수를 부여하는가?','Precision–recall 전반의 순위 성능. 특정 FPR을 고정한 지표는 아니다. 같은 클래스·같은 test에서 G와 비교한다.'],
            ['R@FPR≤0.1%','음성 1,000개당 최대 1개 오탐 수준에서도 양성을 얼마나 찾는가?','동일 오탐 한도에서의 식별력 비교. 여기서는 test ROC의 기술적 요약이며 배포 임계값을 정한 결과가 아니다.'],
            ['Val에서 고정한 임계값의 test recall·FPR','다음 데이터에서도 정한 오탐 수준과 탐지율이 유지되는가?','임계값 전이·안정성 검사. AP가 높아도 test FPR이 크게 벗어나면 그 운영점을 바로 쓸 근거는 없다.'],
            ['관측 가능한 영역 내 비교','미리 정한 입력 영역에 해당 expert를 두는 것이 적절한가?','영역과 expert의 적합성 검사. 단일 클래스 영역의 높은 accuracy는 구분 능력을 증명하지 못하므로 상수 대조·영역 macro-F1을 함께 본다.']],'diagnostic-purpose'),
        '<p><b>AP와 R@0.1%는 모두 ①의 식별력 진단에 속하지만 역할이 다르다.</b> AP는 임계값 전반의 precision–recall 관계이고, R@0.1%가 명시적으로 동일 FPR 한도를 고정한다. 모든 샘플에 같은 점수를 주면 AP는 양성 비율 수준이고, FPR 0.1%에서는 전부 양성이라고 할 수 없다. '
        '0.1%는 이번 비교에 사용한 진단 기준이며 실제 허용 오탐률을 확정한 값은 아니다. One-vs-rest 점수로 계산한 AP·R@0.1%와 원래 다중 클래스 argmax의 F1은 서로 다른 평가다.</p>',quality_plot]
    for ds,title in DATASETS:
        d=data[ds];s=d['summary'];sp=d['3a_specialization'].sort_values('expert');ap=d['4b_average_precision'];roc=d['4c_test_roc_metrics'];fixed=d['4a_fixed_fpr'];quality=[];raw=[];cal=[]
        for _,r in sp.iterrows():
            k=int(r.expert);name=r['class'];own=f'expert{k}';av=ap[ap['class']==name].set_index('model').test_average_precision
            b=roc[(roc['class']==name)&(roc.test_fpr_budget==.001)].set_index('model').test_descriptive_recall_at_fpr
            ao=av.drop(['global',own]);bo=b.drop(['global',own])
            if av[own]>av['global'] and b[own]>b['global']:verdict='두 지표의 관측 우위' if av[own]>=ao.max() and b[own]>=bo.max() else 'G 개선 · 타 expert도 유효'
            elif av[own]<av['global'] and b[own]<b['global']:verdict='G 대비 우위 미확인'
            else:verdict='지표 혼재 / 이득 제한'
            quality.append([f'e{k}',name]+[f(x) for x in [av['global'],av[own],ao.max(),b['global'],b[own],bo.max()]]+[verdict])
            q=d['1b_per_class'];q=q[(q['class']==name)&(q.policy==own+'_always')].iloc[0]
            raw.append([f'e{k}',name,f(s.loc[own+'_always','macro_f1']),f(q.precision),f(q.recall),f(q.f1),f'{int(r.outside_FP):,}'])
            q=fixed[(fixed['class']==name)&(fixed.target_cal_fpr==.001)].set_index('model')
            cal.append([f'e{k}',name,f(q.loc['global','test_recall']),f(q.loc[own,'test_recall']),f'{q.loc["global","test_fpr"]*100:.3f}%',f'{q.loc[own,"test_fpr"]*100:.3f}%'])
        parts += [f'<h3>{title} — 양성·음성 구분 능력</h3>',table(['Expert','주로 포함된 클래스','G AP','Expert AP','타 expert 최고 AP','G R@0.1%','Expert R@0.1%','타 expert 최고 R','읽는 법'],quality,'expert-quality-'+ds),
            '<p class="note">“주로 포함된 클래스”는 context 특화 block의 최다 클래스다. 실제 정답 배정에 사용하지 않는다. 타 expert 최고값은 비교용 요약이며 해당 expert를 test에서 선택한 정책의 점수가 아니다.</p>',
            '<details><summary>검사 A · 원래 예측의 과잉 출력 여부: 전체 test 적용</summary>',
            '<p>특정 클래스 recall이 높을 때 precision과 FP를 함께 확인한다. Precision이 낮고 FP가 많으면 현재 argmax 규칙이 그 클래스를 과잉 출력하는 것이다. 이것만으로 점수 순위까지 나쁘다고 단정하지 않고 위 AP·R@0.1%와 대조한다.</p>',
            table(['Expert','클래스','전면 macro-F1','Precision','Recall','F1','다른 클래스에서 온 FP'],raw,'expert-raw-'+ds),'</details>',
            '<details><summary>검사 B · 정한 오탐 한도를 유지하는가: validation → test</summary>',
            '<p>Val의 음성 점수에서 FPR≤0.1%를 만족하는 임계값을 고정한 뒤 test에 그대로 적용했다. 실제 test FPR을 먼저 읽고 그때의 recall을 비교한다. 서로 다른 test FPR의 recall만 비교하면 동일 오탐 조건의 우위라고 말할 수 없다.</p>',
            table(['Expert','클래스','G test recall','Expert test recall','G test FPR','Expert test FPR'],cal,'expert-cal-'+ds),'</details>']
        region=d['3d_region_constant_comparison'];rr=[]
        for k in sorted(region.region.unique()):
            q=region[region.region==k].set_index('model');own=q.loc[f'expert{k}'];rr.append([f'e{k}',f'{int(own.rows):,}',int(own.classes_present),f(q.loc['global','macro_f1_present']),f(own.macro_f1_present),f(q.loc['train_region_constant','macro_f1_present'])])
        parts += ['<details><summary>검사 C · 정한 영역에서 분류가 필요한가: global·expert·상수 대조</summary>',
            '<p>Test 정답 없이 정한 최근접 centroid 영역에서 비교한다. 상수 대조는 각 영역의 학습용 route 행에서 가장 많은 클래스 하나만 항상 출력한다. 이번 수정에서 추가 계산했다. 영역 내 실제 등장 클래스들의 F1 평균을 표시하며, 같은 행의 세 모델끼리 비교한다.</p>',
            table(['담당 expert / 영역','Test 수','등장 클래스 수','G 영역 macro-F1','Expert 영역 macro-F1','학습 다수 클래스 상수'],rr,'expert-region-'+ds),
            '<p class="note">CIC 영역 3·5는 test에 한 클래스만 있어 상수 예측도 1.0000이다. 이런 영역의 높은 점수는 분류 역량의 증거가 되지 않는다. 영역별 등장 클래스가 달라 이 수치를 서로 다른 영역이나 전체 test macro-F1과 직접 비교하지 않는다. Global·모든 타 expert의 영역 accuracy 원본 행렬도 CSV에 보존했다.</p></details>']
    parts += ['<h2>5. 현재 결과로 내릴 수 있는 판단</h2>',
        '<p><b>현재 bank 전체가 충분하다는 결론은 내릴 수 없다.</b> 근거가 없는 부분을 다음처럼 구분해야 한다. 이미 식별력 저하가 관측된 대상은 context를 개선하고, 상대 우위가 있는 대상은 고정 운영점과 독립 반복으로 검증한다.</p>',
        table(['대상','현재 근거','지금 가능한 판단','다음 확인'],[
            ['CIC infiltration e6','AP 0.2178→0.0944; R@0.1% 0.3007→0.0991','G 대비 식별력 보완 근거가 없다. 0.9140 정답 배정 F1을 근거로 쓰지 않는다.','K<C로 재생성하고 클래스 비율을 맞춘 무작위 context·어려운 음성 대조'],
            ['ToN scanning e2','AP 0.3310→0.8312; R@0.1% 0.1415→0.7462','이 holdout에서 G·타 expert 대비 점수 구분 능력의 이득이 있다.','같은 예산·클래스 비율 대조와 독립 seed/시간 분할'],
            ['ToN ransomware e5','AP 0.0477→0.2267; R@0.1% 0.0310→0.6996','순위 이득은 있지만 val 0.1% 임계값의 test FPR은 0.364%로 목표를 넘는다.','임계값 전이 검증·시간 변화 보정 후 새 test 재평가'],
            ['CIC bank','고정 영역 0.7477; G/S/S+V 0.8107','기존 bank·고정 영역의 결합은 G보다 낮고 실제 정책은 개입하지 않는다.','새 K 조건의 bank 재학습 및 정답 없는 validation 비교']],'evidence-verdict'),
        '<p>“충분”을 최종 판정하려면 목표 클래스·허용 FPR·최소 recall·허용 global 성능 저하를 평가 전에 정해야 한다. 지금의 AP 또는 R@0.1%에 임의의 합격선을 사후 부여하지 않는다. 다음 실험은 그 판정에 필요한 원인·전이·재현성 근거를 채우기 위한 것이다.</p>',
        '<h2>6. 추가로 필요한 실험과 각각의 판정 기준</h2>',
        table(['우선순위 / 실험','무엇을 고정하고 비교하는가','무엇이 확인돼야 하는가','상태'],[
            ['0. 정답 없는 배정 + 상수 expert 대조','동일 cache·고정 영역에서 원래 expert와 학습 다수 클래스 상수 예측 비교','높은 점수가 영역의 단일 클래스 구성만으로 설명되는지 확인','이번 수정에서 계산 완료'],
            ['1. K<C bank 재생성과 context 선택 대조','같은 K·anchor·총 context 행 수·클래스별 행 수에서 현재 실패 패턴 선택 vs 무작위 선택; 별도 동일 크기 균형 context 대조','현재 방식이 클래스 비율을 맞춘 무작위보다 AP·낮은 FPR recall에서 반복적으로 높아야 생성 방식의 효과를 지지','추가 학습 필요'],
            ['2. Global 보정만 한 대조','새 expert 없이 global의 prior/β·온도·클래스 임계값만 validation에서 보정; expert도 같은 선택 절차 사용','expert 이득이 global의 운영점 변경만으로 설명되는지 구별','추가 대조 필요'],
            ['3. 고정 운영점의 시간 전이','목표 FPR 0.1%·1%를 먼저 정하고 val에서 임계값 고정 → 이후 test에서 recall·실제 FPR·FP 수 평가','허용 FPR을 지키면서 G보다 높은 recall이 유지돼야 그 운영점의 이득을 지지','현재 holdout 계산 완료; 새 시간 분할 필요'],
            ['4. 독립 학습 반복·그룹 불확실성','최소 3개의 실제 context 추첨·학습 seed; 새 시간 holdout; 같은 벡터·시간 블록 단위 paired bootstrap','평균 이득과 불확실성 구간을 함께 제시하고 특정 seed·중복 행에만 의존하는지 확인','추가 학습·새 평가 필요'],
            ['선택. Bank 내 개별 expert 제거','해당 expert를 제거하고 같은 validation 절차로 비교 정책 재선택','bank에 꼭 필요한 expert라는 주장에만 필요; 단독 식별력 검증을 대신하지 않는다','고유 시스템 기여를 주장할 때 실행']],'needed-experiments'),
        '<p><b>생성 품질을 주장하기 위한 우선 조합은 1·2·3·4번이다.</b> 새 K 조건을 만족하는 bank를 만들고, 동일 예산 대조에서 식별력 이득이 남는지와 새 데이터에서도 정한 오탐 한도를 지키는지를 확인한다. 현재 이미 계산된 0번과 기존 확률 진단을 새 학습 결과로 세지 않는다.</p>',
        '<p class="note">기존 EXP52의 행 선택·anchor 대조와 EXP53–54의 context 구성·prior 대조는 이미 수행된 부분 근거다. 이를 폐기하거나 미실행으로 세지 않는다. 여기서 추가로 요구하는 것은 K<C의 동일 bank 예산·클래스 구성 통제, 같은 저오탐 지표, 독립 반복·새 시간 평가를 갖춘 비교다.</p>',
        '<details><summary>과거 oracle 결과와 재현 출처</summary>',
        '<p>기존 correctness oracle과 정답 클래스 배정은 주 품질 비교에서 모두 제외했다. 전자는 정오를 보고 유리한 답을 고르고, 후자는 정답 클래스로 양성과 음성을 분리한다. 과거 수치는 재현·차이 설명용으로만 보존한다. 0911 원본 데이터 감사의 과거 참조값도 별도로 구분한다.</p>',
        '<p>이번 수정의 2번·상수 대조·예측 출처 추적: <code>tabpfn/scripts/exp56_label_free_revision.py</code>. 새 평가 기본 규칙: <code>tabpfn/scripts/exp56_five_condition_evaluation.py</code>. '
        '결과: <code>'+html.escape(str(args.results_root))+'</code>. AP·FPR 진단은 동일 cache의 기존 CSV를 hash와 함께 재사용했다. 기존 조건 1·3·4·5의 점수와 S/S+V 예측 배열은 원본과 일치한다. '
        'K 제약: <code>tabpfn/scripts/nfv3_v3_exp31_c0alloc.py</code>. 개수 제약만 구현했으며 K 선택 목적함수를 새로 최적화하거나 기존 bank를 재학습하지 않았다.</p></details>','</section>']
    fragment=('\n'.join(parts)+'\n').replace('K<C', 'K&lt;C')
    (args.out_dir/'exp56_five_condition_results.html').write_text(fragment)
    from lxml import html as lh
    doc=lh.fromstring(fragment);md=['# EXP56 수정: 정답 없는 다섯 조건과 expert 역량 검증']
    for node in doc.xpath('.//h2 | .//h3 | .//p | .//table'):
        if node.tag in ['h2','h3']:md.append(('## ' if node.tag=='h2' else '### ')+node.text_content())
        elif node.tag=='p':md.append(node.text_content())
        else:
            matrix=[[c.text_content() for c in r] for r in node.xpath('.//tr')]
            if matrix:md.extend(['| '+' | '.join(matrix[0])+' |','| '+' | '.join(['---']*len(matrix[0]))+' |']+['| '+' | '.join(r)+' |' for r in matrix[1:]])
    (args.out_dir/'exp56_five_condition_results.md').write_text('\n\n'.join(md)+'\n')
    (args.out_dir/'exp56_results_manifest.json').write_text(json.dumps({'results_root':str(args.results_root),'runs':{ds:data[ds]['meta'] for ds in data}},indent=2)+'\n')
    css=(ASSETS/'exp56_report.css').read_text();font_path=ASSETS/'report_typography.css';font_css=font_path.read_text() if font_path.exists() else ''
    page='<!doctype html>\n<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CIC2018·ToN — Expert 역량 검증과 평가 설계 수정</title><style>'+css+'</style><style id="report-typography">'+font_css+'</style></head><body><main><header><p class="eyebrow">0918 연구 보고서 · 2026-09-22 평가 설계 수정 · EXP56</p><h1>Expert 역량을 검증하는 평가</h1><p class="lede">정답 없는 다섯 조건 비교, 점수 차이의 원인, 지표별 검증 목적과 생성 품질의 근거를 채울 추가 실험.</p></header>\n<!-- EXP56_RESULTS_START -->\n'+fragment+'<!-- EXP56_RESULTS_END -->\n</main></body></html>\n'
    (ROOT/'lablog/html_report/scorer_verifier_target_0918.html').write_text(page)
    print('Wrote revised shared results, 0918 report, and two SVG/PNG figures.')

if __name__=='__main__':main()
