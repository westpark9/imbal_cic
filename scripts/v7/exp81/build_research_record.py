#!/usr/bin/env python3
"""Write completed experiment records from verified measurements only."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap
from lablog import write_record
bootstrap(_root)
import csv,json,subprocess
from mixed_query_eval import SMALL,LARGE,EXPANDED,OLD_BUDGET

MIXED=_root/'tabpfn/results/v7/exp81/20261016_025421_mixed_query_final_s43_b6ba1d34'
DEST=_root/'docs/lablog/evidence/20261016'


def load(run):
    result=json.loads((run/'results.json').read_text())
    verification=json.loads((run/'verification.json').read_text())
    assert result['state']=='complete' and verification['passed']
    return result['jobs'],verification


def link(run,file='results.json'):
    return f'[{file}](../../../{(run/file).relative_to(_root)})'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']+['---:']*(len(headers)-1))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    old,_=load(LARGE);expanded,_=load(EXPANDED);mixed,verification=load(MIXED)
    budget=json.loads((OLD_BUDGET/'results.json').read_text())['jobs']+expanded
    small=[j for j in mixed if j['scope']=='small'];large=[j for j in mixed if j['scope']=='large']
    for j in budget:
        if j['new_per_class']==100 and j['method'] in ['xgb','pfn_pretrained']:
            j['metrics']=next(x for x in small if x['dataset']==j['dataset'] and x['method']==j['method'])['metrics']
    budget=[j for j in budget if j['method'] in ['xgb','pfn_pretrained']]
    lookup={(j['dataset'],j['new_per_class'],j['method']):j for j in budget}
    rows=[];detail=[]
    for ds in ['cic2018','toniot']:
        for n in [10,20,50,100,200,400]:
            x,p=[lookup[ds,n,m] for m in ['xgb','pfn_pretrained']]
            rows.append([ds,n,f"{p['label_budget']:,}",f"{100*x['metrics']['macro_f1']:.2f}",f"{100*p['metrics']['macro_f1']:.2f}",
                f"{100*(p['metrics']['macro_f1']-x['metrics']['macro_f1']):+.2f}",f"{x['update_seconds']:.2f}",f"{p['update_seconds']:.2f}"])
            for j in [x,p]:
                for c in sorted(j['metrics']['classes'],key=lambda c:(-c['support'],c['name'])):
                    detail.append(dict(dataset=ds,new_per_class=n,method=j['method'],class_name=c['name'],support=c['support'],
                        precision=c['precision'],recall=c['recall'],f1=c['f1']))
    with (DEST/'exp80_per_class.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
    expansion=[]
    for ds in ['cic2018','toniot']:
        for n in [200,400]:
            x,p=[lookup[ds,n,m] for m in ['xgb','pfn_pretrained']]
            expansion.append(f"{ds} 신규 {n}개에서 XGB {100*x['metrics']['macro_f1']:.2f}%, PFN {100*p['metrics']['macro_f1']:.2f}%로 차이는 {100*(p['metrics']['macro_f1']-x['metrics']['macro_f1']):+.2f}%p다.")
    changes=[];class_changes=[]
    for ds in ['cic2018','toniot']:
        vals=[]
        for method in ['xgb','pfn_pretrained']:
            before=lookup[ds,100,method];after=lookup[ds,400,method]
            vals.append(f"{method} {100*(after['metrics']['macro_f1']-before['metrics']['macro_f1']):+.2f}%p")
        changes.append(f"{ds}의 신규 100→400개 변화는 {', '.join(vals)}다.")
        classes=sorted(lookup[ds,100,'xgb']['metrics']['classes'],key=lambda c:(-c['support'],c['name']))
        for c in classes:
            row=[ds,c['name'],f"{c['support']:,}"]
            for method in ['xgb','pfn_pretrained']:
                before=next(k for k in lookup[ds,100,method]['metrics']['classes'] if k['name']==c['name'])
                after=next(k for k in lookup[ds,400,method]['metrics']['classes'] if k['name']==c['name'])
                row.append(f"{100*(after['f1']-before['f1']):+.2f}")
            class_changes.append(row)
    def class_f1(ds,n,name):
        return 100*next(c for c in lookup[ds,n,'pfn_pretrained']['metrics']['classes'] if c['name']==name)['f1']
    interpretation=(f"PFN의 CIC2018 Infiltration F1은 {class_f1('cic2018',100,'infiltration'):.2f}→{class_f1('cic2018',400,'infiltration'):.2f}%, "
        f"Benign은 {class_f1('cic2018',100,'benign'):.2f}→{class_f1('cic2018',400,'benign'):.2f}%로 변했다. "
        f"ToN DoS는 {class_f1('toniot',100,'dos'):.2f}→{class_f1('toniot',400,'dos'):.2f}%다. "
        'DoS의 학습 표본 수는 초기 125개로 고정되어 있으므로 ToN의 개선을 신규 공격 클래스 자체의 탐지 개선으로만 해석할 수 없다. '
        '추가 사례가 다중 클래스 결정과 학습 비율을 함께 바꾸는 조건에서 관찰된 결과이며, 특정 변화의 인과적 기제는 별도 통제가 필요하다.')
    a,b=[lookup['cic2018',n,'pfn_pretrained']['metrics'] for n in [100,400]]
    ca,cb=[next(c for c in z['classes'] if c['name']=='infiltration') for z in [a,b]]
    operational=(f"CIC2018 Infiltration은 재현율 {100*ca['recall']:.2f}→{100*cb['recall']:.2f}%와 정밀도 {100*ca['precision']:.2f}→{100*cb['precision']:.2f}%가 반대 방향으로 변했다. "
        f"Benign 오탐률도 {100*a['benign_false_positive_rate']:.2f}→{100*b['benign_false_positive_rate']:.2f}%다. "
        '이는 단순히 새 공격을 덜 탐지하게 된 결과가 아니라, 탐지 확대와 오탐 증가가 함께 나타난 관찰이다. '
        '후속 설계는 Benign·기존 공격·신규 공격의 context 배분을 통제하고, 예산 내 검증 또는 교차적합으로 오탐 제약을 정하는 방향이 적절하다. '
        '이 개선안은 이번 실행에 적용하지 않았으며 test로 문턱을 조정하지 않았다.')
    txt=['# 신규 공격당 200·400개 추가 비교 결과','',
         '사용자가 승인한 200·400개 조건을 seed 43 한 번씩 완료했다. 초기 1,000개 ID는 고정하고 기존 100개 신규 사례를 포함하도록 200·400개를 중첩 구성했다. XGB는 매 단계 누적으로 새 모델을 학습하고 PFN은 사전학습 가중치를 고정한 누적 context를 사용했다. 두 방법에 같은 사례를 공급했으며 최종 모델로 같은 전체 test를 평가했다.','',
         ' '.join(expansion),'',
         ' '.join(changes),'',
         interpretation,'',
         operational,'',
         table(['데이터','신규/클래스','최종 라벨 수','XGB F1 (%)','PFN F1 (%)','PFN−XGB (%p)','XGB 갱신 (초)','PFN 갱신 (초)'],rows),'',
         '갱신 시간은 초기 단계를 제외한 표본 선택·모델 구성·fit의 합계이며 전체 추론 시간은 포함하지 않는다. 전체 test는 CIC2018 3,042,473개, ToN 2,271,723개다. 10·20·50·100개는 기존 EXP78, 200·400개는 EXP80의 결과다. EXP81에서 혼합 query 순서의 영향을 점검했다. 100개 PFN은 전체 재추론, 다른 예산은 혼합 query 8,192개의 예측 일치 감사로 확인했다. XGB는 행별 예측기다.','',
         '이 비교는 표본 수 확대에 따른 이 두 데이터셋의 관측 결과다. 신규 클래스 수를 늘리면 학습 class prior도 바뀌고, 초기 두 공격의 표본 수는 125개로 고정된다. 따라서 모든 클래스의 라벨 수를 같은 비율로 늘린 학습 곡선이나 일반적인 PFN 우위를 뜻하지 않는다. 단일 seed 결과이므로 표본 선택의 변동성을 추정하지 않는다. 후속 선택은 Macro-F1 차이와 클래스별 혼동을 함께 보고 정해야 한다. 클래스별 지표는 [CSV](exp80_per_class.csv)에 평가 support 내림차순으로 정리했다.','',
         '신규 100→400개에서 클래스별 F1 변화(%p), 데이터셋별 평가 support 내림차순:','',
         table(['데이터','클래스','평가 수','XGB 변화','PFN 변화'],class_changes),'',
         f'근거: {link(EXPANDED)}, {link(EXPANDED,"verification.json")}, {link(MIXED,"budget_order_audits.json")}. [실험 설계](exp80_protocol.md). HTML·원고는 수정하지 않았다.','']
    write_record(DEST/'exp80_results.md', '\n'.join(txt))
    methods=['xgb','tabpfn','ours','boostpfn','distpfn']
    def large_rows(jobs):
        return [[ds,case]+[f"{100*next(j for j in jobs if (j['dataset'],j['case'],j['method'])==(ds,case,m))['metrics']['macro_f1']:.2f}" for m in methods] for ds in ['cic2018','toniot'] for case in ['natural','uniform']]
    headings=['데이터','공격 배분','XGB','TabPFN','제안 모델','BoostPFN','DistPFN']
    write_record(DEST/'exp79_results.md', '\n'.join(['# 저장된 10만 개 학습 모델의 최종 전체 test 평가','',
        '기존 마지막 단계 모델을 복구해 가중치 학습과 기준 선택을 반복하지 않고 전체 test를 평가했다. 기존 재생의 query 한 배치에서 모든 모델의 저장 예측을 정확히 재현했고, 원본 모델 파일의 해시와 새 전체 예측의 혼동행렬·Macro-F1을 검증했다. LoCalPFN은 기존 사용자 요청대로 제외했다. 아래는 원본 test 순서, query 8,192개 배치의 Macro-F1 (%)다.','',
        table(headings,large_rows(old)),'',
        '이 결과는 과거의 단계별 서비스 버전 예측 합산과 다르다. 마지막 모델 하나가 이전 단계의 클래스까지 포함한 전체 test를 예측한다. 재학습 없이 평가 기준을 맞출 수 있었으나, 파일의 원본 순서가 대부분 단일 클래스 배치를 형성함을 확인했다. DistPFN의 현재 배치 posterior 보정은 이 조건에 민감할 수 있으므로 원본 순서 결과를 보존하고 EXP81 혼합 배치 평가를 발표 근거로 사용한다.','',
        '별도 검증 데이터는 제안 모델에서 잔차 구성에 사용할 Global 확률의 temperature와 S/V expert 채택 문턱을 선택하는 데 사용했다. S/V 가중치 자체는 route 학습 역할의 데이터로 학습한다. XGB·단일 PFN·BoostPFN·DistPFN은 별도 검증 라벨을 사용하지 않는다. 소규모 제안 모델의 검증은 전체 예산 안에 포함되므로, 대규모와의 비교는 학습량만 바꾼 통제 실험이 아니다.','',
        '큰 context에서 PFN 준비가 느린 것은 가중치 fine-tuning을 반복했기 때문이 아니다. 전체 누적 context의 전처리·표현·캐시를 다시 준비했다. XGB는 동일한 GPU histogram 설정으로 누적 재학습했으며 실측 증가 폭은 PFN보다 작았다. PFN 채택의 비용 근거는 작은 context에서 확인한 범위로 제한하고, 대규모까지 자동 확대하지 않는다. 구체적인 실측 비용과 설정은 [프로토콜](exp79_protocol.md)에 기록했다.','',
        f'근거: {link(LARGE)}, {link(LARGE,"verification.json")}. 최종 발표 기준: [EXP81 결과](exp81_results.md).','']))
    sr=[]
    for method in ['xgb','pfn_pretrained','ours','boostpfn','distpfn_pretrained']:
        sr.append([method]+[f"{100*next(j for j in small if j['dataset']==ds and j['method']==method)['metrics']['macro_f1']:.2f}" for ds in ['cic2018','toniot']])
    paired=[[r['scope'],r['dataset'],r['case'],f"{100*r['original_group_macro_f1']:.2f}",f"{100*r['mixed_macro_f1']:.2f}",f"{100*(r['mixed_macro_f1']-r['original_group_macro_f1']):+.2f}",f"{r['prediction_changes']:,}"] for r in verification['counterfactuals']]
    audits=[[j['scope'],j['dataset'],j['case'],j['method'],str(j.get('mixed_audit_mismatches',j.get('original_order_prediction_mismatches',''))),str(j['full_reinference'])] for j in mixed if 'distpfn' not in j['method']]
    write_record(DEST/'exp81_results.md', '\n'.join(['# 혼합 query 배치로 통일한 최종 평가','',
        'Test 행 ID만 seed 43으로 섞고, 동일한 전체 test를 8,192개 query 배치로 평가했다. 모델·context·S/V 기준·학습 prior는 변경하지 않았다. PFN 확률과 DistPFN 보정은 전체를 다시 계산했다. 행별 예측기 XGB·제안 모델·BoostPFN은 혼합 query 감사에서 정확히 일치한 경우 저장된 전체 예측을 재사용하고 불일치 시 전체 재추론했다. 재사용 여부는 아래 감사 표와 실행 메타데이터에서 구분한다.','',
        '소규모 최종 Macro-F1 (%):','',table(['모델','CIC2018','ToN'],sr),'',
        '대규모 최종 Macro-F1 (%):','',table(headings,large_rows(large)),'',
        '같은 PFN 확률에 query 그룹 구성만 바꾼 DistPFN 대응 비교:','',
        table(['규모','데이터','배분','원본 그룹 F1 (%)','혼합 그룹 F1 (%)','차이 (%p)','예측 변경 수'],paired),'',
        '대응 비교는 저장 확률·학습 prior·평가 행을 고정하므로 관측된 보정 결과 차이는 그룹 구성에서 발생한다. 이 효과를 기존 순차 재생과 최종 평가 사이의 모든 차이로 확대 해석하지 않는다. 과거 재생은 query 배치 크기와 단계별 모델 사용도 다르다. 원본 순서의 높은 DistPFN 점수를 모델의 일관된 우위로 주장하지 않고, 최종 슬라이드는 혼합 배치 기준을 명시한다.','',
        '제안 모델 개선은 같은 Global context의 단일 PFN 대조군으로 라벨 분할 효과를 분리하는 데서 시작한다. 소량 예산에서는 공통 anchor와 전문 사례 배분 및 검증에 남기는 라벨 수를 조정할 수 있다. 이는 후속 설계안이며 이번 test 성능을 보고 threshold를 낮추거나 expert 호출을 강제하지 않았다.','',
        table(['규모','데이터','배분','모델','감사/원본 예측 불일치','전체 재추론'],audits),'',
        '단일 seed이며 혼합 순서는 특정 실제 운영 유입을 재현하지 않는다. 배치 크기는 고정했으며 여러 크기에 대한 강건성은 확인하지 않았다. 감사 표본의 예측 일치도 전체 부동소수점 출력의 수학적 동일성 증명을 뜻하지 않는다. 시드 반복은 사용자 요청대로 수행하지 않았다.','',
        f'근거: {link(MIXED)}, {link(MIXED,"verification.json")}. [실험 설계](exp81_protocol.md).','']))
    for run,note in [(LARGE,'Verified saved-model final evaluation; original class-grouped query order retained as diagnostic'),
                     (EXPANDED,'User-authorized 200/400 labels, seed43 only; verified exploratory lab-meeting comparison'),
                     (MIXED,'Verified mixed-query final comparison and paired DistPFN grouping diagnostic; no retraining or test tuning')]:
        subprocess.run([sys.executable,str(_root/'scripts/results_versions.py'),'mark',str(run),'--evidence','exploratory','--note',note],cwd=_root,check=True)
    print('Wrote verified EXP79/80/81 research records and per-class CSV')


if __name__=='__main__':main()
