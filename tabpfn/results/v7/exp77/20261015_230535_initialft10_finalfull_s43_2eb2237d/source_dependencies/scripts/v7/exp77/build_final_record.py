#!/usr/bin/env python3
"""Build research evidence tables from independently verified EXP77 outputs."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)
import argparse,json
import numpy as np
from exp61_utils import write


def build(run):
    report=json.loads((run/'results.json').read_text());assert json.loads((run/'verification.json').read_text())['passed']
    methods=['xgb','pfn_pretrained','pfn_ft_accumulate','pfn_ft_fifo','pfn_ft_similar','pfn_ft_random','pfn_ft_fixed','pfn_ft_repeated','ours','boostpfn','distpfn_ft']
    labels=['XGB','PFN FT 없음','PFN 초기 FT 누적','PFN 초기 FT FIFO','PFN 초기 FT 유사도','PFN 초기 FT 무작위','PFN 초기 FT 고정','PFN 매 단계 FT','제안 모델','BoostPFN','DistPFN 초기 FT']
    lines=['# EXP77 · 초기 미세조정과 최종 상태 전체 test 평가','',
        'EXP76의 공개 라벨 ID를 유지한 채 초기 IDS 가중치 미세조정을 실제 수행하고, 모든 클래스 도입 후 최종 상태로 전체 test를 평가했다. 단계별 예측을 합산한 점수나 단계 F1의 평균이 아니다. 초기 1,000개 이후 신규 클래스당 100개를 공개했으며 최종 라벨 예산은 CIC2018 1,400개, ToN 1,700개다. 검증·보정 라벨도 이 안에 포함된다. LoCalPFN은 제외했다.','',
        '공식 v3 fine-tuning API, AdamW, learning rate 1e-5, 고정 10 epoch를 사용했다. Context 정책 간에는 동일 초기 FT checkpoint를 공유하고 이후 본체 가중치를 고정했다. 매 단계 FT만 이전 checkpoint에서 누적 공개 학습 행으로 다시 10 epoch 갱신한다. 미세조정의 학습·검증 역할과 제안 모델의 다섯 역할은 특징 그룹 단위로 분리하여 단계 사이에도 유지했다.','',
        '제안 모델은 기존 K=4 anchor/residual expert와 S1V0 구조이며 원래 pretrained backbone을 유지한다. 공개 라벨을 Global/expert/route/tune/cal에 약 40/20/20/8/12% 배정했다. 따라서 같은 라벨 예산의 방법 비교이고 같은 초기 FT backbone의 구조 ablation은 아니다. DistPFN은 초기 FT 누적 PFN의 확률을 현재 query 배치 posterior로 보정한 변형이다. 원 논문의 benchmark 수치와 구분한다.','']
    evidence=[]
    for ds in ['cic2018','toniot']:
        jobs={j['method']:j for j in report['jobs'] if j['dataset']==ds}
        audit=json.loads((run/'finetuning'/ds/'initial'/'audit.json').read_text())
        lines += [f'## {ds} 전체 결과','',
            f"초기 FT: {audit['train_rows']:,}개 학습, {audit['validation_rows']:,}개 검증, {audit['changed_tensors']}개 파라미터 텐서 변화, {audit['seconds']:.2f}초. 마지막 모델의 test {jobs['xgb']['test_rows']:,}개를 모두 평가했다.",'',
            '| 방법 | Macro-F1 (%) | 정상 FPR (%) | 초기 준비 (s) | 후속 갱신 (s) | 전체 추론 (s) | 준비·추론 합계 (s) |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for method,label in zip(methods,labels):
            j=jobs[method];m=j['metrics'];total=j['total_preparation_seconds']+j['serving_seconds']
            lines.append(f"| {label} | {100*m['macro_f1']:.2f} | {100*m['benign_false_positive_rate']:.3f} | {j['initial_seconds']:.2f} | {j['update_seconds']:.2f} | {j['serving_seconds']:.2f} | {total:.2f} |")
            evidence.append(dict(dataset=ds,method=method,macro_f1=m['macro_f1'],initial_seconds=j['initial_seconds'],update_seconds=j['update_seconds'],serving_seconds=j['serving_seconds'],total_seconds=total))
        x=jobs['xgb'];pre=jobs['pfn_pretrained'];ft=jobs['pfn_ft_accumulate'];rep=jobs['pfn_ft_repeated'];ours=jobs['ours']
        lines += ['',f"초기 FT 누적 PFN의 Macro-F1은 {100*ft['metrics']['macro_f1']:.2f}%이며, FT 없는 누적 PFN {100*pre['metrics']['macro_f1']:.2f}%, XGB {100*x['metrics']['macro_f1']:.2f}%와 비교된다. 초기 FT 효과는 {100*(ft['metrics']['macro_f1']-pre['metrics']['macro_f1']):+.2f}%p다. 매 단계 FT는 {100*rep['metrics']['macro_f1']:.2f}%이고 후속 갱신 합계 {rep['update_seconds']:.2f}초로, 초기 FT만 하는 누적 정책의 {ft['update_seconds']:.2f}초와 구분된다. 초기·반복 미세조정의 이득은 이 결과로 판단하며 자동으로 개선된다고 전제하지 않는다.",'',
            f"제안 모델의 Macro-F1은 {100*ours['metrics']['macro_f1']:.2f}%이고 최종 test의 expert 호출은 {ours['calls']:,}건, 변경 채택은 {ours['accepted']:,}건이다. 작은 예산을 다섯 역할로 나누면서 Global context와 보정 표본 수가 함께 줄어드는 조건이다. 이 관찰만으로 구조 자체의 우열을 결정하지 않고, 같은 Global 표본을 쓰는 ablation과 보정 데이터의 최소 요구량을 다음 검증으로 둔다.",'',
            '## '+ds+' 클래스별 F1','',
            '최종 전체 test의 평가 표본 수 내림차순. 수치는 F1 (%)다. 학습 사례 수와 평가 support를 구분한다.','',
            '| 클래스 | 평가 표본 수 | '+ ' | '.join(labels)+' |',
            '|---|---:|'+ '|'.join(['---:']*len(labels))+'|']
        for c in sorted(jobs['xgb']['metrics']['classes'],key=lambda c:-c['support']):
            values=[next(v for v in jobs[m]['metrics']['classes'] if v['name']==c['name'])['f1'] for m in methods]
            lines.append('| '+c['name']+f" | {c['support']:,} | "+' | '.join(f'{100*v:.2f}' for v in values)+' |')
        lines+=['','### 기존·신규 공격의 최종 유지 성능','',
            '각 그룹 내 클래스별 F1의 단순 평균이며, 시간 단계 평균이 아니다. 정상 클래스는 별도로 평가된다.','',
            '| 방법 | 초기 두 공격 F1 평균 (%) | 이후 도입 공격 F1 평균 (%) | 특징 중복 제외 Macro-F1 (%) |',
            '|---|---:|---:|---:|']
        initial_names=set(jobs['pfn_ft_fixed']['fits'][0]['classes'])-{'benign'}
        for method,label in zip(methods,labels):
            j=jobs[method];cs=j['metrics']['classes']
            old=np.mean([c['f1'] for c in cs if c['name'] in initial_names]);new=np.mean([c['f1'] for c in cs if c['name'] not in initial_names|{'benign'}])
            lines.append(f"| {label} | {100*old:.2f} | {100*new:.2f} | {100*j['feature_disjoint_diagnostic']['macro_f1']:.2f} |")
        lines+=['']
    lines += ['## 해석과 후속 판단','',
        '보정 후보를 점검한 결과, 정오답이 바뀐 사례(helpful + harmful)는 CIC2018에서 최대 0건, ToN에서 최대 4건으로 기존 최소 30건 조건에 못 미쳤다. 후보 expert가 Global과 다른 유용한 결정을 충분히 제공하지 못한 것이 관찰된 사실이며, 작은 보정 집합만이 그 원인이라고 단정하지 않는다. 작은 예산에서는 기존 anchor 상한에 걸리지 않아 Global 표본 전체가 각 expert에 들어간다. 공통 anchor 비중을 줄이고 전문 표본 비중을 높이는 대조군, 같은 Global 표본의 구조 ablation, 보정 표본 배분 변화가 다음 개선 방향이다. 이번 test 결과에 맞추어 선택 기준을 낮추거나 재선택하지 않았다. 실행 폴더의 calibration_diagnostics.json과 각 stage의 contexts.json·threshold_grid.csv가 근거다.','',
        '평가 범위 변경도 별도로 확인했다. XGB는 EXP76 마지막 단계와 공통인 CIC2018 657,500개·ToN 473,794개 표본에서 예측이 완전히 일치했다. 따라서 이전 점수와 이번 XGB 점수의 차이는 같은 최종 모델을 전체 test에 적용한 데 따른 차이이며, 추가 학습 라벨을 사용한 결과가 아니다. 근거는 실행 폴더의 evaluation_change_check.json이다.','',
        '후속 갱신 비용, 초기 미세조정 비용, 전체 추론 비용은 서로 다른 비용 항목이다. 전체 workload 합계는 초기 준비와 모든 후속 갱신, 최종 전체 test 추론의 합계이며 디스크 checkpoint 저장·검증 코드는 제외했다. 한 GPU에서 각 방법을 순차 실행했다. 갱신만 빠른 결과를 총비용 절감으로 확대하지 않는다.','',
        '고정 context는 신규 클래스 라벨을 포함하지 않으므로 해당 이름을 출력할 수 없다. 사전학습과 초기 fine-tuning이 필요 없다는 뜻이 아니라, v3의 출력 확률이 추론 context 라벨을 참조한다는 구조상의 제약이다. 누적·버퍼 정책은 기존 클래스 사례를 보존한다. 유사도 삭제는 신규 학습 사례와 가까운 기존 사례를 제거하며, 경계 사례가 없어질 수 있어 단순 중복 제거와 동일시하지 않는다. 개선 후보는 클래스별 최소 표본 수와 정상·공격 경계 보존 제약이다.','',
        '최종 점수는 최종 모델의 품질을 답한다. 도입 중간에 발생한 오류나 망각 회복 과정은 별도의 고정 평가 집합 단계별 곡선이 필요하다. 실제 네트워크 drift, 라벨 확보 이전 OOD 탐지, 통계적 성능 동등성은 이번 실험으로 입증하지 않았다. 단일 seed의 탐색 근거이며 추가 seed·라벨 예산 변화 검증을 거쳐 원고의 최종 성능 주장으로 채택한다.','',
        '1014의 누적 10만 라벨 결과는 다른 예산·평가 집계 조건이므로 이번 슬라이드 본문에서 제외하고 원본 및 이전 자료를 보존한다. 원고는 반영안만 갱신하며 HTML·Overleaf에는 반영하지 않는다.','',
        '## 실행 근거','',f'- 실행: `{run.relative_to(_root)}`',
        f'- [전체 결과](../../../{run.relative_to(_root)}/results.json)',
        f'- [독립 검증](../../../{run.relative_to(_root)}/verification.json)',
        '- [사전 지정 프로토콜](exp77_protocol.md)']
    dest=_root/'docs/research/20261015';(dest/'exp77_results.md').write_text('\n'.join(lines)+'\n')
    write(dest/'exp77_evidence.json',dict(run=str(run.relative_to(_root)),evaluation='final_full_test',results=evidence))
    print(dest/'exp77_results.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);build(p.parse_args().run.resolve())
