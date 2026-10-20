#!/usr/bin/env python3
"""Verify source data and record the completed ToN reproduction diagnostic."""
from pathlib import Path
import sys
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import bootstrap
bootstrap(_root)
import argparse,ast,csv,json,shutil
import numpy as np
from exp61_utils import sha,write
from audit_proposed_ton import SOURCE


def function_ast(path,name):
    tree=ast.parse(path.read_text())
    return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+ ' | '.join(['---']+['---:']*(len(headers)-1))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);run=ap.parse_args().run.resolve()
    result=json.loads((run/'results.json').read_text());verification=json.loads((run/'verification.json').read_text())
    assert result['state']=='complete' and verification['passed']
    calibration=json.loads((run/'calibration_diagnostic.json').read_text())
    order_check=json.loads((run/'query_order_discrepancies.json').read_text())
    data=json.loads((run/'data_audit.json').read_text());base=Path(data['source_manifest']['source'])
    meta=json.loads((base/'manifest.json').read_text())
    for name,digest in meta['files'].items():assert sha(base/name)==digest,(name,'data changed')
    source_rows=[]
    for p in sorted((run/'source_dependencies').rglob('*.py')):
        rel=p.relative_to(run/'source_dependencies');original=SOURCE/'source_dependencies'/rel
        assert original.exists(),rel
        same=sha(p)==sha(original)
        note='Identical original source'
        if not same:
            assert p.name=='finetune_final.py'
            for name in ['load_data','proposed_data']:assert function_ast(p,name)==function_ast(original,name)
            note='Only DistPFN output selection/naming changed; load_data and proposed_data AST identical'
        source_rows.append(dict(file=str(rel),whole_file_identical=same,diagnostic_dependencies_unchanged=True,note=note))
    auxiliary_sources={}
    for p in [Path(__file__),Path(__file__).with_name('audit_calibration.py')]:
        dest=run/'source'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        auxiliary_sources[str(p.relative_to(_root))]=sha(p)
    write(run/'extended_verification.json',dict(passed=True,original_data_files_checked=len(meta['files']),
        source_code_audit=source_rows,original_data_hashes=meta['files'],auxiliary_sources=auxiliary_sources,
        original_query_group_check=order_check))
    jobs={j['condition']:j for j in result['jobs']}
    labels={'saved_proposed':'저장된 제안 모델 전체 재추론','rebuilt_proposed':'1~6단계를 다시 만든 제안 모델',
            'fresh_global_695':'동일 Global context의 새 단일 PFN','fresh_full_1700':'전체 context의 새 단일 PFN'}
    rows=[[labels[key],f"{j['context_rows']:,}",f"{100*j['metrics']['macro_f1']:.5f}",j['calls'],j['reference_prediction_mismatches']] for key,j in jobs.items()]
    first=jobs['rebuilt_proposed'];control=jobs['fresh_global_695'];full=jobs['fresh_full_1700']
    names=data['class_names'];C=len(names)
    detail=[];class_rows=[]
    selected=np.asarray(json.loads((run/'data/toniot/plans.json').read_text())[-1]['released']);ty=np.load(base/'train_y.npy',mmap_mode='r')
    counts=np.bincount(ty[selected],minlength=C)
    lookup={k:{c['name']:c for c in j['metrics']['classes']} for k,j in jobs.items()}
    for c in sorted(first['metrics']['classes'],key=lambda c:(-c['support'],c['name'])):
        name=c['name'];i=names.index(name);f=lookup['fresh_full_1700'][name]
        class_rows.append([name,f"{c['support']:,}",data['roles']['global']['class_counts'][i],int(counts[i]),
            f"{100*c['f1']:.2f}",f"{100*f['f1']:.2f}",f"{100*(f['f1']-c['f1']):+.2f}"])
        detail.append(dict(class_name=name,test_support=c['support'],global_context=data['roles']['global']['class_counts'][i],full_context=int(counts[i]),
            proposed_f1=c['f1'],same_global_pfn_f1=lookup['fresh_global_695'][name]['f1'],full_pfn_f1=f['f1'],
            proposed_precision=c['precision'],proposed_recall=c['recall'],full_pfn_precision=f['precision'],full_pfn_recall=f['recall']))
    dest=_root/'docs/research/20261016';dest.mkdir(parents=True,exist_ok=True)
    with (dest/'exp82_per_class.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
    same=next(c for c in result['comparisons'] if c['a']=='rebuilt_proposed' and c['b']=='fresh_global_695')
    repeated=next(c for c in result['comparisons'] if c['a']=='saved_proposed' and c['b']=='rebuilt_proposed')
    gate=result['threshold_audit']
    text=[
        '# ToN 소량 라벨 제안 모델 재현 및 context 분할 진단',
        '',
        '사용자는 7페이지 ToN 제안 모델 52.89%의 이상 여부와 재실행을 요청했다. EXP82는 v7 안에서 동일 seed 43, 동일 도입 순서·1,700개 라벨·역할별 행 ID·원본 v3 checkpoint를 유지했다. 저장 모델 전체 재추론, 6개 단계 전체 재구축, 동일 Global context 단일 PFN, 전체 context 단일 PFN을 사전에 정한 네 조건으로 실행했다. IDS 가중치 FT·추가 seed·threshold 조정은 수행하지 않았다. 모든 최종 모델은 동일한 2,271,723개 test를 seed 43의 행 인덱스 순열로 섞어 8,192개씩 평가했다.',
        '',
        table(['조건','최종 context 수','Macro-F1 (%)','expert 호출','기존 참조 예측과 불일치'],rows),
        '',
        '참조 예측은 앞의 세 조건에서 EXP77 제안 모델 원본 순서 결과이며, 전체 context 조건은 EXP81 혼합 순서 PFN이다. 전체 Macro-F1은 마지막 단계의 단일 모델이 전체 test를 예측한 값이며 단계 평균이 아니다.',
        '',
        f"같은 혼합 순서에서 저장 모델 대 재구축 모델은 {repeated['prediction_mismatches']:,}행, 재구축 제안 모델 대 동일 Global 단일 PFN은 {same['prediction_mismatches']:,}행이 다르다. 후자의 최대 확률 차이는 {same['max_probability_difference']:.9g}다. 원본 순서와 혼합 순서에서 예측이 달라지는 행은 원본 대조 수로 별도 표시했다. 반올림한 발표 점수는 동일하고, 이 변동은 전체 context 대조군과의 13%p 이상 차이를 설명하지 않는다.",
        f"원본 대조에서 달라진 {order_check['changed_rows']}행을 원본 query 배치에 넣어 다시 예측한 결과, 원본 예측 전체 재현 여부는 {order_check['original_group_reproduces_original']}다. 해당 행의 최대 클래스 확률 변화는 {max(r['max_probability_difference'] for r in order_check['rows']):.6f}이므로 단순한 기계 정밀도 수준의 오차라고 단정하지 않는다. 배치 구성에 따른 내부 변화 원인은 이번 진단에서 분리하지 않았다. 이 확인에는 test 정답 라벨을 사용하지 않았다. 같은 혼합 순서에서 재구축·복원·단일 Global 대조는 확률까지 정확히 일치했다.",
        '',
        '최종 Global은 695개, expert pool은 331개, route는 353개, tune은 123개, cal은 198개다. 모든 클래스가 모든 역할에 존재하고, 역할 간 feature group과 학습/test 행 ID가 분리되어 있다. 클래스 ID 매핑은 전체 10개 클래스 순서와 일치한다. 원본 데이터 8개 파일 hash와 사용 코드도 확인했다. finetune_final.py의 기존 변경은 DistPFN 출력 조건/명명뿐이며 이 진단의 데이터 구성 함수는 원본과 AST가 동일하다.',
        '',
        f"최종 cal {gate['calibration_rows']}개에서 후보 {gate['candidate_threshold_pairs']}개 모두의 정답 여부 변경(helpful+harmful)은 최대 {gate['maximum_helpful_plus_harmful']}건이다. 기존 정책이 요구하는 최소 {gate['minimum_helpful_plus_harmful']}건에 미달해 {gate['candidate_pairs_failing_support']}개 후보가 support 조건을 통과하지 못했고 Global fallback이 선택됐다. 실제 test expert 호출도 0건이다. 이는 expert가 적용된 앙상블의 이득을 검증한 결과가 아니라 695개 Global context로 동작한 현재 설정의 결과다.",
        '',
        'expert 4개의 공통 anchor는 각각 Global 전체 695개이고, 전문 잔차 블록은 각각 5·34·4·288개다. 앞의 세 expert는 Global에 몇 개의 사례만 덧붙인 구성이므로 전문 사례가 차지하는 비중이 매우 작다. 아래는 원래 보류한 cal 198개에서의 직접 비교이며 test로 모델을 선택한 결과가 아니다.',
        '',
        table(['expert','공통 anchor','전문 블록','Global 오답 교정','Global 정답 훼손'],
              [[e['expert'],e['anchor_rows'],e['residual_rows'],e['helpful'],e['harmful']] for e in calibration['experts']]),
        '',
        f"어느 expert라도 Global의 오답을 맞힌 cal 사례는 {calibration['any_expert_can_correct_global_error']}개뿐이다. 이는 정답을 이용한 진단용 상한이며 실제 routing 성능이 아니다. scorer가 선택한 후보는 교정 {calibration['scorer_selected_candidate_helpful']}개, 훼손 {calibration['scorer_selected_candidate_harmful']}개다. 따라서 30건 기준만 낮추면 문제가 해결된다는 근거도 없다. 작은 전체 예산에 맞는 Global·공통 anchor·전문 블록 배분을 함께 설계할 필요가 있다.",
        '',
        '클래스별 F1 및 context 수. 범위는 최종 전체 test, 정렬은 test 표본 수 내림차순이다.',
        '',
        table(['클래스','test 수','Global context','전체 context','제안 F1 (%)','전체 PFN F1 (%)','차이 (%p)'],class_rows),
        '',
        '대조군이 같은 결과를 재현하는 범위에서는, Global context의 배분과 expert 미작동이 현재 결과를 직접 설명한다. 전체 1,700개로 바꾸는 비교는 context 개수뿐 아니라 사례 구성과 클래스 비율도 함께 바꾸므로 이를 단순한 표본 수 효과 하나로 단정하지 않는다. 추가 라벨 없이 같은 1,700개를 활용한 단일 PFN 결과는 개선 설계의 기준선이며, 제안 모델이 그 점수로 개선됐다고 간주하지 않는다.',
        '',
        '개선 방향은 소량 라벨을 5개 고정 역할로 분할하면서 Global의 정보를 크게 잃는 구성을 바꾸는 것이다. 우선 feature group 기준 교차 적합으로 각 검증 행을 context에서 제외한 out-of-fold 예측을 만들고, 그 예측으로 routing·채택 기준을 학습/선택한 뒤 최종 Global이 누적 사례를 활용하도록 구성할 수 있다. S/V 학습과 threshold 선택 사이의 분리도 유지해야 하며, 최종 context 확대에 따른 확률 분포 변화를 별도로 검증해야 한다. 고정 30건 기준의 소량 데이터 적합성도 학습/검증 자료에서 사전 결정할 대상이다. 이번 test를 보고 기준을 낮추거나 호출을 강제한 실험은 하지 않았다. 이 변경은 아직 실행·검증하지 않은 후속 설계안이다.',
        '',
        '7페이지 점수는 재현 여부에 따라 유지할 수 있으나, 이를 제안 앙상블의 잠재 성능 전체나 최종 완성 모델로 해석하지 않는다. 현재 설정의 결과임을 밝히고 Global 695개와 expert 호출 0건을 함께 설명한다. 기존 슬라이드·HTML·원고는 이번 진단에서 수정하지 않았다.',
        '',
        f"근거: [실행 결과](../../../{run.relative_to(_root)}/results.json), [실행 검증](../../../{run.relative_to(_root)}/verification.json), [원본 데이터·코드 검증](../../../{run.relative_to(_root)}/extended_verification.json), [실험 설계](../../../configs/experiments/exp82_protocol.json), [클래스별 CSV](exp82_per_class.csv).",
    ]
    (dest/'exp82_results.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(verification='passed',results=rows,comparisons=result['comparisons'],threshold_audit=gate),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
