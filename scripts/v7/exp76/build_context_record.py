#!/usr/bin/env python3
"""Narrative research record and machine-readable evidence, without HTML edits."""
from pathlib import Path
import sys,json,argparse,hashlib
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0,str(_root/'scripts/common'))
from experiment_paths import repo_root
from lablog import write_record
ROOT=repo_root(__file__)

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve()
 data=json.loads((run/'results.json').read_text());verification=json.loads((run/'verification.json').read_text())
 if data['state']!='complete' or not verification['complete'] or not verification['passed']:raise ValueError('Complete verified results required')
 jobs={(j['dataset'],j['method']):j for j in data['jobs']}
 out=ROOT/'docs/lablog/evidence/20261015';lines=['# EXP76 · 소규모 context 갱신 정책의 품질과 비용','',
 '사용자가 요청한 누적·고정 버퍼·갱신 없음의 차이를 확인하기 위해, 1014와 같은 공격 클래스 도입 순서와 기존 test 전체를 사용했다. 초기 1,000개(Benign 750, 최초 두 공격 각 125)에서 시작해 신규 클래스마다 라벨된 100개를 추가했다. 이 수치는 이번 실행의 설정이며 사용자의 1,000·20 예시를 고정 요구로 해석한 것이 아니다. 최종 공개 사례는 CIC2018 1,400개, ToN 1,700개다. TabPFN-v3의 가중치는 고정하고 context와 캐시를 갱신했다. LoCalPFN은 요청에 따라 제외했다.','',
 '고정 버퍼는 1,000개를 유지하며 FIFO·신규 사례와의 유사도·무작위로 기존 사례를 제거했다. 모든 삭제 정책은 기존 클래스마다 최소 1개를 보존했다. FIFO는 실제 노후화 판정이 아니라 확보 순서의 대리 기준이다. XGB와 BoostPFN은 동일하게 공개된 누적 사례로 다시 학습·구성했고, DistPFN은 누적 PFN의 준비 비용과 확률을 공유하며 현재 배치 posterior로 보정했다.','',
 '## 전체 결과','',
 '평가는 지연 없이 갱신된 버전을 사용하는 순차 도입 재생이다. 갱신 준비 시간은 별도 실측했으며, 이 시간을 예측 버전 배정에 적용한 1014의 지연 재생과 구별한다. 모든 원 test 행을 방법별 주 평가에 한 번 사용했다. 배치는 8,192개, seed 43, RTX 4090 단독 worker다. 후속 갱신 합계는 초기 구축을 제외하고, 추론 시간은 전체 test의 실제 serving 합계다.','',
 '| 데이터 | 방법 | Macro-F1 (%) | Benign FPR (%) | 초기 구축 (s) | 후속 갱신 합계 (s) | 전체 추론 (s) |','|---|---|---:|---:|---:|---:|---:|']
 for j in data['jobs']:
  lines.append(f"| {j['dataset']} | {j['method']} | {100*j['metrics']['macro_f1']:.2f} | {100*j['metrics']['benign_false_positive_rate']:.3f} | {j['initial_seconds']:.2f} | {j['update_seconds']:.2f} | {j['serving_seconds']:.2f} |")
 lines+=['','## 관찰에서 이어지는 개선 방향','']
 for ds,label in [('cic2018','CIC2018'),('toniot','ToN')]:
  x=jobs[ds,'xgb'];a=jobs[ds,'pfn_accumulate'];sim=jobs[ds,'pfn_similar'];dist=jobs[ds,'distpfn']
  lines += [f"{label}의 누적 TabPFN Macro-F1은 {a['metrics']['macro_f1']*100:.2f}%이고 XGB는 {x['metrics']['macro_f1']*100:.2f}%다. 후속 갱신 합계는 각각 {a['update_seconds']:.2f}초와 {x['update_seconds']:.2f}초로, 이 소규모 조건에서는 PFN의 준비 비용이 더 낮았다. 반면 전체 추론은 {a['serving_seconds']:.2f}초와 {x['serving_seconds']:.2f}초였다. 갱신 준비의 절감이 이 전체 workload의 총 연산시간 절감으로 이어졌다고 해석할 수는 없다. 동일 context에서 estimator 수와 cache 위치를 분리해 비교하고, query 처리 비용과 공격별 품질을 함께 측정하는 것이 다음 개선 방향이다.",'',
   f"{label}의 유사도 삭제는 정상 오탐률 {sim['metrics']['benign_false_positive_rate']*100:.3f}%로 누적 PFN의 {a['metrics']['benign_false_positive_rate']*100:.3f}%보다 높았다. 신규 사례와 가까운 기존 사례가 단순히 불필요한 중복이라고 가정하지 않고, 정상·기존 공격의 구분 근거를 남기는 삭제 제약을 검토해야 한다. DistPFN의 F1은 {dist['metrics']['macro_f1']*100:.2f}%였으나, 이는 현재 배치 posterior 보정의 결과이며 다른 배치 크기·유입 비율에서의 안정성은 별도 확인 대상이다.",'']
 lines+=['고정 context는 갱신 비용이 없지만 context에 없는 신규 클래스 이름을 출력할 수 없다. 이에 따른 신규 유형 분류 오류와 기존 공격으로의 이진 공격 탐지는 서로 다른 평가다. 이 기준선의 낮은 신규 유형 F1을 파라미터 최적화의 실패로 해석하지 않는다.','',
 '현재 결과는 같은 기존 test를 사용한 단일 seed의 탐색 근거다. 1014의 10만 라벨과 이번 소규모 예산은 정보량·초기 배분·배치 크기가 달라 직접적인 개선량으로 비교하지 않는다. 두 데이터의 자연 분포에서 충분한 품질을 확보했다는 운영 기준이나 통계적 동등성은 사전 정의하지 않았다. 원고의 최종 모델 선택·성능 우위에 자동 편입하지 않으며, 같은 공개 라벨에서 context 크기만 바꾸는 비교와 반복 seed를 후속으로 검토한다.','']
 for ds,label in [('cic2018','CIC2018'),('toniot','ToN')]:
  lines += [f'## {label} 클래스별 F1','', '전체 test의 실제 평가 표본 수 내림차순이다. 값은 F1 (%)이며 학습·context 사례 수와 구별한다.','',
   '| 클래스 | 평가 표본 수 | XGB | 누적 PFN | FIFO | 유사도 | 무작위 | 고정 | DistPFN | BoostPFN |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
  methods=['xgb','pfn_accumulate','pfn_fifo','pfn_similar','pfn_random','pfn_fixed','distpfn','boostpfn']
  for c in sorted(jobs[ds,'xgb']['metrics']['classes'],key=lambda v:-v['support']):
   vals=[next(v for v in jobs[ds,m]['metrics']['classes'] if v['name']==c['name']) for m in methods]
   lines.append('| '+c['name']+f" | {c['support']:,} | "+' | '.join(f"{100*v['f1']:.2f}" for v in vals)+' |')
  lines += ['']
 rel=run.relative_to(ROOT)
 lines+=['## 출처와 검증','',f'- 실행: `{rel}`',f'- [원 결과](../../../{rel}/results.json)',f'- [독립 검증](../../../{rel}/verification.json)',
  '- [프로토콜](exp76_protocol.md)', '- [가중치 미변경 확인](tabpfn_fit_mechanism_check.json)',
  '- [원고 반영안](manuscript_revision_proposal.md) · 원고·Overleaf 직접 수정 없음','',
  f"예측으로부터 혼동행렬·Macro-F1·Benign FPR을 독립적으로 재계산했고 {verification['verified_jobs']}개 결과를 검증했다. 전체 test ID의 중복·누락, 공개 전 라벨 사용, 버퍼 크기·클래스 보존, 버린 사례의 재유입, 후속 비용 합계를 확인했다. 전체 공개 학습 특징과 겹치는 test를 제외한 공통 집합의 보조 결과도 원 JSON에 보존했다. 이 보조 집합은 외부 환경 검증을 대체하지 않는다."]
 write_record(out/'exp76_results.md', '\n'.join(lines)+'\n')
 manifest=dict(run=str(rel),state='complete_verified',files={str((run/f).relative_to(ROOT)):hashlib.sha256((run/f).read_bytes()).hexdigest() for f in ['config.json','results.json','verification.json']},adoption='exploratory; not automatically adopted in manuscript',html_updated=False,overleaf_updated=False)
 (out/'exp76_evidence.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(out/'exp76_results.md')
if __name__=='__main__':main()
