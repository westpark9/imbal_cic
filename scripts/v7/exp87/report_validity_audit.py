"""Record the 2026-10-27 review without modifying prior runs or manuscripts."""
from pathlib import Path
import sys
_root = next(p for p in Path(__file__).resolve().parents
             if (p / 'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root / 'scripts/common'))
from experiment_paths import repo_root
from lablog import write_record, build_report, check
from audit_fewshot_validity import read, write, sha, one

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = repo_root(__file__)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
                     + ['| '+' | '.join(map(str, row))+' |' for row in rows])


def mean_sd(values):
    a = np.asarray(values) * 100
    return f'{a.mean():.2f} ± {a.std(ddof=1):.2f}'


def main():
    audit = one('results/v7/exp87/*fewshot_validity*')
    restore = one('results/v7/exp87/*original_test_population*')
    assert read(audit/'COMPLETE.json')['diagnostic_fits'] == 110
    assert read(restore/'COMPLETE.json')['jobs'] == 10
    cfg = read(audit/'config.json')
    curve = read(Path(cfg['source_curve'])/'results.json')['jobs']
    exp86 = Path(cfg['source_exp86'])
    old = read(exp86/'results.json')['jobs']
    diagnostics = read(audit/'diagnostics.json')
    restored = read(restore/'results.json')
    overlap = read(audit/'overlap.json')
    secondary = read(audit/'secondary_metrics.json')
    population = read(restore/'population.json')
    evidence = ROOT/'docs/lablog/evidence/20261027'
    evidence.mkdir(parents=True, exist_ok=True)
    static = []
    for file in ['scripts/v7/exp86/attack_fewshot.py', 'scripts/v7/exp71/exp71_bounded_updates.py',
                 'scripts/v7/exp72/replay_models.py', 'scripts/v7/exp72/replay_protocol.py',
                 'scripts/common/preprocess_nfv3_energy_suite.py']:
        original = exp86/'source_dependencies'/file
        assert sha(original) == sha(ROOT/file)
        static.append(dict(path=file, source_snapshot_matches=True, sha256=sha(original)))
    memberships = []
    for ds in ['cic2018', 'toniot']:
        md = read(exp86/'data'/ds/'manifest.json')
        clean = Path(md['clean_manifest']).parent
        ids = np.sort(np.load(clean/'train_idx.npy'))
        for j in old+curve:
            if j['dataset'] != ds:
                continue
            chosen = np.asarray(j['train_source_ids'])
            pos = np.searchsorted(ids, chosen)
            assert (pos < len(ids)).all() and np.array_equal(ids[pos], chosen)
        memberships.append(dict(dataset=ds, all_selected_rows_in_clean_train=True))
    sources = [dict(url='https://staff.itee.uq.edu.au/marius/NIDS_datasets/',
                    accessed='2026-10-27', supports='NF-v3 extraction variant has 53 features; differs from CICFlowMeter representation'),
               dict(url='https://www.unb.ca/cic/datasets/ids-2018.html',
                    accessed='2026-10-27', supports='Original attack collection used specified attacker/victim machines and time intervals')]
    info = dict(analysis_date='2026-10-27', report_date='2026-10-27',
                original_execution_dates=['2026-10-21','2026-10-22'],
                run_paths=[str(audit.relative_to(ROOT)), str(restore.relative_to(ROOT))],
                static_checks=static, train_membership_checks=memberships,
                source_urls=sources, conclusions_adoption='exploratory_not_manuscript_adopted',
                limitations=['PFN predictions recomputed from saved outputs, but no new PFN model fit in this audit',
                             'Original-population restoration evaluated XGB only',
                             'Original population still attack-only; training retains global curation',
                             'Host/session/campaign disjointness and near duplicates remain untested'])
    write(evidence/'exp87_validity_audit.json', info)

    curve_rows=[]
    for n in [2,5,10,20,80,450]:
        cells=[]
        for m in ['xgb','pfn']:
            js=[j for j in curve if j['dataset']=='cic2018' and j['n']==n and j['method']==m]
            cells.append(mean_sd([j['metrics']['macro_f1'] for j in js]))
        curve_rows.append([n,*cells])
    restoration_rows=[]
    for n in [2,20]:
        js=[j for j in restored if j['n']==n]
        restoration_rows.append([n,mean_sd([j['retained']['macro_f1'] for j in js]),
                                    mean_sd([j['original']['macro_f1'] for j in js]),
                                    f"{np.mean([j['original']['macro_f1']-j['retained']['macro_f1'] for j in js])*100:.2f}"])
    ablation_rows=[]
    labels={'all_features':'전체 46개', 'proxy_only':'프로토콜·호스트 관련 후보 11개',
            'without_proxy':'후보 11개 제외, 35개', 'ttl_only':'TTL 2개만',
            'without_ttl':'TTL 제외, 44개', 'depth3_tree':'깊이 3 결정트리, 전체 46개'}
    for variant,label in labels.items():
        ablation_rows.append([label,*[mean_sd([j['macro_f1'] for j in diagnostics if j['n']==n and j['variant']==variant]) for n in [2,20]]])
    selected=sorted(overlap['classes'],key=lambda j:j['test_rows'],reverse=True)
    redundancy_rows=[[j['name'],f"{j['test_rows']:,}",f"{j['unique_vectors']:,}",
                      f"{j['any_train_overlap_rows']/j['test_rows']*100:.2f}"] for j in selected]
    class_rows=[]
    for cls in selected:
        cells=[]
        for n in [2,20,450]:
            for m in ['xgb','pfn']:
                js=[j for j in curve if j['dataset']=='cic2018' and j['n']==n and j['method']==m]
                cells.append(f"{np.mean([next(c['f1'] for c in j['metrics']['classes'] if c['name']==cls['name']) for j in js])*100:.2f}")
        class_rows.append([cls['name'],*cells])

    # Exportable scientific figure; no retained report PDF.
    fig, axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for key,label,color in [('retained','Filtered test','#2369a1'),('original','Original attack test','#bd5837')]:
        vals=[[j[key]['macro_f1']*100 for j in restored if j['n']==n] for n in [2,20]]
        axes[0].errorbar([0,1],[np.mean(v) for v in vals],yerr=[np.std(v,ddof=1) for v in vals],
                         marker='o',capsize=3,label=label,color=color)
        for x,v in enumerate(vals):axes[0].annotate(f'{np.mean(v):.2f}',(x,np.mean(v)),xytext=(4,8),textcoords='offset points')
    axes[0].set_xticks([0,1],['2','20']);axes[0].set_xlabel('Training rows per attack class')
    axes[0].set_ylabel('Attack Macro-F1 (%)');axes[0].set_ylim(35,90);axes[0].set_title('Same XGB models, changed test population')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    shown=['all_features','proxy_only','without_proxy','ttl_only']
    for offset,n,color in [(-.18,2,'#2369a1'),(.18,20,'#bd5837')]:
        vals=[[j['macro_f1']*100 for j in diagnostics if j['n']==n and j['variant']==v] for v in shown]
        axes[1].bar(np.arange(len(shown))+offset,[np.mean(v) for v in vals],.36,
                    yerr=[np.std(v,ddof=1) for v in vals],capsize=2,color=color,label=f'n={n}')
    axes[1].set_xticks(np.arange(4),['All 46','Candidate\n11 only','Other 35','TTL only'])
    axes[1].set_ylim(0,100);axes[1].set_ylabel('Attack Macro-F1 (%)');axes[1].set_title('Feature diagnostics on filtered test')
    axes[1].legend();axes[1].grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'docs/lablog/figures/20261027_exp87_validity.pdf');plt.close(fig)
    pd.DataFrame(diagnostics).drop(columns=['confusion','per_class_f1']).to_csv(ROOT/'docs/lablog/data/20261027_exp87_diagnostics.csv',index=False)

    shuffled=[j['macro_f1'] for j in diagnostics if j['variant'].startswith('label_permutation')]
    paragraph=f"""# EXP86–87 극소수 학습의 데이터·특징·구현 타당성 점검

## 목적과 핵심 판단
평가 모집단의 전역 라벨 기반 정제는 현재 CIC2018 소수 표본 성능을 높이는 중요한 요인이다. 같은 XGBoost (XGB) 모델의 test만 정제 전으로 복원하면 클래스당 2개에서 Macro-F1이 63.24%에서 50.52%로, 20개에서 80.43%에서 62.89%로 내려간다. 이 진단은 데이터가 원래 쉽다는 설명, 특정 특징에 대한 의존, 구현 오류를 구분하기 위한 검토이며 모델 우위나 논문 novelty를 확정하지 않는다. 현재 승인된 연구 버전 v7과 EXP87을 유지한다.

분석·실행·작성일은 2026-10-27이며 원 실험 실행일은 2026-10-21~22다. 첫 실행 {audit.name}은 기존 300조건 재검산, 저장 XGB 10개 재생, 특징 및 라벨 대조 110개를 완료했다. 두 번째 실행 {restore.name}은 저장 XGB 10개의 원래 공격 test 복원 평가를 완료했다. 이 보고서는 EXP86–87의 해석 범위를 보완하며 기존 원시 결과를 덮어쓰지 않는다.

## 평가 대상과 샘플 수 해석
공격 종류 분류가 현재 곡선의 과제이며 정상과 공격을 구분하는 전체 탐지 성능과 구별한다. 실제 입력은 원본 CICFlowMeter CSV가 아니라 NF-CICIDS2018-v3에서 식별자 등을 제외한 46개 특징이고, UQ 공식 문서는 NF-v3와 CICFlowMeter 표현을 별개로 설명한다. [공식 데이터 설명](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)

CIC2018은 6개 공격 family로 묶고 모든 family에 같은 n을 제공하므로 n=2의 학습 행 수는 총 12개, n=20은 120개다. 정상 학습 행은 0개이며 EXP87의 test 역시 공격만 390,270개다. EXP86의 정상 포함 평가는 정상 출력 라벨이 없어서 정상 오탐률 100%인 진단이었다. 원래 공격 세부 유형을 family로 묶었으므로 클래스당 2개가 각 공격 시나리오당 2개를 뜻하지 않는다.

## 학습곡선과 클래스별 난이도
2개에서 20개까지는 두 모델 모두 개선되며, 이후 증가폭이 줄고 특정 클래스에 성능이 집중된다. 아래 값은 같은 고정 공격 test에서 seed 43~47의 평균과 표본 표준편차이며 신뢰구간이나 통계적 동등성 판정이 아니다. 450은 가장 작은 train family의 가용 표본 상한이며 수렴이 입증된 지점이 아니다.

{table(['공격당 n','XGB Macro-F1 (%)','TabPFN Macro-F1 (%)'],curve_rows)}

Bot과 brute-force는 2개부터 높지만 infiltration과 web-attacks는 그렇지 않다. 특히 web-attacks는 450개에서도 낮은 F1이 남아 전체 평균만으로 난이도나 소수 클래스 문제 해결 여부를 판단하면 안 된다. 아래 클래스는 평가 표본 수 내림차순이며 각 칸은 5개 seed 평균 F1(%)다.

{table(['클래스','XGB 2','PFN 2','XGB 20','PFN 20','XGB 450','PFN 450'],class_rows)}

## 전역 정제와 원래 test 복원
정제 코드가 train·validation·test를 합친 특징과 라벨로 충돌 벡터를 결정했다. `cic2018_conflict_clean.py`의 `prepare`는 모든 split의 ID를 합쳐 `conflict_mask`를 호출하고, 같은 특징에 여러 family 라벨이 있으면 해당 벡터의 모든 행을 제거한다. 따라서 단순한 train 내부 중복 제거가 아니라 test 라벨에 의존한 모집단 선택이다. 이는 fit에 test 라벨을 직접 전달하는 오류와는 구별되지만, 엄격한 소수 라벨 예산 밖의 정보 사용이며 원본 모집단에 대한 성능 근거로 그대로 사용할 수 없다.

전체 CIC2018 20,115,529행 중 5,360,112행이 제거됐고, 공격 test는 520,188행에서 390,270행으로 줄었다. FTP-Bruteforce와 SlowHTTPTest는 모든 split에서 전부 제거됐으며, test의 brute-force는 SSH만 남았다. test의 XSS 96행도 전부 제거됐고 web-attacks는 509행에서 127행으로 줄었다. 원래 라벨 충돌을 데이터 오류라고 단정할 수 없으며, 46개 표현이 서로 다른 사건을 같은 값으로 매핑했을 가능성과 원 라벨 품질을 별도로 검토해야 한다.

고정 모델의 평가 대상만 바꾼 결과가 아래 표다. 원래 source pickle의 SHA-256을 확인했고, 남아 있던 test 행의 ID·특징·라벨이 기존 cache와 정확히 같음을 검증했다. 정제 전 공격 520,188개 전체에서 예측했으며 기존 390,270개에 대한 예측도 완전히 동일했다. 따라서 표의 변화는 이 고정 모델에 대해 평가 대상 복원으로 발생한 차이이며, train 정제를 없앴을 때의 학습 효과를 측정한 것은 아니다.

{table(['공격당 n','정제 후 test (%)','정제 전 test (%)','복원 후 변화 (pp)'],restoration_rows)}

Brute-force의 n=20 평균 F1은 99.88%에서 49.32%로 내려가며, DoS는 96.66%에서 64.32%로 바뀐다. 이 진단에는 정상 test 복원이나 TabPFN의 복원 평가가 포함되지 않았다. 학습 표본은 여전히 전역 정제 pool에서 선택됐으므로 이 수치를 누수 없는 최종 benchmark 결과로 채택하지 않는다.

![같은 모델의 평가 모집단 변경과 특징군 대조. 오차막대는 5개 seed 표본 표준편차다.](../../figures/20261027_exp87_validity.pdf)

## 특징 의존과 중복 진단
명시적인 Label·Attack·IP·포트·절대 timestamp·DNS transaction ID는 46개 입력에서 제외됐다. 값 채우기와 clipping은 행별 고정 규칙이며, 이번 코드 검토에서 전체 train으로 학습한 scaler나 supervised feature selection을 소수 모델에 전달하는 경로는 확인하지 못했다. 다만 출발점인 전역 정제는 앞 절의 문제를 가진다.

프로토콜과 호스트 환경 관련 후보 11개는 PROTOCOL, L7_PROTO, MIN_TTL, MAX_TTL, TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT, ICMP_TYPE, ICMP_IPV4_TYPE, DNS_QUERY_TYPE, DNS_TTL_ANSWER, FTP_COMMAND_RET_CODE다. Time to Live (TTL) 두 열만 쓰는 조건과 11개 전체를 쓰는 조건, 각각 제거하는 조건을 비교했다. 같은 train ID와 전체 고정 공격 test를 쓰고 XGB 설정은 원래 조건과 같으며, 깊이 3 결정트리만 별도 단순 모델이다. 특징을 제거하면 기존 열 subsampling의 실제 조합도 달라지므로 개별 변수의 인과적 중요도를 추정한 것으로 해석하지 않는다.

{table(['특징/모델 조건','n=2 F1 (%)','n=20 F1 (%)'],ablation_rows)}

후보 11개만으로 n=2에서 61.04%, n=20에서 78.60%에 도달하므로 이 특징군은 높은 예측 정보를 가진다. 그러나 이를 제외해도 62.53%와 76.91%가 남고 TTL만으로 전체 성능을 재현하지는 못한다. 따라서 특정 한 열의 누수라고 단정하기보다 여러 상관 특징과 반복된 공격 환경이 공유하는 단서를 검토해야 한다. UNB 공식 설명에는 공격자·피해자 장비와 공격 수행 시각이 정해진 수집 구조가 제시됐지만, 실제로 그 환경 단서를 학습했는지는 host/session을 분리한 실험이 필요하다. [원 데이터 수집 설명](https://www.unb.ca/cic/datasets/ids-2018.html)

동일 특징 벡터의 반복은 크지만, 실제 2개 context와의 정확한 중복만으로 점수가 설명되지는 않는다. 390,270개 공격 test는 212,382개 고유 특징 벡터이며, 123,477개는 전체 clean train에 같은 벡터가 존재한다. 이는 전체 pool 중복률이고 소수 모델이 그 모든 행을 봤다는 뜻이 아니다. 아래 비율의 분모는 각 클래스 test 행 수이며 해시는 기존 64-bit 값을 사용해 근접 중복은 측정하지 않았다.

{table(['클래스','test 행 수','고유 벡터 수','전체 train과 일치 (%)'],redundancy_rows)}

Bot의 41,542개 test가 1,386개 고유 벡터에 집중돼 있어 명목 행 수와 독립적인 정보량을 구별해야 한다. EXP86에서 각 실제 n=2 context와 겹치는 test를 제외한 평균은 XGB 62.80%, TabPFN 65.94%로, 원래 63.24%와 66.34%에 가깝다. 각 n에서 test를 다르게 제외하는 문제를 피하기 위해 5개 seed의 최대 n=450 context 합집합과 겹치는 행을 한 번만 제거한 327,753개 고정 패널도 계산했다. 이 패널의 n=2 평균은 XGB 61.10%, TabPFN 66.12%였고, n=450은 84.87%, 84.56%였다. 다만 이 패널은 클래스별 구성과 난이도를 바꾼 보조 진단이며 원래 test나 외부 일반화 결과를 대체하지 않는다.

## 구현 재검산과 음성 대조
기존 300개 조건의 저장 예측을 독립적인 sklearn confusion matrix로 다시 집계했고 모든 Macro-F1과 confusion matrix가 일치했다. train ID 개수와 고유성, 클래스별 n, 정상 행 0개, 실제 clean train 소속, test ID 비중복과 예측 파일 해시를 확인했다. CIC2018의 모델 간 학습 ID 일치와 n별 중첩도 확인했다. 실행 당시 snapshot과 현재 학습·평가 핵심 코드 5개가 동일했다.

저장 XGB 10개를 CPU에서 재생한 결과와 같은 설정으로 새로 fit한 10개 CUDA XGB의 결과가 모두 원래 예측과 정확히 일치했다. 따라서 이번 범위에서는 n이 무시되거나 전체 train이 fit에 들어가거나 저장 지표가 잘못 집계되는 문제를 재현하지 못했다. TabPFN은 저장 예측과 context 경로, 기존 checkpoint 사용 및 당시 가중치 해시 검사를 확인했으며 새 TabPFN fit을 이번에 수행하지는 않았다. 라이브러리 전체와 모든 잠재 오류의 부재까지 입증한 것은 아니다.

학습 라벨만 섞은 n=2 XGB 대조는 seed당 10회, 총 50회 수행했고 평균 {np.mean(shuffled)*100:.2f}%의 Macro-F1을 보였다. 원래 63.24%보다 크게 낮아 특징과 올바른 라벨의 대응이 점수에 필요함을 보여준다. 50회는 5개 학습 표본 집합 안에서 중첩된 permutation이므로 50개의 독립 데이터 반복으로 간주하지 않으며, 단순 1/6을 Macro-F1의 우연 기준으로 사용하지 않는다.

XGB의 n=1 조건은 기존 기록에서 두 데이터의 모든 seed에 split 수가 0이었다. 따라서 해당 PFN 우위는 고정 XGB 설정의 극소 표본 분할 제약과 분리해야 한다. n=2의 CIC2018에서는 519~589개의 실제 split이 있었으므로 n=1의 퇴화 현상을 그대로 적용할 수 없다.

## 다음 검증과 연구 방향
평가 파이프라인의 우선 개선은 test 라벨로 정제 여부를 결정하지 않는 것이다. 분할을 먼저 고정하고 label-dependent 정제는 허용된 train 라벨에 한정하며, 극소 라벨 실험에서는 선택된 n개 밖의 라벨 사용도 예산으로 따로 밝혀야 한다. 원래 test를 주 평가로 유지하고 충돌 행 포함/제외 성능은 분석용으로 병기한다. 특징이 같지만 라벨이 다른 사건은 삭제만으로 해결하지 말고 raw flow·세션 근거, 라벨 품질, 특징 표현의 한계를 확인한다.

두 번째 개선은 host·session·capture/campaign 단위로 train과 test가 분리되는 평가다. 현재 분할은 시나리오마다 시간순 60/20/20이므로 같은 공격 실행 환경이 양쪽에 남을 수 있다. 원본 CSV의 주소·포트·시각은 모델 입력이 아니라 그룹 구성에만 사용하고 그룹 간 겹침과 각 공격의 지원 수를 먼저 확인한다. 단순한 날짜 분할로 어떤 클래스를 train에서 완전히 없애면 현재 few-shot 과제가 바뀌므로 최소 라벨 확보 조건을 함께 명시한다. 이 작업은 근접 중복 및 동일 공격 캠페인 일반화를 확인하기 위한 다음 실험이며 이번에 완료한 것으로 기록하지 않는다.

소수 라벨의 가치는 동일한 정보 예산에서 목표 탐지 품질에 필요한 라벨 수로 검증한다. 정상 행이 충분하고 신규 공격만 희소한 조건과 모든 클래스가 희소한 조건을 구분하고, XGB도 극소 표본에 적합한 분할 제약을 개발 데이터에서 조정한 대조를 포함한다. 정상 포함 false-positive rate와 클래스별 recall/F1, 모델 준비·추론 비용을 함께 비교하되 test 결과를 보고 유리한 특징이나 클래스만 선택하지 않는다. 정제와 분리 조건을 바로잡은 뒤에도 일관된 라벨 효율이 남아야 제안 방법의 실증 근거로 검토할 수 있다.

## 재현 근거와 완료 범위
감사 실행의 원시 예측·선택 ID·모델·설정은 `{audit.relative_to(ROOT)}`에 있고 원래 test 복원 결과는 `{restore.relative_to(ROOT)}`에 있다. 첫 실행의 `artifact_checks.csv`, `diagnostics.json`, `overlap.json`, `secondary_metrics.json`과 두 번째 실행의 `population.json`, `results.json`이 수치 근거다. `docs/lablog/evidence/20261027/exp87_validity_audit.json`은 출처와 추가 검증을 연결하며 생성 코드는 `scripts/v7/exp87/report_validity_audit.py`다. 준비·추론 비용을 공정 비교하는 실험이 아니므로 이번 진단의 wall time을 기존 비용 곡선에 섞지 않는다. 실험 완료와 논문 근거 채택을 구별하고 원고와 발표자료는 수정하지 않았다.
"""
    target=write_record(ROOT/'docs/lablog/evidence/20261027/exp87_validity',paragraph,
        summary='EXP86–87 타당성 감사: 300조건 재검산·XGB 10개 재생·110개 대조 완료; 원래 공격 test 복원 시 XGB 2/20개 F1 63.24/80.43→50.52/62.89%; 전역 정제 영향 확인',compile_pdf=False)
    validation=build_report(target)
    validation['structure']=check()
    assert validation['structure']['passed'],validation['structure']
    write(evidence/'exp87_validity_validation.json',validation)
    print(json.dumps(validation,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
