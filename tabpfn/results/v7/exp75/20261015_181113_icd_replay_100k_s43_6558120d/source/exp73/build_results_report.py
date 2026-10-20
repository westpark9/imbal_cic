#!/usr/bin/env python3
"""Render the explicitly requested EXP73 results into the existing 1014 tab."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

_root = next(p for p in Path(__file__).resolve().parents
             if (p / 'scripts/common/experiment_paths.py').is_file())
sys.path.insert(0, str(_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root
bootstrap(_root)
from build_protocol_report import HTML, REPORT_ID, publish

ROOT = repo_root(__file__)
METHODS = ['xgb', 'tabpfn', 'ours', 'boostpfn', 'localpfn', 'distpfn']
RUN_NAME = '20261014_154250_replay_current_sota_100k_s43_b0f8a9dc'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k != 'confusion'}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def load_verified(run):
    assert run.name == RUN_NAME, 'Narrative is specific to this approved run.'
    config = read(run / 'config.json')
    assert read(run / 'status.json')['state'] == 'complete'
    verification = read(run / 'verification.json')
    assert verification['state'] == 'passed' and verification['verified_policies'] == 48
    baseline = (run / 'jobs/cic2018_natural_xgb').resolve().parent.parent
    bv = read(baseline / 'verification.json')
    assert bv['state'] == 'passed' and bv['verified_policies'] == 24
    jobs, hashes = {}, {}
    for ds in ['cic2018', 'toniot']:
        for case in ['natural', 'uniform']:
            for method in METHODS:
                key = f'{ds}_{case}_{method}'
                path = run / 'jobs' / key
                result = path / 'results.json'
                assert read(path / 'COMPLETE.json')['results_sha256'] == sha(result), key
                schedule = read(path / 'schedule.json')
                fits = []
                for stage in sorted(path.glob('stage_*')):
                    f = read(stage / 'fit_metrics.json')
                    f = {k: v for k, v in f.items() if k != 'phases'}
                    if method == 'ours':
                        f['selection'] = read(stage / 'selection.json')['thresholds']
                    fits.append(f)
                jobs[key] = dict(results=compact(read(result)['results']),
                                 updates=schedule['updates_seconds'], fits=fits)
                for source in [result, path / 'schedule.json', *path.glob('stage_*/fit_metrics.json')]:
                    hashes[str(source.relative_to(ROOT))] = sha(source)
    # Numeric statements in the prose are checked against their actual scope.
    cu = jobs['cic2018_uniform_ours']['results']['rate_1000']
    tu = jobs['toniot_uniform_ours']['results']['rate_1000']
    assert (cu['policy']['helpful'], cu['policy']['harmful']) == (450, 58)
    assert (tu['policy']['helpful'], tu['policy']['harmful']) == (6898, 245)
    assert next(c for c in tu['policy']['classes'] if c['name'] == 'xss')['helpful'] == 6834
    for ds in ['cic2018', 'toniot']:
        assert jobs[f'{ds}_natural_ours']['results']['rate_1000']['policy']['called'] == 0
    return dict(jobs=jobs, methods=METHODS, run=str(run.relative_to(ROOT)),
                baseline=str(baseline.relative_to(ROOT)), source_hashes=hashes,
                verification=dict(new=48, reused=24), seed=config['seed'])


RESULTS = '''<section id="results"><h2>3. 제안 모델과 SOTA의 성능·비용</h2>
<p>2026-10-15 14:19 완료. 6개 방법 × 두 데이터 × 두 배분의 <strong>24개 설정</strong>을 비교했다.
추가 실행 48개 정책과 재사용 기준선 24개 정책이 각각 검증을 통과했다. 아래 기본값은 <strong>1,000행/초 · 전체 재생 구간</strong>이다.</p>
<div class="controls" aria-label="결과 조건 선택">
<label>평가 정책<select id="policy"><option value="rate_1000">1,000행/초 · 갱신 지연 포함</option><option value="rate_100">100행/초 · 갱신 지연 포함</option><option value="ideal">갱신 지연 없음</option></select></label>
<label>상세 데이터<select id="dataset"><option value="cic2018">CIC2018</option><option value="toniot">ToN</option></select></label>
<label>상세 공격 배분<select id="case"><option value="uniform">균등</option><option value="natural">자연비율</option></select></label>
</div>
<p id="active-scope" class="note" aria-live="polite"></p>
<noscript><p>조건별 표를 표시하려면 JavaScript를 활성화해 주세요. 아래 본문 해석은 1,000행/초 결과를 기준으로 합니다.</p></noscript>
<h3>3.1 전체 분류 성능과 클래스별 차이</h3>
<p class="note">Macro-F1 (%)은 재생 전체 행의 혼동행렬로 계산한다. 마지막 버전 하나의 test 성적이 아니다.
지연 없음도 매 단계 새 버전을 사용하는 동일 재생이며, 전문가를 정답으로 고르는 oracle이 아니다.</p>
<div id="overall"></div>
<div class="grid"><article class="card"><h3>CIC2018</h3><p>1,000행/초에서 단일 TabPFN이 자연 <strong>77.47%</strong>, 균등 <strong>81.17%</strong>로 여섯 방법 중 가장 높다.
제안 모델은 균등에서 <strong>80.35%</strong>이며, 내부 Global 대비 Infiltration 교정이 전체 점수를 높인다.</p></article>
<article class="card"><h3>ToN</h3><p>같은 유입률에서 XGB가 자연 <strong>79.28%</strong>, 균등 <strong>74.46%</strong>로 가장 높다.
제안 모델의 균등 교정은 XSS에 집중되고, 다른 클래스의 F1 감소와 상쇄된다. 희소 공격 개선을 구조의 목표로 다시 점검해야 한다.</p></article></div>
<div id="performance-chart" class="card" aria-label="선택 조건의 모델별 Macro-F1 막대그래프"></div>
<details id="class-details"><summary>선택 조건의 전체 클래스별 F1 · Precision · Recall</summary>
<p class="note">해당 데이터의 전체 Test 표본 수 내림차순. 숫자는 우측 정렬하고 같은 열의 자릿수를 통일한다.
Precision의 FP는 Benign과 다른 공격을 포함한 전체 평가 구간에서 계산한다.</p>
<div id="classwise"></div><details><summary>모델별 Precision·Recall·F1 펼치기</summary><div id="class-pr"></div></details></details>
<details id="stage-details"><summary>단계별 성능과 클래스별 상세 펼치기</summary>
<p class="note">단계 F1은 그때까지 등장한 클래스의 평균이다. 단계 간 클래스 구성이 다르므로 점수 변화만으로 망각을 판단하지 않는다.</p>
<div id="stage-summary"></div><label>평가 구간<select id="stage"></select></label><div id="stage-classes"></div></details>

<h3>3.2 Expert와 S/V가 실제로 더한 교정</h3>
<p>각 결과를 <strong>같은 시점에 같은 버전을 사용한 내부 Global</strong>과 비교한다.
단일 TabPFN은 최종 10만 행을 context로 쓰지만, 제안 모델은 같은 학습 풀을 Global/expert/route 약 50/25/25%로 나눈다.
내부 Global은 약 5만 행이므로 단일 TabPFN과 동일한 기준선이 아니다.</p>
<div id="policy-summary"></div>
<p class="note">호출은 S가 선택한 expert를 실제 추론한 행, 수락은 V가 채택한 행이다. 수락해도 Global과 같은 라벨일 수 있다.
교정·훼손은 최종 예측의 정오답 변화이며, Macro-F1은 클래스마다 같은 가중치를 주므로 순교정 건수와 비례하지 않는다.</p>
<div class="grid"><article class="card"><h3>CIC2018 균등 · Infiltration 교정</h3><p>1,000행/초에서 호출 <strong>873건</strong>, 교정 <strong>450건</strong>, 훼손 <strong>58건</strong>이다.
교정은 모두 Infiltration, 훼손은 Benign이다. Infiltration F1은 <strong>32.10 → 37.72%</strong>, 전체 Macro-F1은 <strong>+0.80%p</strong>다.
Web attack은 내부 Global 대비 바뀌지 않았다.</p></article>
<article class="card"><h3>ToN 균등 · 교정은 XSS에 집중</h3><p>같은 유입률에서 교정 <strong>6,898건</strong> 중 <strong>6,834건</strong>이 XSS다.
XSS F1은 <strong>89.52 → 90.55%</strong>지만 DoS는 <strong>93.93 → 92.83%</strong>로 내려가, 전체 증가는 <strong>+0.02%p</strong>에 그친다.
Scanning·MitM·Ransomware의 F1 개선으로 이어지지는 않았다.</p></article></div>
<p><strong>자연비율은 두 데이터 모두 호출 0건</strong>이다. 균등에서는 CIC2018의 최종 Bot 도입 버전과 ToN의 Ransomware·Backdoor 도입 버전만 활성화됐다.
ToN은 그 버전이 서비스할 때 전 행을 호출하지만, 전체 재생의 호출률은 <strong>21.21%</strong>다.
MitM 도입 후 새 버전은 Global만 사용한다. 다만 새 버전 준비 전에는 이전 활성 버전을 사용하므로 MitM 평가 구간 초반에도 호출이 남을 수 있다.</p>
<details><summary>선택 조건의 내부 Global 대비 클래스별 교정·훼손과 P/R/F1</summary><div id="paired-classes"></div></details>
<details><summary>학습 버전별 활성화 선택과 평가 구간별 실제 호출</summary>
<p class="note">위 표는 검증 자료에서 고른 학습 버전별 정책, 아래 표는 갱신 지연을 포함해 실제 서비스한 평가 구간별 집계다.</p>
<div id="selections"></div><div id="policy-stages"></div></details>
<div class="callout orange"><strong>설계 재검토의 근거:</strong> 일부 교정은 확인됐지만, 현재 bank·S/V를 구성하는 방식이 모든 조건에서 단일 TabPFN보다 높은 전체 F1로 이어지지는 않았다.
동일 버전의 내부 Global 비교는 expert/S/V의 채택 효과를 보여준다. Global 단독 배치라면 더 빨리 갱신될 수 있으므로, 이 표는 Global 단독 시스템의 운영비용까지 맞춘 ablation은 아니다.</div>

<h3>3.3 갱신비용과 추론비용</h3>
<p>아래 표는 선택한 데이터·배분·평가 정책의 비용이다. 갱신 합계는 <strong>초기 구축을 제외한</strong> CIC2018 후속 4회, ToN 후속 5회다.
추론 합계는 선택 정책의 모든 재생 배치 서비스시간 합이며 배치 수집·큐 대기를 제외한다.</p><div id="costs"></div>
<p class="note">제안 모델의 갱신에는 Global·residual bank·expert·S/V 학습 및 검증 기반 정책 선택을 포함한다.
LoCalPFN은 미세조정·검증·검색 인덱스 구축을 포함한다. 모든 모델의 checkpoint 저장·복구와 결과 파일 기록은 실험 overhead로 제외한다.
XGB·TabPFN의 기존 실측 비용을 재사용하며 0으로 처리하지 않는다. RTX 4090에서 GPU 작업을 하나씩 실행했다.</p>
<p>제안 모델의 후속 갱신 합계는 <strong>114~140초</strong>로 단일 TabPFN <strong>41~50초</strong>, XGB <strong>4~12초</strong>보다 크다.
호출 0건인 조건에서도 bank와 S/V를 준비하는 비용은 이미 발생한다.
내부 Global의 context가 작아 일부 조건의 서비스 추론은 단일 TabPFN보다 빠르지만, 이것만으로 전체 구조의 비용 우위를 주장할 수는 없다.</p>
<details><summary>선택 조건의 단계별 갱신비용과 실제 정보 사용량</summary>
<p class="note">XGB·TabPFN은 기존 3회 측정 중 첫 값을 시계에 사용했다. 새 제안 모델·BoostPFN·LoCalPFN 비용은 각 단계 한 번의 실측이다.
DistPFN의 기반 fit은 TabPFN 실측을 공유한다. 서로 다른 backbone·context 길이를 사용하므로 동일 backbone의 구조만 분리한 비교가 아니다.</p><div id="fit-costs"></div></details>

<h3>3.4 갱신 지연에 따른 성능과 추론 대기열을 구분한다</h3>
<div id="delay-effects"></div><div id="latencies"></div>
<div class="callout"><strong>LoCalPFN 결과를 읽는 방법:</strong> 자연비율의 지연 없는 F1은 CIC2018 <strong>77.26%</strong>, ToN <strong>77.31%</strong>다.
1,000행/초에서는 각각 <strong>62.27%</strong>, <strong>48.05%</strong>로 낮아진다.
갱신이 오래 걸려 구버전으로 분류한 행이 많기 때문이다. 별도로, 약 3.1~3.2초/1,024행인 서비스가 유입 속도를 따라가지 못해 추론 대기열도 쌓인다.</div>
<p class="note">모델 버전은 배치 수집 완료 시점에 고정한다. 추론 큐에서 기다리는 동안 새 버전으로 바꾸지 않는다.
모든 행을 끝까지 처리하므로 <strong>응답이 늦었다는 이유로 오답 처리하지 않는다.</strong>
LoCalPFN의 갱신 지연에 따른 F1 감소와 추론 지연은 서로 다른 지표다.
100행/초의 비교적 큰 배치 수집 대기와, 1,000행/초 LoCalPFN의 지속적인 큐 누적도 구분한다.</p>
<details><summary>선택 조건의 탐지 누락·공격 유형 오분류·정상 오탐</summary><div id="error-metrics"></div></details>

<h3>3.5 구조 개선으로 이어갈 판단</h3>
<p><strong>첫째, Global에 남기는 정보와 expert를 만드는 정보를 함께 설계한다.</strong>
동일 10만 행을 나누는 과정에서 Global context가 줄어드는 영향을 분리하고, expert가 그 손실 이상으로 교정하는지 확인한다.
현재 비교만으로 Global 감소가 점수 차이의 원인이라고 단정하지 않는다.</p>
<p><strong>둘째, expert의 특화와 유사 클래스의 오탐을 같은 목표로 평가한다.</strong>
CIC2018의 Infiltration 교정은 유지할 후보지만 ToN의 XSS 교정만으로 tail 탐지 목표가 달성되지는 않는다.
DoS의 FP 증가, Scanning·MitM·Ransomware의 정밀도·재현율을 함께 개선하도록 context와 채택 기준을 검토한다.</p>
<p><strong>셋째, bank 구축이 추가하는 갱신 시간을 모델의 가치에 포함한다.</strong>
호출률을 억지로 높이기보다, 현재 준비 비용을 감수할 만큼 희소 공격 교정이 반복되는 구조인지 확인한다.
이 결과는 단일 seed의 첫 시나리오 비교이며, 구조 개편 후 동일 정보 예산과 시계에서 다시 평가할 기준점이다.</p>
</section>'''


MODELS = '''<section id="models"><h2>6. 여섯 비교 방법의 실제 적용</h2>
<div class="table-wrap"><table><thead><tr><th>모델</th><th>단계별 반영 방식</th><th>별도 validation 사용</th></tr></thead><tbody>
<tr><td>XGBoost</td><td>누적 전체 CUDA hist 재학습 · 300 trees · depth 8</td><td>고정 설정 · 미사용</td></tr>
<tr><td>TabPFN-v3</td><td>누적 전체 context · ensemble 4 · frozen backbone · CPU KV cache</td><td>고정 설정 · 미사용</td></tr>
<tr><td>제안 모델</td><td>Global/expert/route 약 50/25/25% · K=4 · anchor+residual block · S1V0를 단계마다 갱신</td><td>temperature와 S/V 임계값 선택</td></tr>
<tr><td>BoostPFN <a href="#ref-boost">[5]</a></td><td>v1 기반 · 50 learners · learner당 500 context · 결합 가중치 재구성</td><td>고정 설정 · 미사용</td></tr>
<tr><td>LoCalPFN <a href="#ref-local">[6]</a></td><td>v1 기반 · 검색 context 1,000행 · 미세조정·모델 선택·검색 준비</td><td>checkpoint 선택</td></tr>
<tr><td>DistPFN <a href="#ref-dist">[7]</a></td><td>기반 TabPFN-v3 갱신 + 현재 수집 완료 배치의 posterior 평균으로 보정</td><td>미사용 · test 정답 미사용</td></tr>
</tbody></table></div><p class="note">세 SOTA는 원 방법을 단계마다 반복 적용했다. 원 논문이 신규 클래스 증분 갱신 알고리즘을 제안했다는 뜻은 아니다.
DistPFN에는 미래 test 배치의 분포를 제공하지 않는다. BoostPFN·LoCalPFN의 v1과 다른 방법의 v3는 backbone이 다르다.</p>
<details><summary>현재 expert 구성과 선택 정책</summary><p>공통 anchor는 Global pool에서 클래스당 최대 2,000개를 선택한다.
특화 block은 expert pool의 Global 확률·embedding PCA·정답 residual로 만든 네 군집이다. 이번 pool에서는 기존 block 상한보다 작아 군집 전체가 들어간다.
정답 residual은 학습 시 bank 구성에만 사용하고 test 라우팅에 주지 않는다.</p>
<p>S1은 route의 클래스 균형 교정/훼손 목표를, V0는 NLL 이득의 0.25 분위수를 회귀한다.
검증에서 Macro-F1 개선, 정상 오탐 증가 한도 0.05%p, 보호 클래스·지정 tail 제약을 함께 확인한다.
정책이 Global만 사용하도록 선택되면 test expert 추론을 생략한다.</p></details></section>'''


def build(run):
    data = load_verified(run)
    page = HTML
    page = re.sub(r'<title>.*?</title>', '<title>순차 공격 도입: 제안 모델·SOTA 성능과 비용</title>', page)
    header = '''<header><div class="kicker">10/14 시나리오 · 2026-10-15 결과 갱신 · 연구 버전 v7</div>
<h1>새 공격 반영의 성능과 비용<br>제안 모델과 SOTA 여섯 방법 비교</h1>
<p class="lead">같은 공격 사례를 순서대로 공개했을 때, 희소 공격을 얼마나 잘 구분하고 그 능력을 언제부터 사용할 수 있는지 평가한다.</p>
<div><span class="pill">6개 방법 · 24개 설정 완료</span><span class="pill">CIC2018 · ToN</span><span class="pill">최종 train 10만 행 + 별도 validation</span><span class="pill">Seed 43 · 제안 모델 K=4</span></div>
<div class="callout"><strong>현재 판단:</strong> 1,000행/초에서는 CIC2018은 단일 TabPFN, ToN은 XGB의 전체 F1이 가장 높다.
제안 모델은 균등 조건에서 일부 교정을 더했지만 단일 TabPFN을 넘지는 못했다. 희소 공격의 교정과 bank 갱신비용을 함께 개선할 구조를 검토한다.</div>
<nav aria-label="문서 목차"><a href="#question">1. 평가 목적</a><a href="#scenario">2. 데이터와 도입 순서</a><a href="#results">3. 성능·S/V·비용 결과</a><a href="#clock">4. 시뮬레이션 시계</a><a href="#evaluation">5. 측정 정의</a><a href="#models">6. 비교 모델</a><a href="#references">7. 근거 논문</a></nav></header>'''
    page = re.sub(r'<header>.*?</header>', lambda _: header, page, flags=re.S)
    page = page.replace('<section id="question"><h2>1. 갱신 후 성능과 실제 반영 시점을 구분한다</h2>',
        '<section id="question"><h2>1. 갱신 후 성능과 실제 반영 시점을 구분한다</h2>'
        '<p>10/13은 갱신 완료 후 성능·비용을 본 사전 실험, 10/14의 XGB·TabPFN 비교는 순차 재생 프로토콜의 사전 검증이다. '
        '<strong>이번에는 같은 프로토콜에 현 제안 모델과 SOTA 3종을 추가했다.</strong> 최종 방법론과 기여를 확정하기 전의 첫 시나리오 비교다.</p>')
    page = page.replace('두 모델은 같은 학습 사례와 같은 test 행을 받는다.', '여섯 방법은 같은 학습 사례와 같은 test 행에 접근한다.')
    page = page.replace('신규 공격과 추가 정상 사례를 두 모델에 동시에 제공한다.', '신규 공격과 추가 정상 사례를 여섯 방법에 동시에 제공한다.')
    page = page.replace('XGB는 누적 재학습, TabPFN은 누적 context와 캐시를 준비한다.', 'XGB는 누적 재학습, PFN 방법은 각 방식의 context·모델·정책을 준비한다.')
    page = page.replace('이번 XGB·TabPFN은 고정 설정으로 실행하며 추가 검증 라벨을 소비하지 않는다.',
        '별도 정제 validation도 해당 클래스의 도입 시점부터 제공한다. XGB·단일 TabPFN·BoostPFN·DistPFN은 고정 설정으로 미사용하고, 제안 모델과 LoCalPFN이 사용한다.')
    validation = '''<details><summary>별도 validation과 제안 모델 내부 분할</summary><p>10만 train 행을 줄이지 않고 기존 정제 validation에서 현재까지 등장한 클래스의 사례를 별도로 제공한다.
최종 실제 validation은 CIC2018 자연 8,211행·균등 8,047행, ToN 자연 8,668행·균등 8,605행이다.
현재 train과 feature가 겹치는 validation 행은 제외한다. Test 정답은 정책 선택에 사용하지 않는다.</p>
<p>제안 모델은 train을 Global/expert/route 약 50/25/25%로 나누고, 동일 feature 그룹의 역할은 한 곳에 고정한다.
별도 validation을 tune/cal 약 40/60%로 나눠 temperature와 S/V 임계값을 고른다. 이번에는 시간순 cal 확인용 재분할 없이 cal 전체로 선택한다.
과거 EXP62/63의 역할별 대규모 pool과 동일 조건이라고 해석하지 않는다.</p></details>'''
    page = page.replace('<section id="clock">', validation + RESULTS + '\n<section id="clock">', 1)
    page = page.replace('<h2>3. 갱신이', '<h2>4. 갱신이').replace('<h2>4. 분류 능력', '<h2>5. 분류 능력').replace('<h2>6. 근거 논문', '<h2>7. 근거 논문')
    page = re.sub(r'<section id="models">.*?</section>', lambda _: MODELS, page, flags=re.S)
    page = page.replace('모델 생성·학습/캐시 준비 및 train-only 첫 1,024행 응답을 단계당 3회 측정한다.',
        'EXP72의 XGB·TabPFN은 모델 생성·학습/캐시 준비 및 train-only 첫 1,024행 응답을 단계당 3회 측정했다. 새 제안 모델·BoostPFN·LoCalPFN은 단계별 1회 실측이며, DistPFN은 기반 TabPFN fit을 공유한다.')
    page = re.sub(r'<div class="callout"><strong>이 페이지에는 EXP72.*?</div>',
        '<p class="note">검증된 여섯 방법의 결과는 3절에 수록했다. 갱신 완료 후 성능·비용 사전 실험은 10/13 탭에 보존한다.</p>', page, flags=re.S)
    footer = f'''<footer>10/15 사용자 요청으로 결과 반영 · 승인된 연구 버전 v7 유지 · Seed 43
<details><summary>실행 출처와 검증</summary><p>추가 비교: <code>{data['run']}/results.json</code><br>
기준선: <code>{data['baseline']}/results.json</code></p><p>추가 48/48 정책, 기존 24/24 정책 검증 통과.
원본 train ID·순서, 각 test 행 1회 사용, 준비 시점에 맞는 모델 버전, 저장 예측의 혼동행렬·F1 및 S/V 교정·훼손 일치를 확인했다.</p>
<p>프로토콜: <code>docs/research/20261014/exp73_protocol.md</code><br>해석·출처: <code>docs/research/20261015/exp73_results.md</code><br>
보고서 생성: <code>scripts/v7/exp73/build_results_report.py</code></p></details></footer>'''
    page = re.sub(r'<footer>.*?</footer>', lambda _: footer, page, flags=re.S)
    css = (Path(__file__).with_name('report_ui.css')).read_text()
    js = (Path(__file__).with_name('report_ui.js')).read_text()
    packed = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    page = page.replace('</head>', '<style>' + css + '</style></head>')
    page = page.replace('</body>', f'<script id="exp73-data" type="application/json">{packed}</script><script>{js}</script></body>')
    title = '순차 공격 도입 · 제안 모델과 SOTA 결과'
    card = '<div class="status-cell"><p class="label">10/14 시나리오 · 10/15 여섯 방법 완료</p><p class="value">CIC2018은 단일 TabPFN, ToN은 XGB F1 우세. 제안 모델의 일부 교정과 추가 갱신비용을 함께 확인했다. <button class="jump" data-jump="'+REPORT_ID+'">성능·S/V·비용·클래스별 결과</button></p></div>'
    flow = '<div class="flow-step"><time>10-14<br>↳ 10-15</time><div class="flow-rail-dotcol"><div class="d"></div><div class="l"></div></div><div class="body"><button class="jump" data-jump="'+REPORT_ID+'"><h3>'+title+'</h3></button><p>제안 모델·SOTA까지 24개 설정 완료. 지연 없는 분류 능력, 갱신 중 구버전 사용, 실제 S/V 교정과 비용을 구분한다.</p></div></div>'
    publish(page, title=title, verdict='6개 방법 완료 · 성능 / S/V / 비용',
            title_en='Sequential replay · Proposed model and SOTA results', card=card, flow=flow)
    index = ROOT / 'lablog/html_report/post_tabpfn.html'
    text = index.read_text().replace('2026-08-18 &ndash; 2026-10-14', '2026-08-18 &ndash; 2026-10-15')
    index.write_text(text)
    dest = ROOT / 'docs/research/20261015'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'exp73_report_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print('Read-only result rendering complete; no experiments launched.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    build(parser.parse_args().run.resolve())
