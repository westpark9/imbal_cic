#!/usr/bin/env python3
"""Publish verified EXP72 results after an explicit user request; no model calls."""
import argparse
from html import escape
import hashlib
import json
from pathlib import Path
import re

from build_protocol_report import HTML, REPORT_ID, ROOT, publish

DATASETS = {'cic2018': 'CIC2018', 'toniot': 'ToN'}
CASES = {'natural': '자연비율', 'uniform': '균등'}
METHODS = {'xgb': 'XGB', 'tabpfn': 'TabPFN'}
POLICIES = {'ideal': '갱신 지연 없음', 'rate_100': '100행/초', 'rate_1000': '1,000행/초'}
CLASSES = {'benign': 'Benign', 'brute_force': 'Brute force', 'web_attacks': 'Web attack',
           'infiltration': 'Infiltration', 'bot': 'Bot', 'dos': 'DoS', 'ddos': 'DDoS',
           'xss': 'XSS', 'password': 'Password', 'injection': 'Injection', 'backdoor': 'Backdoor',
           'scanning': 'Scanning', 'mitm': 'MitM', 'ransomware': 'Ransomware'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, digits=2):
    return '—' if value is None else f'{value:,.{digits}f}'


def pct(value, digits=2):
    return '—' if value is None else number(value * 100, digits)


def table(headers, rows, numeric_from=0):
    def cells(values, tag):
        return ''.join(f'<{tag}' + (' class="numcell"' if i >= numeric_from else '') +
                       f'>{escape(str(value))}</{tag}>' for i, value in enumerate(values))
    return '<div class="table-wrap"><table><thead><tr>' + cells(headers, 'th') + \
           '</tr></thead><tbody>' + ''.join('<tr>' + cells(row, 'td') + '</tr>' for row in rows) + \
           '</tbody></table></div>'


def class_table(xgb, pfn):
    by_name = {c['name']: c for c in pfn['classes']}
    rows = []
    for x in sorted(xgb['classes'], key=lambda c: -c['support']):
        t = by_name[x['name']]
        if not x['support']:
            continue
        rows.append([CLASSES[x['name']], f"{x['support']:,}"] +
                    [pct(c[k]) for c in (x, t) for k in ('precision', 'recall', 'f1')])
    return table(['클래스', 'Test 표본 수', 'XGB P (%)', 'XGB R (%)', 'XGB F1 (%)',
                  'TabPFN P (%)', 'TabPFN R (%)', 'TabPFN F1 (%)'], rows, 1)


def get_run(run):
    status, verification = read(run/'status.json'), read(run/'verification.json')
    if status['state'] != 'complete' or verification['state'] != 'passed' or verification['verified_policies'] != 24:
        raise ValueError('Publication requires completed and independently verified 24-policy results')
    jobs, fits = {}, {}
    for ds in DATASETS:
        for case in CASES:
            for method in METHODS:
                path = run/'jobs'/f'{ds}_{case}_{method}'
                if read(path/'COMPLETE.json')['results_sha256'] != sha(path/'results.json'):
                    raise ValueError(f'Result hash mismatch: {path}')
                key = ds, case, method
                jobs[key] = read(path/'results.json')['results']
                fits[key] = [read(p) for p in sorted(path.glob('stage_*/fit_metrics.json'))]
                for f in fits[key]:
                    p = path/f"stage_{f['stage']}"
                    if read(p/'FIT_COMPLETE.json')['metrics_sha256'] != sha(p/'fit_metrics.json'):
                        raise ValueError(f'Fit hash mismatch: {p}')
    return jobs, fits


def result_sections(jobs, fits, run):
    overview, costs, effects, latencies, repeats, metrics = [], [], [], [], [], []
    detail = []
    for ds, ds_name in DATASETS.items():
        for case, case_name in CASES.items():
            x, t = [jobs[ds, case, m]['rate_1000'] for m in METHODS]
            overview.append([ds_name, case_name, pct(x['macro_f1']), pct(t['macro_f1']),
                             f"{100*(t['macro_f1']-x['macro_f1']):+.2f}",
                             pct(x['benign_false_positive_rate'], 3), pct(t['benign_false_positive_rate'], 3)])
            for method, method_name in METHODS.items():
                z = jobs[ds, case, method]
                f = fits[ds, case, method]
                current, ideal = z['rate_1000'], z['ideal']
                costs.append([ds_name, case_name, method_name, number(f[0]['update_seconds']),
                              number(sum(v['update_seconds'] for v in f[1:])),
                              number(f[-1]['update_seconds']), number(current['predict_service_seconds'])])
                effects.append([ds_name, case_name, method_name, pct(ideal['macro_f1']),
                                pct(z['rate_100']['macro_f1']), pct(current['macro_f1']),
                                number(100*(current['macro_f1']-ideal['macro_f1']), 3),
                                f"{current['latency']['rows_using_outdated_version']:,}",
                                f"{current['attack_to_benign']-ideal['attack_to_benign']:+,}"])
                for policy in ('rate_100', 'rate_1000'):
                    a, l = z[policy], z[policy]['latency']
                    latencies.append([ds_name, case_name, method_name, POLICIES[policy],
                                      number(l['mean_seconds'], 3), number(l['p95_seconds'], 3),
                                      number(l['queue_wait_max_seconds'], 3),
                                      number(l['drain_after_last_arrival_seconds'], 3)])
                for policy in POLICIES:
                    a = z[policy]
                    metrics.append([ds_name, case_name, method_name, POLICIES[policy],
                                    pct(a['macro_f1']), pct(a['attack_detection_recall']),
                                    pct(a['benign_false_positive_rate'], 3),
                                    f"{a['attack_to_benign']:,}", f"{a['wrong_attack_family']:,}"])
                for a in f:
                    samples = [s['update_seconds'] for s in a['cost_repetitions']]
                    repeats.append([ds_name, case_name, method_name, a['stage'], f"{a['train_rows']:,}",
                                    number(a['update_seconds'], 3), number(a['update_median_seconds'], 3),
                                    number(min(samples), 3), number(max(samples), 3),
                                    number(a['cost_repetitions'][0]['first_batch_seconds'], 3)])
            detail.append(f'<details><summary>{ds_name} · {case_name} — 전체 클래스 및 단계별 P/R/F1</summary>'
                          '<p class="note">각 표의 Test 표본 수 내림차순. P/R/F1은 %, FP는 Benign과 다른 공격을 포함한 '
                          '해당 평가 구간 전체에서 계산한다. 단계별로 아직 등장하지 않은 클래스는 생략한다.</p>')
            for policy, policy_name in POLICIES.items():
                x, t = [jobs[ds, case, m][policy] for m in METHODS]
                detail.append(f'<details><summary>{policy_name}</summary><h4>전체 재생 행의 클래스별 지표</h4>' + class_table(x, t))
                stage_rows = [[a['stage'], ' · '.join(CLASSES[c] for c in a['new_classes']) or '초기 클래스',
                               f"{a['rows']:,}", pct(a['macro_f1']), pct(b['macro_f1'])]
                              for a, b in zip(x['stages'], t['stages'])]
                detail.append('<details><summary>단계별 성능 펼치기</summary>' +
                              '<p class="note">단계 F1은 그때까지 등장한 클래스의 평균이다. 클래스 구성이 달라지므로 '
                              '단계 간 F1 차이만으로 망각을 판단하지 않는다.</p>' +
                              table(['단계', '신규 클래스', 'Test 표본 수', 'XGB F1 (%)', 'TabPFN F1 (%)'], stage_rows, 2))
                for a, b in zip(x['stages'], t['stages']):
                    detail.append(f'<details><summary>단계 {a["stage"]} 클래스별 지표</summary>' + class_table(a, b) + '</details>')
                detail.append('</details></details>')
            detail.append('</details>')
    html = '''<!-- EXP72_RESULTS_START --><section id="results"><h2>3. 사전 검증 결과</h2>
<p>2026-10-14 15:02 완료. XGB·TabPFN × 두 데이터 × 자연/균등의 <strong>8개 설정</strong>과 설정별 세 재생 정책을 검증했다.
아래 기본 표는 <strong>1,000행/초, 갱신 지연 포함, 전체 재생 구간</strong>의 결과다.</p>
<h3>3.1 분류 성능과 정상 오탐</h3>
<p class="note">F1은 모든 재생 행의 혼동행렬에서 계산한 Macro-F1이다. 최종 모델 하나로 전체 test를 분류한 10/13 결과와 집계 방식이 다르다.
오탐률은 Benign 중 공격으로 예측한 비율이다. 모든 행을 끝까지 처리하며, 지연된 응답을 오답이나 드롭으로 세지 않는다.</p>'''
    html += table(['데이터', '공격 배분', 'XGB F1 (%)', 'TabPFN F1 (%)', '차이 (%p)',
                   'XGB 정상 오탐 (%)', 'TabPFN 정상 오탐 (%)'], overview, 2)
    html += '''<div class="grid"><article class="card"><h3>CIC2018</h3>
<p>TabPFN은 자연비율 <strong>+0.69%p</strong>, 균등 <strong>+4.15%p</strong> 높은 Macro-F1과 낮은 정상 오탐률을 보인다.
균등 조건에서는 Infiltration·Web attack도 XGB보다 높다. 다만 자연비율의 Infiltration은 XGB가 높아 모든 희소 공격의 개선으로 일반화할 수 없다.</p></article>
<article class="card"><h3>ToN</h3><p>TabPFN은 자연비율 <strong>−5.89%p</strong>, 균등 <strong>−1.74%p</strong> 낮다.
균등에서 Scanning은 개선되지만 MitM의 차이가 크다. 배분 변경의 효과가 클래스별로 달라, 데이터 확대만으로 해결됐다고 보기 어렵다.</p></article></div>
<h3>3.2 갱신비용과 추론비용은 XGB가 낮다</h3>
<p>초기 구축과 이후 갱신을 분리한다. 갱신 합계는 CIC2018의 후속 4회, ToN의 후속 5회이며, 각 단계의 첫 실측값을 사용한다.
TabPFN은 가중치를 학습하지 않지만 매 단계 <strong>누적 context 전체의 전처리·캐시를 재구축</strong>한다.</p>'''
    html += table(['데이터', '공격 배분', '모델', '초기 구축 (초)', '후속 갱신 합계 (초)',
                   '최종 1회 갱신 (초)', '재생 추론 합계 (초)'], costs, 3)
    html += '''<p class="note">갱신 = 입력 구성 + 모델 생성 + fit/cache 준비. Train-only 첫 1,024행 probe는 별도 측정하며 갱신 합계에 넣지 않는다.
추론 합계는 1,000행/초 정책에서 사용한 각 배치의 실측 서비스시간 합으로, 트래픽 유입을 기다리는 시간과 배치 수집 시간은 포함하지 않는다.
Checkpoint 저장·복구 및 파일 기록은 실험 overhead로 제외한다. RTX 4090, XGB CUDA hist, TabPFN-v3 ensemble 4·CPU KV cache 조건이다.</p>
<div class="callout orange"><strong>채택 근거의 범위:</strong> CIC2018의 분류 성능과 정상 오탐 감소는 활용 가능성을 보여준다.
그러나 이번 <strong>누적 10만 행 전체 재구축</strong>에서는 TabPFN의 갱신비용 우위가 확인되지 않았다.
‘가중치 재학습 없음’과 ‘갱신이 저렴함’을 같은 의미로 쓰면 안 된다.</div>
<details><summary>단계별 갱신비용과 3회 반복 범위</summary><p class="note">시계 사용값은 첫 반복 + 입력 구성 시간이다.
중앙값·최소·최대는 3회 모델 생성+fit/cache 시간이며 입력 구성은 제외한다. 첫 응답 probe는 첫 반복의 train-only 1,024행 추론이다.</p>'''
    html += table(['데이터', '공격 배분', '모델', '단계', '누적 학습 행', '시계 사용 (초)',
                   '중앙값 (초)', '최소 (초)', '최대 (초)', '첫 응답 probe (초)'], repeats, 3) + '</details>'
    html += '''<h3>3.3 갱신 지연은 추가 오류를 만들지만 현재 격차의 주된 설명은 아니다</h3>
<p>1,000행/초에서 TabPFN의 구버전 처리량은 <strong>38,912–48,128행</strong>이다.
갱신 지연이 없는 동일 재생 대비 F1 감소는 <strong>0.182–0.282%p</strong>이며, XGB는 0.015–0.090%p다.
ToN의 격차는 지연을 제거해도 남으므로, 갱신속도 외에 context와 분류 성능을 개선해야 한다.</p>'''
    html += table(['데이터', '공격 배분', '모델', '지연 없음 F1 (%)', '100행/초 F1 (%)',
                   '1,000행/초 F1 (%)', '지연 영향 (%p)', '구버전 처리 행', '공격→Benign 순증가'], effects, 3)
    html += '''<p class="note">뒤의 세 열은 1,000행/초 기준이다. 구버전 처리 행은 정오답과 무관한 입력 수다.
‘공격→Benign 순증가’는 지연 없는 비교군과의 건수 차이이며, 다른 공격 유형으로 분류한 오류는 별도다.</p>
<p>응답 p95는 1,000행/초에서 XGB <strong>0.975–0.978초</strong>, TabPFN <strong>1.256–1.277초</strong>다.
측정한 두 유입률에서는 지속적인 추론 큐 누적은 나타나지 않았다. 100행/초에서 p95가 약 10초로 더 큰 것은
1,024행 배치를 채우는 대기 때문이며, 모델 자체의 추론이 더 느려진 결과가 아니다.</p>
<details><summary>유입률별 응답 지연과 입력 종료 후 잔여 처리시간</summary>
<p class="note">행별 응답시간 = 배치 수집 + 추론 큐 대기 + 서비스시간. 갱신과 서비스는 독립 자원 가정이다.</p>'''
    html += table(['데이터', '공격 배분', '모델', '유입률', '평균 (초)', 'p95 (초)',
                   '최대 큐 대기 (초)', '입력 후 잔여 (초)'], latencies, 4) + '</details>'
    html += '<details><summary>전체 정책의 탐지 누락·공격 유형 오분류·정상 오탐</summary>'
    html += table(['데이터', '공격 배분', '모델', '정책', 'F1 (%)', '공격 탐지 R (%)',
                   '정상 오탐 (%)', '공격→Benign', '다른 공격 유형'], metrics, 4) + '</details>'
    html += '''<h3>3.4 희소 클래스는 개별적으로 확인한다</h3>
<p>1,000행/초 기준, CIC2018 균등의 Infiltration F1은 XGB <strong>20.32%</strong> / TabPFN <strong>31.50%</strong>,
Web attack은 <strong>20.18% / 39.64%</strong>다. ToN 균등의 Scanning은 <strong>14.83% / 25.75%</strong>이지만
MitM은 <strong>46.75% / 22.07%</strong>다. ToN 자연비율의 Scanning은 TabPFN <strong>0.00%</strong>다.
일부 희소 클래스의 개선과 남아 있는 병목을 분리해 본 실험의 목표를 정한다.</p>'''
    html += ''.join(detail)
    html += '''<div class="callout"><strong>다음 개선의 초점:</strong> 전체 누적 사례는 보존하되,
TabPFN에 넣는 <strong>context의 크기와 선택</strong>을 제어해 캐시 재구축·추론 비용을 줄이면서 희소 공격의 P/R/F1을 유지·개선할 수 있는지 확인한다.
희소 공격과 다양한 Benign을 보존하는 구성은 검증할 후보이며, 효과가 확인된 해법은 아니다.
같은 사례 접근권한·갱신 시점·자원 조건의 XGB와 비교한다. 이번에는 이 후속 실험을 추가 실행하지 않았다.</div>
<p class="note">이는 본 실험 평가 프로토콜을 XGB·TabPFN으로 사전 검증한 결과다.
제안 모델의 기여는 이 프로토콜에서 희소 공격 개선·정상 오탐·비용을 함께 검증하고, SOTA 및 구성요소 비교로 확인할 계획이다.</p>
</section><!-- EXP72_RESULTS_END -->'''
    return html


def build(run):
    if run.name != '20261014_134735_replay_100k_xgb_tabpfn_s43_de8b4a78':
        raise ValueError('The interpretation in this report is specific to the approved EXP72 pilot run.')
    jobs, fits = get_run(run)
    title = '순차 공격 도입: XGB·TabPFN 사전 검증 결과'
    html = HTML.replace('<title>새 공격 반영: 갱신 지연을 포함한 순차 재생 평가</title>', f'<title>{title}</title>')
    html = html.replace('연구 버전 v7 유지 · 평가 설계', '연구 버전 v7 유지 · 사전 검증 완료')
    html = html.replace('<h1>새 공격의 사례를 반영하는 동안<br>탐지기는 무엇을 놓치는가?</h1>',
                        '<h1>새 공격 반영의 성능과 비용<br>XGB·TabPFN 순차 재생 사전 검증</h1>')
    html = html.replace('설명만 수록 · 이번 실험 결과 제외', '8개 설정 완료 · 24개 정책 검증 통과')
    html = html.replace('<nav aria-label="문서 목차">',
        '<div class="callout"><strong>핵심 결과:</strong> CIC2018에서는 TabPFN의 F1과 정상 오탐률이 유리하고, '
        'ToN에서는 XGB의 F1이 높다. 누적 10만 행 조건의 갱신·추론 비용은 모두 XGB가 낮다. '
        '이 결과를 바탕으로 context 구성과 비용을 함께 개선할 조건을 좁힌다.</div><nav aria-label="문서 목차">')
    html = html.replace('<a href="#clock">3. 시뮬레이션 시계</a>',
                        '<a href="#results">3. 결과와 다음 판단</a><a href="#clock">4. 시뮬레이션 시계</a>')
    html = html.replace('>4. 평가·비용</a>', '>5. 측정 정의</a>').replace('>5. 비교 모델</a>', '>6. 비교 모델</a>').replace('>6. 근거 논문</a>', '>7. 근거 논문</a>')
    html = html.replace('<h2>3. 갱신이', '<h2>4. 갱신이').replace('<h2>4. 분류 능력', '<h2>5. 분류 능력')
    html = html.replace('<h2>5. 이번 비교', '<h2>6. 이번 비교').replace('<h2>6. 근거 논문', '<h2>7. 근거 논문')
    html = html.replace('<h3>이번 순차 재생 평가</h3>', '<h3>이번 순차 재생 사전 검증</h3>')
    html = html.replace('<section id="question"><h2>1. 갱신 후 성능과 실제 반영 시점을 구분한다</h2>',
        '<section id="question"><h2>1. 갱신 후 성능과 실제 반영 시점을 구분한다</h2>'
        '<p><strong>10/13은 갱신 완료 후 성능·비용을 본 사전 실험, 10/14는 본 실험에 사용할 평가 프로토콜의 사전 검증</strong>이다. '
        '이번 결과만으로 제안 모델의 기여를 평가한 것은 아니다.</p>')
    html = html.replace('<section id="clock">', result_sections(jobs, fits, run) + '\n<section id="clock">')
    html = re.sub(r'<div class="callout"><strong>이 페이지에는 EXP72.*?</div>',
                  '<p class="note">검증된 결과는 3절에 수록했다. 기존 갱신 완료 후 성능·비용은 10/13 탭에 보존한다.</p>', html, flags=re.S)
    html = html.replace('결과 자동 게시 비활성화', '2026-10-14 사용자 요청으로 검증 결과 게시')
    relative = str(run.relative_to(ROOT))
    html = html.replace('</footer>', '<details><summary>실행 출처와 검증</summary><p>결과: <code>' + escape(relative) +
        '/results.json</code></p><p>검증: 같은 실행의 <code>verification.json</code> — 24/24 통과. '
        '모든 test 행 1회 평가, 원본 train IDs 일치, 버전 준비 시점 준수, 저장 예측과 혼동행렬·F1 재계산 일치를 확인했다.</p>'
        '<p>게시 코드: <code>scripts/v7/exp72/build_results_report.py</code>. '
        '중간 CPU 추론 기록은 audit에 보존하고 제외했으며 XGB 추론은 CUDA로 다시 완료한 결과만 사용했다.</p></details></footer>')
    card = '<div class="status-cell"><p class="label">10/14 · 순차 재생 사전 검증 완료</p><p class="value">'
    card += 'CIC2018은 TabPFN F1 우세, ToN은 XGB 우세. 10만 행 누적 갱신·추론 비용은 XGB가 낮다. '
    card += f'<button class="jump" data-jump="{REPORT_ID}">성능·비용·클래스별 상세 결과</button></p></div>'
    flow = '<div class="flow-step"><time>10-14</time><div class="flow-rail-dotcol"><div class="d"></div><div class="l"></div></div>'
    flow += f'<div class="body"><button class="jump" data-jump="{REPORT_ID}"><h3>{title}</h3></button>'
    flow += '<p>8개 설정·24개 정책 검증 완료. 갱신 지연의 추가 영향과 기본 분류 성능의 차이를 구분하고, context·비용 개선 과제를 정리했다.</p></div></div>'
    publish(html, title=title, verdict='완료 · 사전 검증 · 성능·비용·클래스별 결과',
            title_en='Sequential replay · XGB and TabPFN pilot results', card=card, flow=flow)
    return jobs, fits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    build(args.run.resolve())


if __name__ == '__main__':
    main()
