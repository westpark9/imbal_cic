"""Build the 2026-11-06 Korean report deck on replacing CSV inputs with PCAP encoders.

The story follows the presenter's outline: prior problems on CIC CSV, how the PCAP
features are made with residual flows deleted, then the two problems again.  The
statistics view is left out because the goal is an encoder feature.
"""
from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = next(path for path in Path(__file__).resolve().parents if (path / 'scripts/common/experiment_paths.py').exists())
sys.path.insert(0, str(ROOT / 'scripts/common'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_20261023_references as r
import build_20261104_exp94 as b04

HERE = ROOT / 'docs/slides'
QA = HERE / '_support/qa/20261106_merged'
OUT = HERE / 'kor/20261106_PCAP_통합'
ASSETS = HERE / '_support/assets/20261106_merged'
EV = ROOT / 'docs/lablog/evidence'
AUDIT17 = json.loads((EV / '20261103/original_cic2017_csv_audit.json').read_text(encoding='utf-8'))
AUDIT18 = json.loads((EV / '20261030/original_cic2018_csv_audit.json').read_text(encoding='utf-8'))
PCAP = json.loads((EV / '20261104/exp94_pcap_summary.json').read_text(encoding='utf-8'))
COUNTS = json.loads((EV / '20261106/exp95_residual_option_counts.json').read_text(encoding='utf-8'))
BOUNDARY, WALK = b04.BOUNDARY, b04.WALK
PAIRED = ROOT / 'tabpfn/results/v7/exp95/20261106_042628_drop_paired_418e36f3'
MAIN_ARM = 'netfound-exp94-pca'
ALL_CLASS = ROOT / 'tabpfn/results/v7/exp95/20261106_051907_all_class_0ecc759e'
AC = json.loads((ALL_CLASS / 'comparison.json').read_text(encoding='utf-8'))['table']
SIDE = json.loads((ROOT / 'docs/lablog/evidence/20261106/exp95_drop_fewshot_summary.json').read_text(encoding='utf-8'))['side_column_conflicts']['nine_class_labels']

txt, box, arrow, block = r.txt, r.box, r.arrow, r.block
NAVY, BLUE, INK, GREY, PALE = r.NAVY, r.BLUE, r.INK, r.GREY, r.PALE
RED, TEAL, AMBER, GREEN = b04.RED, b04.TEAL, b04.AMBER, b04.GREEN
card, table, chip = b04.card, b04.table, b04.chip


def fonts():
    from matplotlib.font_manager import FontProperties
    return FontProperties(fname=b04.FONT), FontProperties(fname=b04.BOLD)


def image(name: str, x: float, y: float, w: float, h: float) -> dict:
    from PIL import Image
    path = ASSETS / f'{name}.png'
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    ww, hh = iw * scale, ih * scale
    return dict(kind='image', x=x + (w - ww) / 2, y=y + (h - hh) / 2, w=ww, h=hh, path=str(path))


def pct(v: float, digits: int = 1) -> str:
    return f'{100 * v:.{digits}f}'


# ----------------------------------------------------------------------------- data
def csv_macro(regime: str, method: str) -> dict[int, tuple[float, float]]:
    return {m['budget']: (100 * m['mean'], 100 * m['sample_sd']) for m in PCAP['views']['csv']['macro_f1'] if m['regime'] == regime and m['method'] == method}


def csv_class(regime: str, method: str, budget: int) -> dict[str, float]:
    return {c['name']: 100 * c['mean'] for c in PCAP['views']['csv']['class_f1'] if c['regime'] == regime and c['method'] == method and c['budget'] == budget}


def ac(arm: str, method: str, budget: int) -> dict:
    return next(t for t in AC if t['arm'] == arm and t['method'] == method and t['budget'] == budget)


def ac_curve(arm: str, method: str) -> dict[int, tuple[float, float]]:
    return {n: (100 * ac(arm, method, n)['macro9'], 100 * ac(arm, method, n)['macro9_sd']) for n in (2, 3, 5, 10, 20, 50)}


def paired() -> list[dict]:
    with (PAIRED / 'summary_pairs.csv').open() as f:
        return list(csv.DictReader(f))


def drop_macro(rows: list[dict], regime: str, method: str, arm: str = MAIN_ARM) -> dict[int, tuple[float, float]]:
    import numpy as np
    out = {}
    for n in (2, 3, 5, 10, 20, 50):
        v = [100 * float(x['macro_drop']) for x in rows if x['arm'] == arm and x['regime'] == regime and x['method'] == method and int(x['budget']) == n]
        out[n] = (float(np.mean(v)), float(np.std(v, ddof=1)))
    return out


def drop_class(rows: list[dict], method: str, budget: int, arm: str = MAIN_ARM) -> dict[str, float]:
    import numpy as np
    sel = [x for x in rows if x['arm'] == arm and x['regime'] == 'with_benign' and x['method'] == method and int(x['budget']) == budget]
    return {c: 100 * float(np.mean([float(x[f'f1_{c}_drop']) for x in sel])) for c in ('benign', 'dos', 'portscan', 'ddos', 'bf', 'bot', 'web')}


# ----------------------------------------------------------------------------- charts
def saturation_chart() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    font, bold = fonts()
    classes = ['dos', 'portscan', 'ddos', 'bf', 'bot', 'web']
    budgets = [2, 3, 5, 10, 20, 50]
    fig = plt.figure(figsize=(13.4, 4.2), dpi=190)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=.2)
    ax = fig.add_subplot(grid[0])
    for method, label, color in (('xgb', 'XGB', '#B44948'), ('tabpfn', 'TabPFN', '#007BC6')):
        m = csv_macro('attack_only', method)
        ys = [m[n][0] for n in budgets]
        ax.plot(budgets, ys, marker='o', ms=5.4, lw=2.4, color=color, label=label)
        ax.annotate(f'{ys[-1]:.1f}', (50, ys[-1]), xytext=(6, 3 if method == 'tabpfn' else -12), textcoords='offset points', color=color, fontsize=10, fontproperties=bold)
    ax.set_xscale('log', base=2)
    ax.set_xticks(budgets, [str(n) for n in budgets], fontproperties=font, fontsize=10)
    ax.set_ylim(35, 90); ax.set_ylabel('macro-F1 (%)', fontproperties=font, fontsize=11)
    ax.set_xlabel('클래스당 학습 라벨 수 n', fontproperties=font, fontsize=11)
    ax.set_title('공격 6종 macro-F1', fontproperties=bold, fontsize=14, pad=9)
    ax.grid(axis='y', color='#D9E1E8', lw=.75); ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='lower right', prop=font)
    ax = fig.add_subplot(grid[1])
    M = np.array([[csv_class('attack_only', 'tabpfn', n)[c] for n in budgets] for c in classes])
    ax.imshow(M, aspect='auto', cmap='Blues', vmin=0, vmax=100)
    for yy in range(len(classes)):
        for xx in range(len(budgets)):
            ax.text(xx, yy, f'{M[yy, xx]:.0f}', ha='center', va='center', fontsize=11, color='white' if M[yy, xx] >= 62 else '#172033', fontproperties=bold)
    ax.set_xticks(range(len(budgets)), [f'n={n}' for n in budgets], fontproperties=font, fontsize=10.5)
    ax.set_yticks(range(len(classes)), classes, fontproperties=bold, fontsize=11)
    ax.set_title('클래스별 F1 (%) · TabPFN', fontproperties=bold, fontsize=14, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(ASSETS / 'saturation_cic2017.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fewshot_chart(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    font, bold = fonts()
    budgets = [2, 3, 5, 10, 20, 50]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.45), dpi=200, sharey=True)
    for ax, title, get in ((axes[0], 'CSV 통계 78개', lambda m: ac_curve('csv', m)), (axes[1], 'PCAP + netFound 64', lambda m: ac_curve('netfound', m))):
        for method, label, color in (('xgb', 'XGB', '#B44948'), ('tabpfn', 'TabPFN', '#007BC6')):
            v = get(method)
            ys = [v[n][0] for n in budgets]
            ax.errorbar(budgets, ys, yerr=[v[n][1] for n in budgets], color=color, marker='o', ms=4.6, lw=2.1, capsize=2.4, label=label)
            ax.annotate(f'{ys[-1]:.1f}', (50, ys[-1]), xytext=(5, 4 if method == 'tabpfn' else -12), textcoords='offset points', color=color, fontsize=9.5, fontproperties=bold)
        ax.set_xscale('log', base=2)
        ax.set_xticks(budgets, [str(n) for n in budgets], fontproperties=font, fontsize=9)
        ax.set_ylim(10, 70); ax.set_title(title, fontproperties=bold, fontsize=12.5, pad=7)
        ax.set_xlabel('클래스당 학습 라벨 수 n', fontproperties=font, fontsize=9.6)
        ax.grid(axis='y', color='#D9E1E8', lw=.75); ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='y', labelsize=8.8)
    axes[0].set_ylabel('macro-F1 (%) · 전체 9종', fontproperties=font, fontsize=9.6)
    axes[1].legend(frameon=False, loc='lower right', prop=font)
    fig.subplots_adjust(left=.075, right=.975, top=.9, bottom=.165, wspace=.09)
    fig.savefig(ASSETS / 'fewshot_csv_netfound.png', facecolor='white')
    plt.close(fig)


# ----------------------------------------------------------------------------- slides
def scenes(packet_slide: dict) -> None:
    page = r.page
    rows = paired()
    saturation_chart(); fewshot_chart(rows)

    # 1. Where we stand.
    a = [txt(.65, 1.08, 12.0, 'CSV 입력의 한계를 확인하고, 패킷 원본에서 feature를 직접 만드는 실험으로 전환', 17, False, GREY)]
    a += card(.65, 1.70, 3.85, 3.72, '이전 실험', 'CICFlowMeter CSV 벡터 입력\n\nXGB · TabPFN\nSOTA · 제안 모델\n\n평가 시나리오 표현 시도', accent=NAVY, body_size=19)
    a += [arrow(4.60, 3.42, .30)]
    a += card(5.00, 1.70, 3.85, 3.72, '확인한 문제', '분류 성능 부진\n\nfeature가 같은데 라벨이 다른 행\n\nfew-shot에서도 일부 공격의\n성능이 높아 개선 여지 부족', accent=RED, body_size=19)
    a += [arrow(8.95, 3.42, .30)]
    a += card(9.35, 1.70, 3.33, 3.72, '이번 보고', 'CSV 대신 PCAP 사용\n\nnetFound로 feature\n네트워크 데이터용\ntransformer 인코더', accent=TEAL, body_size=19)
    a += [box(.65, 5.68, 12.03, .90, 'F6F9FC', 'D3DFE9'), txt(.90, 5.94, 11.5, '검증 질문  패킷 기반 feature로 앞의 문제가 개선되는가', 20, True, NAVY, 'center')]
    page('11/06  CSV에서 PCAP으로', a, 'CSV 대신 PCAP과 netFound 인코더로 feature를 만들어 앞의 문제가 개선되는지 실험', '01 배경',
         '이전 실험은 CICFlowMeter CSV 벡터로 XGB, TabPFN, SOTA, 제안 모델을 돌렸고 평가 시나리오를 표현하려 했다. 그런데 분류 성능이 잘 나오지 않았고, 데이터 자체에 feature가 일치하는데 라벨이 다른 문제가 있었으며, few-shot인데도 일부 공격은 성능이 너무 높아 더 개선할 여지가 적었다. 그래서 이번 보고에서는 CSV 대신 PCAP을 쓰고 netFound 같은 네트워크 데이터용 transformer 인코더로 feature를 가공해서 앞의 문제가 개선되는지 실험한 결과를 보고한다.', [])

    # 2. How CIC data is made and why conflicts appear.
    a = [txt(.65, 1.04, 12.0, '실험망에서 PCAP을 수집하고, CICFlowMeter가 패킷을 flow로 묶어 미리 정한 통계를 계산', 16, False, GREY)]
    stages = [(.65, 2.55, '실험망', '정상·공격 통신'), (3.55, 2.55, 'PCAP', '시간순 원시 패킷'), (6.45, 3.15, 'CICFlowMeter', 'flow 묶기 · 고정 통계 계산'), (9.95, 2.73, 'CSV', '통계 벡터 + 라벨')]
    for x, w, title, body in stages:
        a += block(x, 1.42, w, .92, title, body, body_size=13.5)
    for x in (3.24, 6.14, 9.64):
        a.append(arrow(x, 1.76, .27))
    a.append(txt(.65, 2.44, 12.03, '고정 통계  flow마다 같은 계산식 78개  ·  지속 시간 · 보낸·받은 패킷 수 · 패킷 길이 평균 · 패킷 간격 평균 · FIN 수', 14.5, True, NAVY))
    a.append(txt(.65, 2.86, 12.0, '같은 feature에 다른 라벨이 생기는 경로', 17, True, NAVY))
    a += card(.65, 3.24, 4.15, 1.72, '① flow 경계 결함 (주원인)', '첫 FIN에서 끊고 RST 무시\n남은 ACK·FIN이 별도 flow\n이런 flow가 CIC2017의 25.9%', accent=RED, body_size=13.5)
    a.append(arrow(4.87, 3.96, .24))
    a += card(5.18, 3.24, 3.62, 1.72, '② 패킷이 매우 적은 flow', '남은 패킷 1~2개로 계산한 통계\n값이 쉽게 일치', accent=NAVY, body_size=13.5)
    a.append(arrow(8.87, 3.96, .24))
    a += card(9.18, 3.24, 3.50, 1.72, '③ 라벨은 원래 flow를 따름', '같은 모양의 꼬리 flow에\n정상과 공격 라벨이 혼재', accent=NAVY, body_size=13.5)
    a.append(txt(.65, 5.03, 12.03, '근거  Engelen 2021 · Liu 2022  ·  PCAP 재구성에서 꼬리 flow를 빼면 통계 충돌 6,367 → 69', 14, False, GREY))
    g17 = AUDIT17['representative_conflicting_groups'][0]; g18 = AUDIT18['representative_conflicting_groups'][0]
    p17 = {x['label']: x['rows'] for x in g17['per_label_rows']}; p18 = {x['label']: x['rows'] for x in g18['per_label_rows']}
    ex = [('CIC2017', '목적 포트 80 · 1 µs · 보낸 2 · 받은 0', f"BENIGN {p17['BENIGN']:,}행  ·  DoS Hulk {p17['DoS Hulk']:,}행"),
          ('CIC2018', '목적 포트 21 · 2 µs · 보낸 1 · 받은 1', f"SlowHTTPTest {p18['DoS attacks-SlowHTTPTest']:,}행  ·  FTP {p18['FTP-BruteForce']:,}행  ·  SSH {p18['SSH-Bruteforce']:,}행")]
    a += table(.65, 5.42, [1.30, 4.15, 6.58], ['데이터', '통계가 모두 같은 flow의 예', '붙어 있는 라벨'], ex, size=13, row_h=.45, align=['left', 'left', 'left'])
    page('문제1  CIC-IDS 데이터의 동일 feature·상이 라벨', a, 'flow 경계 결함이 패킷이 매우 적은 flow를 만들고, 같은 통계에 서로 다른 라벨이 붙음', '02 문제',
         'CIC-IDS 데이터는 실험망에서 PCAP을 수집하고 CICFlowMeter가 패킷을 flow로 묶어 미리 정한 통계를 계산해 만든다. 고정 통계란 flow마다 똑같은 계산식 78개를 적용한다는 뜻으로, 지속 시간, 보낸 패킷 수, 패킷 길이 평균 같은 값이다. 동일 feature 상이 라벨의 주원인은 flow 경계 결함이다. Engelen 등은 CICFlowMeter가 첫 FIN에서 flow를 끝내고 RST를 종료로 보지 않아 남은 ACK와 FIN이 별도 flow가 되며, 이런 TCP appendix가 CIC2017의 25.9%이고 원래 flow의 라벨을 받는다고 보고했다. 이 꼬리 flow는 패킷이 1~2개라 통계가 쉽게 일치하는데 라벨은 원래 flow를 따르므로 같은 통계에 다른 라벨이 붙는다. 우리가 PCAP을 직접 열어 재구성했을 때도 꼬리 flow를 빼면 통계 충돌이 6,367개에서 69개로 줄었다. 표의 두 예는 통계가 모두 같은 실제 묶음이다.', ['engelen', 'liu'])

    # 3. Class-wise conflicts on both CSVs.
    f17 = {x['family']: x for x in AUDIT17['nfv3_family_audit']['per_family']}
    f18 = {x['family']: x for x in AUDIT18['nfv3_family_audit']['per_family']}
    def cell(x):
        return (f"{x['rows']:,}", f"{x['conflicting_rows']:,}", f"{100 * x['conflicting_fraction']:.2f}") if x else ('–', '–', '–')
    rows3 = []
    for fam in ('benign', 'dos', 'portscan', 'ddos', 'bf', 'web', 'bot', 'inf', 'heartbleed'):
        rows3.append((fam, *cell(f17.get(fam)), *cell(f18.get(fam))))
    t17, t18 = AUDIT17, AUDIT18
    rows3.append(('합계', f"{t17['totals']['rows']:,}", f"{t17['nfv3_family_audit']['rows_in_conflicting_groups']:,}", f"{100 * t17['nfv3_family_audit']['fraction_rows_in_conflicting_groups']:.2f}",
                  f"{t18['totals']['rows']:,}", f"{t18['nfv3_family_audit']['rows_in_conflicting_groups']:,}", f"{100 * t18['nfv3_family_audit']['fraction_rows_in_conflicting_groups']:.2f}"))
    a = [txt(.65, 1.08, 12.0, '원본 CSV 전체 행 기준  ·  통계가 완전히 같은데 class가 다른 묶음에 속한 행  ·  행 순서는 CIC2017 행 수 내림차순', 15.5, False, GREY)]
    a += table(.65, 1.56, [1.65, 1.95, 1.60, 1.40, 2.15, 1.60, 1.68], ['class', 'CIC2017 행', '충돌 행', '비율 %', 'CIC2018 행', '충돌 행', '비율 %'], rows3, size=13.5, row_h=.395, bold_last=True)
    a.append(txt(.65, 6.10, 12.0, 'CIC2018은 bf 42.65% · inf 25.49% · dos 11.81%로 공격 class 충돌이 큼  ·  CIC2017은 dos 2.07%가 가장 큼', 15, True, NAVY))
    a.append(txt(.65, 6.48, 12.0, 'PCAP 크기 때문에 실험은 CIC2017로 진행 (CIC2018 450 GB, CIC2017 51 GB)', 14, False, GREY))
    a.append(box(7.225, 1.50, .05, 4.52, NAVY))
    page('문제1  CIC2017·2018 CSV의 class별 충돌', a, '충돌 행은 CIC2018 2.51%, CIC2017 0.25%이고, CIC2018은 공격 class에 크게 집중', '02 문제',
         'CIC2017과 CIC2018 원본 CSV에서 통계가 완전히 같은데 class가 다른 행을 class별로 셌다. CIC2017은 전체 0.25%인 7,020행이고 dos가 2.07%로 가장 크다. CIC2018은 전체 2.51%인 406,811행이고 bf 42.65%, inf 25.49%, dos 11.81%로 공격 class에 집중된다. CIC2018 전체 PCAP은 450 GB라 내려받지 못했고, 전체 망을 하루 한 파일로 캡처한 CIC2017 51 GB로 실험했다.', ['cic17', 'cic18'])

    # 4. Early saturation on CIC2017 CSV.
    c20, c50 = csv_class('attack_only', 'tabpfn', 20), csv_class('attack_only', 'tabpfn', 50)
    a = [txt(.65, 1.08, 12.0, 'CIC2017 CSV  ·  공격만 학습·평가  ·  클래스당 라벨 n개  ·  seed 5개 평균', 16, False, GREY)]
    a.append(image('saturation_cic2017', .55, 1.50, 12.25, 4.25))
    a.append(txt(.65, 5.92, 12.0, f"n=20에서 portscan {c20['portscan']:.0f}% · ddos {c20['ddos']:.0f}% · dos {c20['dos']:.0f}%  ·  n=50에서 {c50['portscan']:.0f}% · {c50['ddos']:.0f}% · {c50['dos']:.0f}%", 16, True, NAVY))
    a.append(txt(.65, 6.32, 12.0, f"반면 web은 n=50에서도 {c50['web']:.0f}%, bot {c50['bot']:.0f}%는 CSV 라벨 결함  ·  평균 성능만으로는 난이도 판단 불가", 15, False, INK))
    page('문제2  few-shot에서도 일부 공격은 이미 높은 성능', a, '대량·반복 공격은 라벨 20개에서 이미 높아 라벨을 늘려도 개선 여지가 적음', '02 문제',
         'CIC2017 CSV에서 공격만 학습하고 평가한 few-shot 곡선이다. portscan, ddos, dos 같은 대량 공격은 클래스당 라벨 20개에서 이미 높은 F1에 도달해 라벨을 늘려도 개선 여지가 적다. 반면 web은 낮고, bot의 0은 평가 집합의 bot이 모두 master 종료 뒤 재접속 flow인 CSV 라벨 결함 때문이다. 이전 발표의 그림은 CIC2018 기준이었고 실험 데이터에 맞춰 CIC2017로 다시 그렸다.', [])

    # 5. Pipeline in true order.
    a = [txt(.65, 1.04, 12.0, '라벨은 feature 계산에 쓰지 않음  ·  PCA를 학습 데이터로만 맞추기 위해 시간순 분할이 PCA보다 먼저', 16, False, GREY)]
    top = [(.65, 2.55, '① PCAP', 'CIC2017 월~금\n하루 1개, 파일 5개\n51 GB'),
           (3.45, 3.25, '② flow로 묶기', '프로토콜·IP·포트 쌍이 같은 패킷\nRST · 양쪽 FIN 뒤 ACK\n새 SYN · 120초에서 종료'),
           (6.95, 2.85, '③ 라벨 붙이기', '공격자·피해자 IP인가\n공격 시간창 안인가\n공격 행위 증거가 있는가'),
           (10.05, 2.63, '④ 시간순 분할', '날짜·공격별로\n앞 70% 학습\n다음 15% · 뒤 15% 평가')]
    for x, w, title, body in top:
        a += block(x, 1.56, w, 2.30, title, body, body_size=15)
    for x in (3.21, 6.71, 9.81):
        a.append(arrow(x, 2.50, .23))
    a.append(arrow(11.25, 3.93, .24, .22, down=True))
    a += block(10.05, 4.28, 2.63, 2.40, '⑧ 평가', '', body_size=15)
    low = [(.65, 3.65, '⑤ 토큰화', 'IP  헤더 길이·TOS·전체 길이·flag·TTL\nTCP  flag·window·seq·ack·urgent\nIP·포트·시각·payload 제외'),
           (4.55, 2.45, '⑥ netFound', '토큰열 → 512차원\n가중치 고정'),
           (7.25, 2.45, '⑦ PCA', '512차원 → 64차원\n학습 데이터로 맞춤')]
    for x, w, title, body in low:
        a += block(x, 4.28, w, 2.40, title, body, body_size=15)
    a.append(arrow(4.31, 5.30, .23)); a.append(arrow(7.01, 5.30, .23)); a.append(arrow(9.75, 5.30, .25))
    page('PCAP에서 feature까지', a, '패킷을 flow로 묶고 라벨을 붙인 뒤, 헤더 토큰을 인코더에 넣어 flow당 64차원 feature를 생성', '03 파이프라인',
         'CIC2017은 스위치에서 전체 망을 월요일부터 금요일까지 하루 한 파일로 캡처해 PCAP이 5개이고 합쳐 51 GB다. 패킷을 flow로 묶고 flow마다 라벨을 붙인다. 그다음 날짜와 공격별로 시간순으로 나눈다. 분할이 feature보다 먼저인 이유는 PCA를 학습 데이터로만 맞추기 위해서다. 라벨은 feature 계산에 쓰이지 않으므로 라벨을 먼저 붙여도 feature에는 영향이 없다. 그 뒤 헤더를 토큰으로 바꿔 netFound에 넣고 PCA로 64차원을 얻어 평가한다.', ['netfound'])

    # 6. Packet.
    P = WALK['packet']
    a = [txt(.65, 1.08, 12.0, f"패킷은 네트워크로 오가는 데이터 조각 하나  ·  아래는 월요일 사내 PC가 웹서버로 보낸 실제 패킷 {P['frame_bytes']}바이트", 16, False, GREY)]
    unit, x = 12.03 / P['frame_bytes'], .65
    for name, size, color in (('이더넷', P['ethernet_bytes'], GREY), ('IP 헤더', P['ip_header_bytes'], NAVY), ('TCP 헤더', P['tcp_header_bytes'], BLUE), ('payload', P['payload_bytes'], GREEN)):
        a += chip(x, 1.60, size * unit - .03, f'{name} {size} B', color, 'FFFFFF', 12.5, .46)
        x += size * unit
    a += card(.65, 2.28, 2.35, 3.75, '이더넷 14 B', '바로 옆 장비까지\n전달하기 위한 표\n\n보낸 장비 번호\n받는 장비 번호', accent=GREY, body_size=15)
    a += card(3.12, 2.28, 3.05, 3.75, 'IP 헤더 20 B', f"어느 컴퓨터로 갈지 적는 칸\n\n보낸 주소  {P['src_ip']}\n받는 주소  {P['dst_ip']}\n전체 길이  {P['ip_total_length']} B\nTTL  {P['ttl']}\n프로토콜  {P['protocol']}번 (TCP)", accent=NAVY, body_size=14.5, align='left')
    a += card(6.29, 2.28, 3.05, 3.75, 'TCP 헤더 20 B', f"어느 프로그램으로 갈지 적는 칸\n\n보낸 포트  {P['src_port']}\n받는 포트  {P['dst_port']}\nseq  {P['seq']}\nack  {P['ack']}\n신호  {P['tcp_flags']}\nwindow  {P['window']}", accent=BLUE, body_size=14.5, align='left')
    a += card(9.46, 2.28, 3.22, 3.75, 'payload 97 B', '실제 내용\n\n' + '\n'.join(P['payload_lines']), accent=GREEN, body_size=14.5, align='left')
    page('패킷의 생김새', a, packet_slide['summary'], '03 파이프라인', packet_slide['notes'], packet_slide['sources'])

    # 8. What makes one flow.
    a = [txt(.65, 1.08, 12.0, 'flow = 두 컴퓨터의 두 프로그램이 같은 약속으로 주고받은 패킷 묶음, 즉 대화 한 건', 16, False, GREY)]
    keys = ['프로토콜', 'IP A', '포트 A', 'IP B', '포트 B']
    for i, k in enumerate(keys):
        a += chip(.65 + i * 2.05, 1.62, 1.90, k, NAVY, 'FFFFFF', 19, .72)
    a.append(txt(10.95, 1.74, 1.75, '다섯 값이\n모두 같으면', 15, True, RED, 'center'))
    a.append(txt(.65, 2.54, 12.03, '같은 flow  ·  보낸 쪽과 받는 쪽이 바뀌어도 같은 flow', 22, True, RED, 'center'))
    conv = [('요청', 'TCP', '192.168.10.9 : 1057', '69.31.33.224 : 80', 'flow A'), ('응답', 'TCP', '69.31.33.224 : 80', '192.168.10.9 : 1057', 'flow A  방향만 반대'),
            ('인증 접속 1', 'TCP', '192.168.10.9 : 1055', '192.168.10.3 : 88', 'flow B'), ('인증 접속 2', 'TCP', '192.168.10.9 : 1056', '192.168.10.3 : 88', 'flow C  임시 포트가 다름'),
            ('도메인 조회', 'UDP', '192.168.10.9 : 59285', '192.168.10.3 : 53', 'flow D  프로토콜과 상대가 다름')]
    a += table(.65, 3.24, [1.80, 1.10, 2.75, 2.75, 3.63], ['패킷', '프로토콜', '보낸 쪽', '받는 쪽', '묶인 결과'], conv, size=13.5, row_h=.44, align=['left', 'left', 'left', 'left', 'left'],
               styles={0: ('E6F1F8', True, NAVY), 1: ('E6F1F8', True, NAVY)})
    a.append(txt(.65, 6.02, 12.03, '묶는 이유  패킷 하나로는 행동을 알 수 없고, 대화 전체를 봐야 요청·응답·종료를 판단 가능', 15, True, NAVY))
    a.append(txt(.65, 6.42, 12.03, '다섯 값이 같아도 대화가 끝나면 다음 패킷은 새 flow  ·  끝나는 조건은 다음 장', 14, False, GREY))
    page('같은 대화의 패킷을 한 flow로 묶기', a, '프로토콜과 양쪽 IP·포트 다섯 값이 같은 패킷이 한 flow', '03 파이프라인',
         '패킷 하나로는 무슨 일이 일어나는지 알기 어려워 대화 단위인 flow로 묶는다. 기준은 프로토콜과 양쪽 IP, 양쪽 포트 다섯 값이다. 보낸 쪽과 받는 쪽이 바뀌어도 같은 대화이므로 요청과 응답은 같은 flow다. 같은 서버의 같은 서비스라도 임시 포트가 다르면 다른 flow이고, 프로토콜이나 상대가 다르면 다른 flow다.', [])

    # 9. Where a flow ends; residual packets are dropped.
    a = [txt(.65, 1.08, 12.0, '→ PC가 보냄   ← 서버가 보냄   ·   소속 칸이 각 패킷의 처리   ·   두 대화 모두 월요일 실제 패킷', 15.5, False, GREY)]
    head = ['#', '시각 (µs)', '방향', 'TCP flag', '길이', 'payload', '소속']
    w7 = [.40, 1.02, .52, 1.00, .50, .80, 1.66]
    al7 = ['right', 'right', 'center', 'left', 'right', 'right', 'left']
    pk = [('1', '0', '→', 'SYN', '52', '0', 'flow A'), ('2', '19,958', '←', 'SYN+ACK', '52', '0', 'flow A'), ('3', '20,008', '→', 'ACK', '40', '0', 'flow A'),
          ('4', '20,083', '→', 'ACK+PSH', '137', '97', 'flow A'), ('5', '40,027', '←', 'ACK', '40', '0', 'flow A'), ('6', '40,640', '←', 'ACK+PSH', '219', '179', 'flow A'),
          ('7', '40,643', '←', 'ACK+FIN', '40', '0', 'flow A'), ('8', '40,685', '→', 'ACK', '40', '0', 'flow A'), ('9', '40,689', '→', 'ACK+FIN', '40', '0', 'flow A'), ('10', '60,574', '←', 'ACK', '40', '0', 'flow A 마감')]
    a.append(txt(.65, 1.49, 5.9, '케이스 1  FIN으로 끝남  ·  192.168.10.9:1057 ↔ 69.31.33.224:80', 13.5, True, NAVY))
    a += table(.65, 1.80, w7, head, pk, size=12, row_h=.335, align=al7, styles={9: ('D5E8F4', True, NAVY)})
    ex = [('1', '0', '→', 'SYN', '60', '0', 'flow B'), ('2', '42,246', '←', 'SYN+ACK', '60', '0', 'flow B'), ('3', '42,350', '→', 'ACK', '52', '0', 'flow B'),
          ('4', '5,079,690', '→', 'ACK+FIN', '52', '0', 'flow B'), ('5', '5,084,903', '←', 'ACK+FIN', '264', '212', 'flow B'), ('6', '5,084,965', '→', 'RST', '40', '0', 'flow B 마감'),
          ('7', '5,121,931', '←', 'ACK', '52', '0', '버림'), ('8', '5,121,962', '→', 'RST', '40', '0', '버림')]
    a.append(txt(6.78, 1.49, 5.9, '케이스 2  RST로 끝남  ·  192.168.10.51:38488 ↔ 209.15.36.31:80', 13.5, True, NAVY))
    a += table(6.78, 1.80, w7, head, ex, size=12, row_h=.335, align=al7, styles={5: ('D5E8F4', True, NAVY), 6: ('F8E1E0', True, RED), 7: ('F8E1E0', True, RED)})
    a.append(txt(6.78, 4.90, 5.9, 'PC가 4번 FIN으로 먼저 닫아 도착하는 5·7번마다 RST로 응답', 13, True, NAVY))
    a.append(txt(6.78, 5.19, 5.9, '첫 RST까지가 한 flow  ·  뒤따르는 패킷은 앞 대화의 꼬리라 버림', 13, True, RED))
    a += card(.65, 5.60, 3.75, 1.30, 'TCP 신호', 'SYN 연결 요청  ·  ACK 받았다는 확인\nPSH 내용 바로 전달  ·  FIN 보낼 것이 끝남\nRST 연결을 버림, 확인 없이 즉시 종료', accent=BLUE, body_size=12.5)
    a += card(4.52, 5.60, 3.75, 1.30, '버리는 꼬리 패킷', '닫힌 연결에서 2초 안에 도착\nSYN·payload 없는 패킷 2개 이하', accent=RED, body_size=12.5)
    a += card(8.39, 5.60, 4.29, 1.30, 'flow가 끝나는 때', 'RST  ·  양쪽 FIN 뒤 ACK  ·  같은 끝점의 새 SYN\n120초 무통신  ·  시작 후 120초\nUDP·ICMP는 시간 조건만 사용', accent=NAVY, body_size=12.5)
    page('flow의 끝: FIN과 RST', a, 'FIN 종료는 마지막 ACK까지, RST 종료는 첫 RST까지가 한 flow이고 뒤따르는 패킷은 버림', '03 파이프라인',
         '케이스 1의 flow A는 7번과 9번의 FIN으로 양쪽이 종료를 알리고 10번 ACK에서 닫힌다. 케이스 2의 flow B는 PC가 4번 FIN으로 먼저 닫은 경우다. 서버의 5번과 7번이 뒤늦게 도착하자 PC는 이미 닫은 연결이므로 패킷마다 RST로 답한다. RST가 오면 flow를 닫으므로 6번까지를 flow B로 본다. 근거는 두 가지다. TCP 규격에서 RST는 연결을 재설정하는 신호이고, Engelen 등은 CICFlowMeter가 RST를 종료로 보지 않는 것을 결함으로 지적하고 RST에서 flow를 끝내도록 고쳤다. 뒤의 7번과 8번은 앞 대화의 꼬리이므로 버린다. 이렇게 버리는 패킷이 만들던 잔여 flow가 전체의 11.7%인 284,600개였고, 이것이 충돌의 주요 원인이었다. 데이터 감소는 라벨 flow 6.5%, 패킷 0.57%이고 소수 class는 줄지 않는다.', ['engelen'])

    # 10. Tokens, netFound, PCA.
    tk = WALK['tokens']
    a = [txt(.65, 1.08, 12.0, 'flow의 패킷마다 헤더 값을 숫자 토큰으로 옮겨 netFound에 입력  ·  아래는 케이스 1 flow A의 패킷', 16, False, GREY)]
    head10 = ['패킷', 'IP 헤더\n길이', 'TOS', '전체\n길이', 'IP\nflag', 'TTL', 'TCP\nflag', 'window', 'seq\n앞', 'seq\n뒤', 'ack\n앞', 'ack\n뒤', 'urgent']
    rows10 = [['1번 SYN'] + [str(v) for v in tk['first_syn']], ['2번 SYN+ACK'] + [str(v) for v in tk['second_syn_ack']], ['4번 요청'] + [str(v) for v in tk['fourth_request']], ['…'] + ['…'] * 12]
    a += table(.65, 1.54, [1.71] + [.86] * 12, head10, rows10, size=12, row_h=.40)
    a.append(txt(.65, 3.58, 12.03, 'flow 한 건의 헤더 토큰  ·  패킷 1개 = 숫자 12개  ·  IP·포트·시각·payload 내용 제외', 14.5, True, NAVY, 'center'))
    a.append(arrow(6.54, 3.93, .26, .24, down=True))
    steps10 = [(.65, 2.75, '헤더 토큰', 'flow의 패킷 순서대로\n간격·방향 함께 입력'), (3.71, 3.15, 'netFound', '사전학습 transformer\n가중치 고정'), (7.17, 2.45, 'PCA', '512차원 → 64차원\n학습 데이터로 맞춤'), (9.93, 2.75, 'feature', 'flow 한 건 = 64개\n분류기의 한 행')]
    for x, w, title, body in steps10:
        a += block(x, 4.28, w, 1.34, title, body, color=(RED if title == 'feature' else NAVY), body_size=14)
    for x in (3.44, 6.90, 9.66):
        a.append(arrow(x, 4.85, .23))
    a += [box(.65, 5.80, 12.03, .96, 'F6F9FC', 'D3DFE9'), txt(.90, 5.95, 11.5, 'netFound  네트워크 트래픽용 사전학습 transformer', 15.5, True, NAVY, 'center'),
          txt(.90, 6.30, 11.5, '라벨 없는 트래픽에서 가려진 헤더 토큰을 맞히며 학습, 본 실험은 인코더로만 사용', 14, False, INK, 'center')]
    page('헤더 토큰에서 netFound feature로', a, 'flow의 헤더 값을 토큰으로 netFound에 넣어 flow 한 건당 feature 64개를 얻음', '03 파이프라인',
         '각 flow는 앞의 표처럼 패킷의 나열이다. 패킷마다 헤더의 칸 값을 숫자 토큰으로 옮기면 패킷 하나가 숫자 12개가 되고, seq와 ack는 상대값으로 바뀐다. 이 토큰을 패킷 순서대로 netFound에 넣는다. netFound는 네트워크 트래픽용으로 사전학습된 transformer로, 라벨 없는 트래픽에서 가려진 헤더 토큰을 맞히며 학습했다. 본 실험은 가중치를 고정한 인코더로 쓴다. flow 한 건을 넣으면 512차원 벡터가 나오고, 학습 데이터로 맞춘 PCA로 64차원이 된다. 그래서 flow 한 건이 feature 64개짜리 한 행이 된다.', ['netfound'])

    # 11. Labels as a decision flow.
    a = [txt(.65, 1.08, 12.0, '각 flow를 왼쪽부터 판정  ·  CNS 2022의 교정 규칙 53개를 그대로 적용', 16, False, GREY)]
    a += chip(.65, 1.62, 1.00, 'flow', GREY, 'FFFFFF', 16, 1.00)
    qs = [(2.02, '공격자 IP ↔\n피해자 IP인가'), (4.99, '공격 시간창\n안인가'), (7.96, '공격 행위의\n증거가 있는가')]
    for x, q in qs:
        a.append(box(x, 1.62, 2.55, 1.00, NAVY))
        a.append(txt(x + .1, 1.82, 2.35, q, 15.5, True, 'FFFFFF', 'center'))
    for x in (1.68, 4.60, 7.57, 10.54):
        a.append(arrow(x, 2.00, .31))
    for x in (4.60, 7.57, 10.54):
        a.append(txt(x - .02, 1.70, .36, '예', 13, True, BLUE, 'center'))
    a += chip(10.90, 1.62, 1.78, '공격', RED, 'FFFFFF', 20, 1.00)
    for x in (3.17, 6.14, 9.11):
        a.append(arrow(x, 2.68, .25, .28, down=True))
    a.append(txt(3.47, 2.70, .8, '아니오', 12.5, True, GREY))
    a.append(txt(6.44, 2.70, .8, '아니오', 12.5, True, GREY))
    a.append(txt(9.41, 2.70, .8, '없음', 12.5, True, GREY))
    a += chip(2.02, 3.02, 5.52, '정상', BLUE, 'FFFFFF', 20, .70)
    a += chip(7.96, 3.02, 2.55, '보류', GREY, 'FFFFFF', 20, .70)
    three = [('공격', 'IP·시간창·증거를 모두 만족', '학습·평가에 공격으로 사용'), ('보류', 'IP·시간창은 맞지만 증거 없음', '학습·평가에서 제외'), ('정상', 'IP나 시간창이 규칙 밖', '학습·평가에 정상으로 사용')]
    a += table(.65, 3.96, [1.00, 3.55, 3.05], ['라벨', '조건', '처리'], three, size=13.5, row_h=.46, align=['left', 'left', 'left'],
               styles={0: ('F8E1E0', True, RED), 1: ('EEF1F4', False, GREY), 2: ('E6F1F8', False, NAVY)})
    a += card(8.45, 3.96, 4.23, 2.80, 'CNS 2022  Liu 외', 'Error Prevalence in NIDS Datasets\n\nCIC-IDS 2017·2018의 flow 생성과\n라벨 오류를 패킷 단위로 조사\n교정한 라벨 규칙과 도구를 공개', accent=BLUE, body_size=13.5)
    a.append(txt(.65, 5.92, 7.6, '증거 예  FTP·SSH brute force는 공격자 payload ≥ 1 B', 14, False, INK))
    a.append(txt(.65, 6.26, 7.6, '증거 예  Portscan은 SYN 1개와 무응답 또는 RST', 14, False, INK))
    page('flow에 라벨 붙이기', a, 'IP·시간창·공격 행위 증거 세 조건으로 공격, 보류, 정상을 나누고 보류는 학습·평가에서 제외', '04 라벨',
         '라벨은 앞에서 만든 flow마다 왼쪽부터 판정한다. CNS 2022는 Liu 등이 CIC-IDS 2017과 2018의 flow 생성과 라벨 오류를 패킷 단위로 조사한 논문으로, 교정한 라벨과 도구를 공개했다. 본 실험은 이 규칙 53개를 그대로 적용했다. 공격자와 피해자 IP가 아니거나 시간창 밖이면 정상이고, 둘 다 맞아도 공격 행위의 증거가 없으면 보류이며, 세 조건을 모두 만족하면 공격이다. 보류는 공격 효과가 없는 시도로, 저자 권고에 따라 독립 class로 쓰지 않고 학습과 평가에서 제외한다.', ['liu', 'attempted'])

    # 12. Final composition after deletion.
    comp = [('benign', '월요일 전일 + 공격일 배경', '2,273,097', '1,609,980'), ('portscan', '금 Portscan 161,202 · 목 NMAP 76,947', '158,930', '238,149'),
            ('dos', 'Hulk 158,546 · GoldenEye 7,567 · Slowloris 3,859 · Slowhttptest 1,741', '252,661', '171,713'), ('ddos', 'LOIC-HTTP', '128,027', '95,683'),
            ('bf', 'FTP 4,021 · SSH 2,961', '13,835', '6,982'), ('bot', 'Ares', '1,966', '736'), ('web', 'Brute Force 73 · XSS 18 · SQL Injection 13', '2,180', '104'),
            ('inf', 'Communication 32 · CoolDisk 4', '36', '36'), ('heartbleed', 'Heartbleed', '11', '11'), ('합계', '', '2,830,743', '2,123,394')]
    a = [txt(.65, 1.08, 12.0, '잔여 flow 삭제 후 분할 전 기준  ·  CSV는 공식 라벨 행 수, PCAP은 라벨 있는 flow 수  ·  행 순서는 PCAP flow 수 내림차순', 15, False, GREY)]
    a += table(.65, 1.55, [1.50, 7.13, 1.70, 1.70], ['class', '세부 공격 (PCAP flow 수)', 'CSV 행', 'PCAP flow'], comp, size=12.5, row_h=.405, align=['left', 'left', 'right', 'right'], bold_last=True)
    a.append(txt(.65, 6.20, 12.0, 'CSV와 수가 다른 이유  flow 경계와 라벨 규칙 교정  ·  목요일 NMAP은 CSV에서 BENIGN, 교정 규칙에서 portscan', 14, False, INK))
    a.append(txt(.65, 6.55, 12.0, 'inf는 학습 25개·test 6개, heartbleed는 학습 7개·test 2개로 표본이 매우 적음', 14, False, GREY))
    page('최종 데이터 구성', a, '라벨 있는 flow 2,123,394개, 9종을 학습·평가', '04 라벨',
         '이렇게 얻은 데이터 구성이다. 잔여 flow를 삭제한 기준이고 라벨 있는 flow는 2,123,394개다. CSV와 수가 다른 이유는 flow 경계와 라벨 규칙을 고쳤기 때문이다. 예를 들어 목요일 NMAP 스캔은 CSV에서 BENIGN이었지만 교정 규칙에서는 portscan이다. inf와 heartbleed는 학습 표본이 라벨 예산 50개보다 적어 있는 만큼만 쓰고, test가 6개와 2개라 점수의 변동이 크다.', ['liu'])

    # 13. Problem 1 again, with side information folded in.
    dr = COUNTS['counts']['drop']['L2']
    a = [txt(.65, 1.08, 12.0, '잔여 flow 삭제 후  ·  feature가 같은데 class가 다른 flow  ·  패킷이 매우 적은 flow에 집중', 16, False, GREY)]
    head13 = [('CSV 통계 78개', f"{AUDIT17['nfv3_family_audit']['rows_in_conflicting_groups']:,}", f"{100 * AUDIT17['nfv3_family_audit']['fraction_rows_in_conflicting_groups']:.2f}%"), ('netFound', f"{dr['conflict']:,}", f"{dr['conflict_pct']:.2f}%")]
    a += table(.65, 1.56, [3.05, 1.40, 1.25], ['feature', '충돌 flow', '비율'], head13, size=14, row_h=.46, styles={1: ('F8E1E0', True, RED)})
    a.append(txt(.65, 3.04, 5.7, '남은 충돌  Portscan 42,609 · 정상 770 · DoS 1', 14, True, NAVY))
    bins = {'1': '1', '2': '2', '3-5': '3~5', '6-20': '6~20', '21+': '21 이상'}
    rows13 = [(bins[k], f"{v['rows']:,}", f"{v['same_pct']:.1f}", f"{v['conflict']:,}") for k, v in dr['by_packets'].items()]
    a += table(6.75, 1.56, [1.40, 1.55, 1.48, 1.50], ['패킷 수', 'flow 수', '입력 같은 %', '충돌 flow'], rows13, size=13.5, row_h=.335, styles={1: ('F8E1E0', True, RED)})
    a += card(.65, 3.72, 5.90, 3.04, '원인', '패킷이 매우 적은 flow는 숫자가 적고 대부분 고정값\n예  스캔 SYN과 거절 RST, 패킷 2개\n\nnetFound가 입력 전에 버리는 정보\nIP · 포트 · 시각 · payload 내용', accent=RED, body_size=14.5)
    side_rows = SIDE['none']['rows']; side_base = SIDE['none']['conflict']
    side_pct = lambda key: 100 * SIDE[key]['conflict'] / side_rows
    side_cut = lambda key: 100 * (1 - SIDE[key]['conflict'] / side_base)
    def side_conflict(name, key):
        return (name, f"{SIDE[key]['conflict']:,}", f"{side_pct(key):.2f}%", ('–' if key == 'none' else f"{side_cut(key):.1f}%"))
    rows13b = [side_conflict('netFound', 'none'), side_conflict('+ 원본 seq 표시', 'seq_ack_arm'), side_conflict('+ 최근 60초 포트 수', 'ports_arm'), side_conflict('+ 둘 다', 'all_three')]
    a += card(6.75, 3.72, 5.93, 3.04, '추가 정보로 극복 가능', 'netFound 벡터 뒤에 열을 붙여 분류기에 입력, IP·포트 값은 넣지 않음', accent=TEAL, body_size=14)
    a += table(6.95, 4.62, [2.33, 1.20, 1.00, 1.00], ['입력', '충돌 flow', '비율', '감소'], rows13b, size=13, row_h=.33, styles={2: ('D5E8F4', False, NAVY)}, bold_cells={(3, 1), (3, 2), (3, 3)})
    a.append(txt(6.93, 6.36, 5.57, f"최근 60초 포트 수 하나로 충돌 flow {side_cut('ports_arm'):.1f}% 감소", 13.5, True, NAVY, 'center'))
    page('문제1 재점검: 동일 feature·상이 라벨', a, f"남은 충돌 {dr['conflict_pct']:.2f}%는 패킷이 매우 적은 flow와 netFound가 버리는 정보에서 발생, 추가 정보로 {side_pct('ports_arm'):.2f}%까지 감소", '05 결과',
         '잔여 flow를 삭제하기로 했으므로 삭제 후 기준으로 본다. netFound feature의 충돌은 2.04%로 CSV의 0.25%보다 크다. 원인은 두 가지가 겹친 것이다. 패킷이 매우 적은 flow는 남는 숫자가 적고 대부분 고정값이라 겹치기 쉽고, netFound는 입력 전에 IP, 포트, 시각, payload 내용을 버린다. 충돌 flow 43,380개 중 42,443개가 패킷 2개 flow이고 12 burst 제한과는 무관하다. 이 충돌은 추가 정보로 극복할 수 있다. 공개 가중치는 입력 구성이 고정되어 있으므로 인코더는 그대로 두고 벡터 뒤에 열을 붙인다. 최근 60초 포트 수는 같은 두 IP 사이의 포트 종류 수로, 충돌 flow를 43,380개에서 964개로 97.8% 줄인다. 원본 seq 표시는 42,571개로 1.9%만 줄인다. 다만 충돌 수는 라벨이 섞인 입력의 flow를 전부 세는 값이고, 입력마다 가장 많은 라벨을 골라도 틀리는 flow는 212개에서 64개로 줄어든다. 둘을 함께 쓰면 302개, 0.01%다.', [])

    # 15. Few-shot on all nine classes.
    d_t, d_x = ac_curve('netfound', 'tabpfn'), ac_curve('netfound', 'xgb')
    c_t, c_x = ac_curve('csv', 'tabpfn'), ac_curve('csv', 'xgb')
    a = [txt(.65, 1.08, 12.0, '전체 9종  ·  시간순 분할  ·  seed 5개  ·  두 모델에 같은 학습 행  ·  netFound는 잔여 flow 삭제 데이터', 16, False, GREY)]
    a.append(image('fewshot_csv_netfound', .50, 1.46, 7.75, 3.40))
    n20 = [('CSV', f'{c_x[20][0]:.1f}', f'{c_t[20][0]:.1f}', f'{c_x[50][0]:.1f}', f'{c_t[50][0]:.1f}'), ('netFound', f'{d_x[20][0]:.1f}', f'{d_t[20][0]:.1f}', f'{d_x[50][0]:.1f}', f'{d_t[50][0]:.1f}')]
    a += table(8.40, 1.56, [1.06, .74, .86, .74, .88], ['macro-F1', 'XGB\nn=20', 'TabPFN\nn=20', 'XGB\nn=50', 'TabPFN\nn=50'], n20, size=12.5, row_h=.50)
    ov = BOUNDARY['fewshot_overlap_n50_netfound']
    diffs = [100 * (float(x['macro_drop']) - float(x['macro_exp94'])) for x in rows if x['arm'] == MAIN_ARM and x['regime'] == 'with_benign' and x['method'] == 'tabpfn' and int(x['budget']) == 50]
    gain = sum(diffs) / len(diffs)
    a += card(8.40, 3.20, 4.28, 1.66, 'CSV와 다른 점', "CSV는 두 모델이 비슷\nnetFound는 TabPFN이 크게 우위\n잔여 삭제 전후 7종 n=50 " + f"{gain:+.1f} pp".replace('-', '−'), accent=BLUE, body_size=14)
    a += card(.65, 5.00, 5.90, 1.82, 'TabPFN이 우위인 이유  n=50, 삭제 전 측정', f"학습과 같은 입력의 정확도는 두 모델 모두 {ov['acc_on_those']['tabpfn']:.1f}%\n나머지 행은 XGB {ov['acc_on_rest']['xgb']:.1f}% · TabPFN {ov['acc_on_rest']['tabpfn']:.1f}%\n처음 보는 입력에서 격차 발생\nPCA 벡터는 열 단위 분할에 불리 (추정)", accent=RED, body_size=14)
    dc = {k: 100 * v for k, v in ac('netfound', 'tabpfn', 50)['class_f1'].items()}
    a += card(6.78, 5.00, 5.90, 1.82, 'netFound + TabPFN n=50 class별 F1', f"benign {dc['benign']:.1f} · portscan {dc['portscan']:.1f} · dos {dc['dos']:.1f} · ddos {dc['ddos']:.1f} · bf {dc['bf']:.1f}\nbot {dc['bot']:.1f} · web {dc['web']:.1f} · inf {dc['inf']:.1f} · heartbleed {dc['heartbleed']:.1f}\ninf·heartbleed는 test 6개·2개, 재현율은 높고 정밀도 낮음", accent=NAVY, body_size=14)
    page('문제2 재점검: few-shot 성능', a, 'few-shot에서 netFound + TabPFN이 가장 높고, 격차는 학습과 겹치지 않는 입력에서 발생', '05 결과',
         f"전체 9종 few-shot 결과다. CSV에서는 XGB와 TabPFN이 비슷하지만 netFound feature에서는 TabPFN이 크게 앞서고, n=50 macro-F1은 {d_t[50][0]:.1f}으로 CSV의 {c_t[50][0]:.1f}보다 높다. 학습 행과 입력이 같은 test의 정확도는 두 모델이 같고, 격차는 처음 보는 입력에서 나온다. 해석하면 CSV의 열은 한 열씩 뜻이 있어 XGB의 축 방향 분할에 잘 맞지만, PCA 64차원은 열 하나의 뜻이 약해 벡터 전체를 비교하는 TabPFN이 유리한 것으로 보인다. 이 해석은 추정이다. inf와 heartbleed는 학습 표본이 25개와 7개뿐이고 test가 6개와 2개다. 두 class로 예측된 행은 seed 평균 7,089개인데 정답은 8개뿐이고, 오판의 72%는 정상, 27%는 DoS다. 그래서 두 class의 F1이 낮고 9종 macro-F1은 이전 7종 값보다 낮다. 이 점수는 참고 수준이다. netFound 점수는 잔여 flow를 삭제한 데이터에서 EXP94의 학습 행과 PCA를 유지하고 잔여 flow로 뽑힌 행만 교체해 잰 값이다. 7종 기준 같은 test 행에서 삭제 전후 차이는 n=50 TabPFN −0.5 pp, XGB −0.1 pp로 사실상 없다.", [])

    # Side information in few-shot.
    base5, seq5, port5 = ac('netfound', 'tabpfn', 50), ac('netfound+seq', 'tabpfn', 50), ac('netfound+ports', 'tabpfn', 50)
    x5, xp5 = ac('netfound', 'xgb', 50), ac('netfound+ports', 'xgb', 50)
    def side_row(name, arm):
        return (name, f"{100 * ac(arm, 'xgb', 20)['macro9']:.1f}", f"{100 * ac(arm, 'xgb', 50)['macro9']:.1f}", f"{100 * ac(arm, 'tabpfn', 20)['macro9']:.1f}", f"{100 * ac(arm, 'tabpfn', 50)['macro9']:.1f}",
                (f"{100 * ac(arm, 'tabpfn', 50)['validation_gain']:+.1f}".replace('-', '−') if arm != 'netfound' else '–'), f"{100 * ac(arm, 'tabpfn', 50)['benign_alarm']:.2f}", f"{100 * ac(arm, 'xgb', 50)['class_f1']['portscan']:.1f}")
    rows14 = [side_row('netFound', 'netfound'), side_row('+ 원본 seq 표시', 'netfound+seq'), side_row('+ 최근 60초 포트 수', 'netfound+ports')]
    # Bold marks the best value per column, shading alone marks the row of interest. Column 6 is false alarms, where lower is better.
    best14 = set()
    for j in range(1, len(rows14[0])):
        vals = [(float(r[j].replace('−', '-')), i) for i, r in enumerate(rows14) if r[j] != '–']
        top = (min if j == 6 else max)(v for v, _ in vals)
        best14 |= {(i, j) for v, i in vals if v == top}
    a = [txt(.65, 1.08, 12.0, '전체 9종 macro-F1  ·  netFound 벡터 뒤에 열 추가  ·  seed 5개  ·  같은 학습 행  ·  잔여 flow 삭제 데이터', 16, False, GREY)]
    a += table(.65, 1.56, [2.83, 1.10, 1.10, 1.25, 1.25, 1.50, 1.50, 1.50], ['입력', 'XGB\nn=20', 'XGB\nn=50', 'TabPFN\nn=20', 'TabPFN\nn=50', 'val 차이 pp\nTabPFN n=50', '정상 오탐 %\nTabPFN n=50', 'Portscan F1\nXGB n=50'], rows14, size=13.5, row_h=.54, styles={2: ('D5E8F4', False, NAVY)}, bold_cells=best14)
    a += card(.65, 3.92, 5.90, 2.84, '최근 60초 포트 수', f"TabPFN n=50 {100 * port5['gain_vs_netfound']:+.1f} pp, seed 5개 중 {port5['seeds_improved']}개 상승\nvalidation에서도 {100 * port5['validation_gain']:+.1f} pp\n\n정상 오탐 {100 * base5['benign_alarm']:.2f} → {100 * port5['benign_alarm']:.2f}%\nXGB Portscan F1 {100 * x5['class_f1']['portscan']:.1f} → {100 * xp5['class_f1']['portscan']:.1f}", accent=TEAL, body_size=17)
    a += card(6.78, 3.92, 5.90, 2.84, '원본 seq 표시', f"TabPFN n=50 {100 * seq5['gain_vs_netfound']:+.1f} pp, seed 5개 중 {seq5['seeds_improved']}개 상승\nvalidation은 {100 * seq5['validation_gain']:+.1f} pp".replace('-', '−') + "\n\n변화가 seed 편차 안에 있어 효과 불분명", accent=AMBER, body_size=17)
    page('추가 정보를 붙인 few-shot', a, '최근 60초 포트 수는 validation과 test에서 소폭 개선, 원본 seq 표시는 효과가 불분명', '05 결과',
         f"앞에서 본 두 추가 정보를 netFound 벡터 뒤에 붙여 전체 9종 few-shot을 다시 쟀다. 같은 학습 행을 쓰는 짝 비교다. 최근 60초 포트 수는 TabPFN n=50에서 {100 * port5['gain_vs_netfound']:+.1f} pp, validation에서 {100 * port5['validation_gain']:+.1f} pp 올랐고 정상 오탐도 줄었다. XGB의 Portscan F1은 {100 * x5['class_f1']['portscan']:.1f}에서 {100 * xp5['class_f1']['portscan']:.1f}으로 올랐다. 이 정보가 겨냥한 스캔 구분이 실제로 좋아진 것이다. 원본 seq 표시는 변화가 seed 편차 안에 있고 validation의 부호도 섞여 있어 효과를 말하기 어렵다. 두 정보 모두 bf, bot 같은 소수 공격의 개선을 겨냥한 것은 아니다.", [])

    # 16. Wrap up.
    a = []
    a += card(.65, 1.28, 3.95, 2.72, '문제1  동일 feature·상이 라벨', '잔여 flow 삭제로 4.86 → 2.04%\n\n남은 충돌은 패킷이 적은 스캔\n60초 포트 수로 0.05%까지 감소', accent=RED, body_size=15.5)
    a += card(4.85, 1.28, 3.95, 2.72, '문제2  few-shot 성능', f"few-shot은 여전히 잘 동작\n9종 n=50 macro {d_t[50][0]:.1f}%, CSV {c_t[50][0]:.1f}%\n\n대량 공격은 소수 라벨로 포화\nbot·web·inf는 여전히 낮음", accent=BLUE, body_size=15.5)
    a += card(9.05, 1.28, 3.63, 2.72, 'PCAP 직접 가공의 장점', 'flow 경계와 라벨을 직접 교정\n\nCSV에 없는 패킷 순서·맥락 사용\n\n입력 구성을 실험 가능', accent=TEAL, body_size=15.5)
    a += [box(.65, 4.24, 12.03, 2.50, 'F6F9FC', 'D3DFE9'), txt(.90, 4.44, 11.5, '앞으로', 21, True, NAVY)]
    steps = [('데이터', 'PCAP으로 가공할 수 있는 데이터 추가 탐색'), ('모델', 'netFound feature로 SOTA와 제안 모델 실험'), ('feature', '최근 60초 포트 수를 입력에 포함해 모델 비교')]
    for i, (k, s_) in enumerate(steps):
        y = 4.96 + i * .54
        a += chip(.90, y, 1.25, k, NAVY, 'FFFFFF', 14, .44)
        a.append(txt(2.32, y + .07, 10.1, s_, 17.5, False, INK))
    page('정리와 다음 단계', a, 'PCAP을 직접 가공하면 데이터 문제를 고칠 수 있고, 다음은 데이터 확장과 SOTA·제안 모델 실험', '06 정리',
         '정리하면 동일 feature 상이 라벨 문제는 잔여 flow를 삭제해 줄었고, 남은 충돌은 패킷이 적은 스캔이며 최근 60초 포트 수를 붙이면 0.05%까지 줄고 few-shot도 소폭 개선됐다. few-shot은 여전히 잘 동작하고 netFound + TabPFN이 CSV보다 높지만, 대량 공격의 포화는 그대로이고 bot, web, inf는 낮다. PCAP을 직접 가공하면 flow 경계와 라벨을 고칠 수 있고 CSV에 없는 패킷 순서와 맥락을 쓸 수 있으며 입력 구성을 실험할 수 있다. 앞으로 데이터를 더 찾아보고 SOTA와 제안 모델로 실험하겠다.', [])


def main() -> None:
    QA.mkdir(parents=True, exist_ok=True); ASSETS.mkdir(parents=True, exist_ok=True)
    for old in list(QA.glob('page-*.png')) + list(ASSETS.glob('*.png')):
        old.unlink()
    r.QA, r.ASSETS, r.OUT, r.N_SLIDES = QA, ASSETS, OUT, 15
    r.SOURCES = {key: (value['title'], f"{value['venue']}, {value['year']}", value['url'], value['evidence']) for key, value in b04.SOURCE_DATA.items()}
    r.SCENES = []
    recorded: list[dict] = []
    b04.page = lambda title, items, summary, section, notes, sources: recorded.append(dict(title=title, items=items, summary=summary, section=section, notes=notes, sources=sources))
    b04.scenes()
    for extra in ('layer_conflicts', 'fewshot_three_views'):
        (ASSETS / f'{extra}.png').unlink(missing_ok=True)
    r.SCENES = []
    scenes(recorded[3])
    r.validate_scenes()
    r.pptx()
    from pptx import Presentation
    presentation = Presentation(OUT.with_suffix('.pptx'))
    presentation.core_properties.title = '11/06 CSV에서 PCAP으로'
    presentation.core_properties.subject = 'PCAP과 netFound로 만든 feature의 동일 feature·상이 라벨과 few-shot 재점검'
    presentation.save(OUT.with_suffix('.pptx'))
    checks = r.pdf()
    for index in range(1, len(checks) + 1):
        source = QA / f'page-{index}.png'
        target = QA / f'page-{index:02d}.png'
        if source.exists():
            source.replace(target)
    from PIL import Image, ImageDraw
    for old in QA.glob('contact-*.png'):
        old.unlink()
    paths = sorted(QA.glob('page-*.png'))
    canvas = Image.new('RGB', (1500, 8 * 445), '#DCE3EB')
    for index, image_path in enumerate(paths):
        im = Image.open(image_path)
        im.thumbnail((730, 411))
        x, y = (index % 2) * 750 + 10, (index // 2) * 445 + 8
        canvas.paste(im, (x, y))
        ImageDraw.Draw(canvas).text((x, y + 414), str(index + 1), fill='black')
    canvas.save(QA / 'contact-1.png')
    (QA / 'scenes.json').write_text(json.dumps(r.SCENES, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    manifest = {'date': '2026-11-06', 'slides': len(r.SCENES), 'fewshot_run': str(ALL_CLASS.relative_to(ROOT)), 'paired_run': str(PAIRED.relative_to(ROOT)),
                'evidence': ['docs/lablog/evidence/20261106/exp95_residual_option_counts.json', 'docs/lablog/evidence/20261106/exp95_drop_fewshot_summary.json', 'docs/lablog/evidence/20261103/original_cic2017_csv_audit.json', 'docs/lablog/evidence/20261030/original_cic2018_csv_audit.json', 'docs/lablog/evidence/20261104/exp94_pcap_summary.json'],
                'validation': 'shared editable PPTX/HTML-PDF scenes; geometry/font/text checks; visual contact sheet reviewed', 'pages': checks,
                'pptx_sha256': hashlib.sha256(OUT.with_suffix('.pptx').read_bytes()).hexdigest(), 'pdf_sha256': hashlib.sha256(OUT.with_suffix('.pdf').read_bytes()).hexdigest()}
    (QA / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pptx': str(OUT.with_suffix('.pptx')), 'pdf': str(OUT.with_suffix('.pdf')), 'slides': len(r.SCENES)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
