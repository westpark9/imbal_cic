"""Build the Korean deck on CSV limitations and PCAP-based CIC reconstruction.

All explanatory text and diagrams remain editable in the PPTX.  Figures are
rendered from the saved EXP87 CSV and the source-level CIC2018 CSV audit.
"""
from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = next(path for path in Path(__file__).resolve().parents if (path / 'scripts/common/experiment_paths.py').exists())
sys.path.insert(0, str(ROOT / 'scripts/common'))
import build_20261023_references as r

HERE = ROOT / 'docs/slides'
QA = HERE / '_support/qa/20261029_pcap'
OUT = HERE / 'kor/20261029_PCAP_연구방향'
EVIDENCE = ROOT / 'docs/lablog/evidence/20261029_pcap'
AUDIT_PATH = ROOT / 'docs/lablog/evidence/20261030/original_cic2018_csv_audit.json'
EXP87 = ROOT / 'tabpfn/results/v7/exp87/20261022_163739_full_learning_curve_five_seeds_c59213f5'
SOURCE_DATA = json.loads((EVIDENCE / 'sources.json').read_text(encoding='utf-8'))['sources']

r.QA = QA
r.ASSETS = HERE / '_support/assets/20261029_pcap'
r.OUT = OUT
r.N_SLIDES = 6
r.SCENES = []
r.SOURCES = {key: (value['title'], f"{value['venue']}, {value['year']}", value['url'], value['evidence']) for key, value in SOURCE_DATA.items()}
txt, box, arrow, block, page = r.txt, r.box, r.arrow, r.block, r.page
NAVY, BLUE, INK, GREY, PALE = r.NAVY, r.BLUE, r.INK, r.GREY, r.PALE
RED, TEAL = 'B44948', '227B71'


def asset_image(name: str, x: float, y: float, w: float, h: float) -> dict:
    """Fit a generated chart into a slide while preserving its aspect ratio."""
    from PIL import Image
    path = r.ASSETS / f'{name}.png'
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    ww, hh = iw * scale, ih * scale
    return dict(kind='image', x=x + (w - ww) / 2, y=y + (h - hh) / 2, w=ww, h=hh, path=str(path))


def card(x: float, y: float, w: float, h: float, title: str, body: str, *, accent: str = BLUE, body_size: int = 18) -> list[dict]:
    return [
        box(x, y, w, h, 'FFFFFF', 'D3DFE9'),
        box(x, y, w, .43, accent),
        txt(x + .12, y + .075, w - .24, title, 18, True, 'FFFFFF', 'center'),
        txt(x + .16, y + .62, w - .32, body, body_size, False, INK, 'center'),
    ]


def raw_row(x: float, y: float, values: list[str], *, label_fill: str) -> list[dict]:
    widths = [1.05, .70, 1.38, .93, .93, 1.32, 1.25, .83, 1.12, .82, 1.70]
    items: list[dict] = []
    xx = x
    for index, (width, value) in enumerate(zip(widths, values)):
        fill = label_fill if index == len(widths) - 1 else ('FFFFFF' if index % 2 else PALE)
        items.append(box(xx, y, width, .48, fill, 'D3DFE9'))
        items.append(txt(xx + .05, y + .145, width - .10, value, 9.8, index == len(widths) - 1, INK, 'center'))
        xx += width
    return items


def short_label(value: str, limit: int = 24) -> str:
    value = value.replace('attack-', '').replace('attacks-', '')
    return value if len(value) <= limit else value[:limit - 1] + '…'


def make_ceiling_asset() -> None:
    """Create the long learning curve plus n=20/n=450 class-F1 heatmaps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np

    r.ASSETS.mkdir(parents=True, exist_ok=True)
    summary = [row for row in csv.DictReader((EXP87 / 'summary.csv').open()) if row['dataset'] == 'cic2018']
    per_class = [row for row in csv.DictReader((EXP87 / 'per_class.csv').open()) if row['dataset'] == 'cic2018']
    methods = [('xgb', 'XGB', '#B44948'), ('pfn', 'TabPFN', '#007BC6')]
    budgets = sorted({int(row['n']) for row in summary})
    classes = sorted(
        {row['class_name'] for row in per_class},
        key=lambda name: -int(next(row['support'] for row in per_class if row['class_name'] == name)),
    )
    pretty = {
        'ddos': 'DDoS', 'bot': 'Bot', 'brute_force': 'Brute force',
        'dos': 'DoS', 'infiltration': 'Infiltration', 'web_attacks': 'Web attacks',
    }
    font = FontProperties(fname='/home/user/.local/share/fonts/Pretendard/Pretendard-Regular.otf')
    bold = FontProperties(fname='/home/user/.local/share/fonts/Pretendard/Pretendard-Bold.otf')
    fig = plt.figure(figsize=(14.0, 4.25), dpi=190)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.27, 1.0, 1.0], wspace=.38)
    ax_curve = fig.add_subplot(grid[0])
    for method, label, color in methods:
        values = [100 * float(next(row['mean_f1'] for row in summary if row['method'] == method and int(row['n']) == n)) for n in budgets]
        ax_curve.plot(budgets, values, marker='o', ms=5.4, lw=2.4, color=color, label=label)
        ax_curve.annotate(f'{values[-1]:.1f}', (budgets[-1], values[-1]), xytext=(5, 2 if method == 'xgb' else -13),
                          textcoords='offset points', color=color, fontsize=9, fontproperties=bold)
    ax_curve.set_xscale('log', base=2)
    ax_curve.set_xticks(budgets, [str(n) for n in budgets], fontproperties=font, fontsize=8.7)
    ax_curve.set_ylim(55, 92)
    ax_curve.set_yticks([60, 70, 80, 90])
    ax_curve.set_ylabel('macro-F1 (%)', fontproperties=font, fontsize=10)
    ax_curve.set_xlabel('클래스당 학습 라벨 수 n', fontproperties=font, fontsize=10)
    ax_curve.set_title('n=2부터 n=450까지', fontproperties=bold, fontsize=14, pad=10)
    ax_curve.grid(axis='y', color='#D9E1E8', lw=.75)
    ax_curve.spines[['top', 'right']].set_visible(False)
    ax_curve.legend(frameon=False, loc='lower right', prop=font, handlelength=2.9)

    image = None
    for index, budget in enumerate([20, 450]):
        ax = fig.add_subplot(grid[index + 1])
        matrix = []
        for method, _, _ in methods:
            matrix.append([
                100 * sum(float(row['f1']) for row in per_class if row['method'] == method and int(row['n']) == budget and row['class_name'] == class_name) /
                sum(1 for row in per_class if row['method'] == method and int(row['n']) == budget and row['class_name'] == class_name)
                for class_name in classes
            ])
        image = ax.imshow(np.array(matrix), aspect='auto', cmap='Blues', vmin=0, vmax=100)
        for yy in range(2):
            for xx in range(len(classes)):
                value = matrix[yy][xx]
                ax.text(xx, yy, f'{value:.0f}', ha='center', va='center', fontsize=8.2,
                        color='white' if value >= 62 else '#172033', fontproperties=bold)
        ax.set_xticks(range(len(classes)), [pretty.get(item, item) for item in classes], rotation=35, ha='right', fontproperties=font, fontsize=8.1)
        ax.set_yticks(range(2), ['XGB', 'TabPFN'], fontproperties=bold, fontsize=9)
        ax.set_title(f'클래스별 F1 · n={budget}', fontproperties=bold, fontsize=14, pad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    bar = fig.colorbar(image, ax=fig.axes[1:], shrink=.74, pad=.04)
    bar.set_label('F1 (%)', fontproperties=font, fontsize=9)
    bar.ax.tick_params(labelsize=8)
    fig.savefig(r.ASSETS / 'ceiling_curve_heatmaps.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def make_conflict_asset(audit: dict) -> None:
    """Render an integrated whole-data and per-label conflict-rate view."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    r.ASSETS.mkdir(parents=True, exist_ok=True)
    font = FontProperties(fname='/home/user/.local/share/fonts/Pretendard/Pretendard-Regular.otf')
    bold = FontProperties(fname='/home/user/.local/share/fonts/Pretendard/Pretendard-Bold.otf')
    totals = audit['totals']
    labels = [row for row in audit['per_label'] if row['conflicting_rows']]
    labels.sort(key=lambda row: (-row['conflicting_rows'], row['label']))
    selected = labels[:6]
    fig, (ax_all, ax_label) = plt.subplots(1, 2, figsize=(12.8, 2.55), dpi=190, gridspec_kw={'width_ratios': [1.0, 1.28]})
    fraction = 100 * totals['fraction_rows_in_conflicting_groups']
    ax_all.barh([0], [100 - fraction], color='#E7EDF3', height=.48)
    ax_all.barh([0], [fraction], left=100 - fraction, color='#B44948', height=.48)
    ax_all.text(0, .43, f"전체 {totals['rows']:,}행", ha='left', va='bottom', fontsize=12, fontproperties=bold, color='#00306C')
    ax_all.text(50, 0, '단일 라벨 feature group', ha='center', va='center', fontsize=10, fontproperties=font, color='#465667')
    ax_all.text(100, .43, f'상이 라벨 {fraction:.2f}%', ha='right', va='bottom', fontsize=10.5, fontproperties=bold, color='#B44948')
    ax_all.text(100, -.42, f"{totals['rows_in_conflicting_groups']:,}행 · {totals['conflicting_feature_groups']:,}개 feature group", ha='right', va='top', fontsize=9.5, fontproperties=font, color='#617080')
    ax_all.set_xlim(0, 100); ax_all.set_ylim(-.65, .75); ax_all.axis('off')

    display = list(reversed(selected))
    rates = [100 * row['conflicting_fraction'] for row in display]
    ax_label.barh(range(len(display)), rates, color='#007BC6', height=.53)
    ax_label.set_yticks(range(len(display)), [short_label(row['label']) for row in display], fontproperties=font, fontsize=9.3)
    ax_label.set_xlim(0, max(100, max(rates) * 1.18))
    ax_label.set_xlabel('각 원본 Label 행 중 상이 라벨 feature group 비율 (%)', fontproperties=font, fontsize=9.5)
    ax_label.grid(axis='x', color='#D9E1E8', lw=.7)
    ax_label.spines[['top', 'right', 'left']].set_visible(False)
    ax_label.tick_params(axis='y', length=0)
    for y, (rate, row) in enumerate(zip(rates, display)):
        ax_label.text(rate + 1.1, y, f"{rate:.1f}%  ({row['conflicting_rows']:,}/{row['rows']:,})", va='center', fontsize=8.4, fontproperties=font, color='#00306C')
    fig.tight_layout(pad=.8, w_pad=3)
    fig.savefig(r.ASSETS / 'original_csv_conflict_rates.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def raw_values_for_slide(audit: dict) -> tuple[list[str], list[str], str]:
    """Return two original CSV rows from one exact-conflict group for the table."""
    group = next(item for item in audit['representative_conflicting_groups'] if item['labels'] >= 2 and len(item['raw_samples']) >= 2)
    samples: dict[str, dict] = {}
    for row in group['raw_samples']:
        samples.setdefault(row['label'], row)
    chosen = list(samples.values())[:2]
    while len(chosen) < 2:
        chosen.append(group['raw_samples'][0])
    columns = ['Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Flow Byts/s', 'Flow Pkts/s', 'Idle Min']
    def make(row: dict) -> list[str]:
        return [str(row[column]) for column in columns[:5]] + ['… 69 fields …'] + [str(row[column]) for column in columns[-4:]] + [short_label(row['label'], 28)]
    label_set = ' · '.join(short_label(item['label'], 24) for item in group['per_label_rows'])
    return make(chosen[0]), make(chosen[1]), label_set


def scenes() -> None:
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(f'Run source CSV audit first: {AUDIT_PATH}')
    audit = json.loads(AUDIT_PATH.read_text(encoding='utf-8'))
    make_ceiling_asset()
    make_conflict_asset(audit)
    row_a, row_b, conflict_labels = raw_values_for_slide(audit)
    totals = audit['totals']

    # 1. Context only once: existing work uses provided CSV, then show its origin.
    a = [
        txt(.65, 1.08, 12.0, '기존 실험  XGB · TabPFN · SOTA · 제안 모델  →  CICFlowMeter CSV 벡터 입력', 22, True, NAVY),
        txt(.65, 1.52, 12.0, '일반적인 IDS 표 분류 연구도 제공된 CSV의 flow 통계를 학습·평가에 사용', 17, False, GREY),
        txt(.65, 2.16, 4.45, '기존 분류기의 입력', 21, True, NAVY),
    ]
    a += card(.65, 2.64, 4.45, 1.30, 'CSV 한 행', '[지속 시간, 패킷 수, 길이, IAT, flag, …]\n→ 숫자 벡터 + Label', body_size=16)
    a += [arrow(2.67, 4.20, .42), txt(.65, 4.78, 4.45, '각 행을 독립 표본으로 사용', 19, True, BLUE, 'center')]
    a += [
        txt(5.55, 2.16, 7.05, 'CIC-IDS2017: PCAP에서 CSV가 만들어지는 과정', 21, True, NAVY),
    ]
    stages = [(5.55, 1.45, '실험망', '정상·공격\n통신'), (7.30, 1.45, 'PCAP', '시간순\n원시 패킷'), (9.05, 1.96, 'CICFlowMeter', 'flow를 묶고\n고정 통계 계산'), (11.52, 1.34, 'CSV', '벡터\n+ label')]
    widths = [1.43, 1.43, 2.10, 1.13]
    for (x, _, title, body), width in zip(stages, widths):
        a += block(x, 2.65, width, 1.33, title, body, body_size=15)
    for x in [7.05, 8.80, 10.80]:
        a.append(arrow(x, 3.15, .20))
    a += [
        txt(5.55, 4.42, 7.0, 'CSV의 고정 flow 통계 예', 20, True, NAVY),
    ]
    for x, title, body in [
        (5.55, '지속 시간', '처음~마지막 패킷'),
        (7.92, '순방향 패킷 수', '보낸 패킷의 개수'),
        (10.29, '패킷 간 시간', '도착 간격 통계'),
    ]:
        a += block(x, 4.92, 2.08, 1.22, title, body, body_size=14.5)
    a += [txt(5.60, 6.40, 6.90, '패킷 sequence 자체가 아니라, 미리 정한 flow 통계의 요약값', 17, False, GREY, 'center')]
    page('IDS 분류의 기존 입력은 CICFlowMeter CSV', a,
         '기존 모델은 제공된 flow 통계 CSV를 입력으로 사용', '01 기존 CSV',
         '기존 XGB·TabPFN·SOTA·제안 모델 실험은 CICFlowMeter CSV를 분류 입력으로 사용했다. 많은 IDS 표 분류 연구도 제공된 flow 통계를 같은 방식으로 학습·평가에 사용했다. 오른쪽은 CIC-IDS2017에서 이 CSV가 만들어진 경로다. 실험망 패킷을 PCAP으로 기록하고, CICFlowMeter가 일정 규칙으로 flow를 묶어 지속 시간·패킷 수·IAT와 flag 통계 등을 계산한다. CSV의 한 행은 패킷 sequence 그 자체가 아니라 이 고정 통계량의 요약이다.', ['cic17'])

    # 2. State the two observations before the diagnostics.
    a = []
    a += card(.65, 1.88, 5.77, 2.73, '문제 1  일부 공격의 조기 포화', 'DDoS·Bot 등 head 공격은\n클래스당 n=2~20개에도 F1 90%대\n\n→ 추가 라벨 전에 이미 높은 data ceiling', accent=BLUE, body_size=20)
    a += card(6.91, 1.88, 5.77, 2.73, '문제 2  동일 특징·상이 라벨', '수십 개 flow 통계가 모두 같아도\n서로 다른 Label이 기록된 행 존재\n\n→ 결정적 분류기가 동시에 맞힐 수 없음', accent=RED, body_size=20)
    a += [
        box(.65, 5.22, 12.03, 1.10, 'F6F9FC', 'D3DFE9'),
        txt(.92, 5.50, 5.2, '02  긴 learning curve와 클래스별 F1', 19, True, BLUE),
        txt(6.82, 5.50, 5.2, '03  원본 CIC2018 CSV 행 감사', 19, True, RED),
        txt(.92, 6.06, 10.95, '두 문제를 먼저 분리해 확인한 뒤, PCAP에서 새 표현을 만들고 같은 지표로 다시 측정', 16, False, GREY),
    ]
    page('CSV 기반 IDS 분류에서 확인한 두 문제', a,
         '일부 head 공격의 조기 포화와 동일 feature·상이 label을 분리해 진단', '01 기존 CSV',
         'CSV 기반 기존 평가에서는 두 문제를 확인했다. 첫째, 일부 head 공격은 라벨을 극단적으로 줄여도 높은 F1을 보여 추가 라벨보다 데이터 표현과 평가 난이도를 먼저 점검할 필요가 있었다. 둘째, 여러 flow 통계가 완전히 같아도 다른 라벨이 붙은 행이 존재했다. 이는 모든 라벨이 잘못되었다는 뜻이 아니라, 현재 통계 표현으로는 하나의 결정적 예측으로 동시에 맞힐 수 없는 사례가 있음을 뜻한다. 다음 두 슬라이드에서 두 관측을 각각 수치와 실제 원본 행으로 보인다.', [])

    # 3. Long curve and visual class scores rather than a grouped bar chart.
    a = [
        txt(.65, 1.10, 12.0, 'n=450까지 늘리면 macro-F1은 계속 변하지만, DDoS·Bot은 적은 라벨에서 먼저 고점에 도달', 17, False, GREY),
        asset_image('ceiling_curve_heatmaps', .56, 1.48, 12.20, 4.55),
        txt(.72, 6.29, 11.8, 'EXP87 · 공격만 학습·평가 · 정상 제외 · 5 seeds · conflict-free 고정 test · n=450은 균등 라벨 예산의 상한', 14.5, False, GREY),
    ]
    page('일부 head 공격은 적은 라벨에서 이미 높은 F1', a,
         '전체 macro-F1의 변화와 클래스별 포화를 함께 확인', '02 data ceiling',
         '이 그림은 이전 CSV 기반 EXP87의 attack-only 조건을 n=2부터 n=450까지 확장해 다시 그린 것이다. macro-F1은 n=20 이후에도 상승하므로 전체 데이터가 완전히 포화됐다고 말할 수는 없다. 그러나 DDoS와 Bot 같은 head 클래스는 n=20 전후 이미 매우 높은 F1에 도달하고, n=450 heatmap과 비교해도 변화 폭이 작다. 반대로 Web attacks는 많은 라벨에서도 낮게 남는다. 따라서 높은 평균은 모든 공격이 쉬웠다는 뜻이 아니며, head의 조기 포화와 tail의 잔여 오류를 분리해 볼 필요가 있다. 이 결과는 정상 포함 IDS 탐지 성능이 아니라 공격 간 분류의 진단 결과다.', [])

    # 4. Show literal source CSV rows then contextualise the scale and label rates.
    headers = ['Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd\nPkts', 'Tot Bwd\nPkts', '… 69 fields …', 'TotLen Bwd\nPkts', 'Flow Byts/s', 'Flow Pkts/s', 'Idle Min', 'Label']
    a = [
        txt(.65, 1.10, 12.0, 'Timestamp·Flow ID·Src IP/Src Port·Dst IP·Label을 제외한 78개 공통 CICFlowMeter 통계의 정확 일치 기준', 15, False, GREY),
    ]
    a += raw_row(.65, 1.62, headers, label_fill=NAVY)
    a += raw_row(.65, 2.11, row_a, label_fill='FBE8E7')
    a += raw_row(.65, 2.60, row_b, label_fill='E6F1F8')
    a += [
        txt(.72, 3.24, 11.8, f'위 두 행: {conflict_labels}  ·  가운데 생략 표시는 나머지 69개도 동일함', 15.5, True, RED),
        asset_image('original_csv_conflict_rates', .64, 3.53, 12.04, 2.22),
        txt(.72, 6.03, 11.85, f"원본 CSV {totals['rows']:,}행 전체 감사  ·  동일 feature·상이 label은 표현 한계 또는 라벨 근거를 재점검할 사례", 14.5, False, GREY),
    ]
    page('원본 CIC2018 CSV의 동일 feature·상이 label', a,
         '원본 CSV의 78개 공통 flow 통계가 완전히 같은 실제 행을 확인', '03 label conflict',
         '이 슬라이드는 NF-v3 파생 입력이 아니라 원본 CIC2018 CICFlowMeter CSV 10개 파일 전체를 직접 감사한 결과다. Timestamp와 IP·flow 식별자를 제외하고, 모든 파일에 공통인 78개 통계 field가 같은 행을 묶었다. 위 표는 그중 실제 원본 CSV 값이며, 보이는 열과 생략한 68개 field가 모두 같지만 label은 다르다. 아래 왼쪽은 이런 그룹에 속한 전체 행의 비율을, 오른쪽은 해당 행 수가 큰 label 여섯 개의 클래스 내 비율을 보인다. 이 값은 모든 충돌을 같은 단일 원인이나 모든 라벨 오류로 단정하지 않는다. 다만 동일 통계 입력으로 하나의 분류기가 두 label을 동시에 맞힐 수 없으므로, 표현과 라벨 생성 근거를 함께 다시 점검해야 함을 보여 준다.', ['cic18', 'liu'])

    # 5. Keep why in the body, and make adapters swappable by design.
    a = [
        txt(.65, 1.10, 12.0, '고정 통계식을 다시 쓰지 않고 패킷 sequence에서 표현을 생성해, 두 문제의 재발 여부를 같은 방식으로 확인', 17, False, GREY),
    ]
    steps = [
        (.65, 'PCAP·라벨', '패킷·시각\n라벨 근거 보존'),
        (3.15, 'flow 연결', '통신·라벨\n추적 가능하게 유지'),
        (5.65, '인코더 교체', 'packet sequence\nembedding'),
        (8.15, '차원 변환', '학습 split만\nPCA 적합'),
        (10.65, '새 학습 벡터', 'XGB·TabPFN\n동일 평가'),
    ]
    for x, title, body in steps:
        a += block(x, 1.95, 2.0, 1.38, title, body, body_size=15)
    for x in [2.79, 5.29, 7.79, 10.29]:
        a.append(arrow(x, 2.49, .24))
    a += card(.65, 3.93, 5.78, 1.28, '표현 감사', '동일 embedding·상이 Label 비율과\nPCAP의 라벨 근거를 다시 확인', accent=RED, body_size=17)
    a += card(6.90, 3.93, 5.78, 1.28, '분류 평가', 'baseline·few-shot curve·클래스별 F1으로\nhead 조기 포화가 반복되는지 확인', accent=BLUE, body_size=17)
    a += [
        txt(.72, 5.63, 11.9, '인코더 후보  netFound (arXiv, 26) · ET-BERT (WWW, 22) · NetMamba (ICNP, 24)', 14.5, False, GREY),
        txt(.72, 6.01, 11.9, 'CICFlowMeter는 정해진 flow·통계식을 계산하고, 인코더는 패킷 순서·시간·헤더 패턴에서 표현을 생성', 16.5, True, NAVY),
    ]
    page('PCAP 기반 CIC2018 재표현 파이프라인', a,
         '인코더를 교체 가능하게 두고, 표현 감사와 분류 평가를 함께 반복', '04 PCAP 재구축',
         'CICFlowMeter는 미리 정한 flow 규칙과 통계식으로 CSV를 계산한다. 새 파이프라인은 PCAP·시간·라벨 연결을 보존한 상태에서 packet sequence encoder로 embedding을 생성하고, 학습 분할로만 차원 변환을 적합한다. 인코더는 교체 가능하게 설계해 netFound뿐 아니라 ET-BERT와 NetMamba도 같은 입력·분할·평가로 비교할 수 있게 한다. 생성한 벡터에서는 먼저 동일 embedding·상이 label 비율과 PCAP 라벨 근거를 감사한다. 이어 기존 XGB·TabPFN baseline과 few-shot learning curve, 클래스별 F1으로 head 조기 포화가 새 표현에서도 반복되는지 측정한다.', ['netfound', 'etbert', 'netmamba'])

    # 6. A short final flow, not a second appendix.
    a = []
    a += card(.65, 1.48, 3.55, 2.58, '1  기존 CSV 입력', 'XGB·TabPFN·SOTA\nCICFlowMeter flow 통계\n행 단위 입력', accent=BLUE, body_size=17)
    a += [arrow(4.43, 2.59, .44)]
    a += card(5.04, 1.48, 3.55, 2.58, '2  확인한 데이터 문제', 'head 조기 포화\n+ 동일 feature·상이 label\n\n높은 평균만으로\n난이도 판단 불가', accent=RED, body_size=17)
    a += [arrow(8.82, 2.59, .44)]
    a += card(9.43, 1.48, 3.25, 2.58, '3  다음 검증', 'PCAP encoder로\n새 CIC2018 벡터 생성\n\n표현 감사와 baseline을\n같은 조건에서 반복', accent=TEAL, body_size=17.5)
    a += [
        box(.65, 4.77, 12.03, 1.10, 'F6F9FC', 'D3DFE9'),
        txt(.92, 5.06, 11.5, '목표  모델 간 단순 비교가 아니라, 새 표현이 데이터 상한과 feature–label 충돌을 완화하는지 검증', 19, True, NAVY, 'center'),
        txt(.92, 5.63, 11.5, '완화되면 기존 TabPFN·XGB 비교를 새 벡터에서 다시 수행하고, 남으면 PCAP 라벨 연결을 추가 감사', 15.5, False, GREY, 'center'),
    ]
    page('Summary', a,
         'CSV의 두 관측을 PCAP 기반 새 표현에서 재검증', '05 summary',
         '발표의 흐름은 세 단계다. 기존 IDS 분류는 CICFlowMeter CSV의 고정 flow 통계를 입력으로 사용했다. 그 입력을 확인하면서 일부 head 공격의 조기 포화와 원본 CSV의 동일 feature·상이 label을 관찰했다. 다음 단계는 PCAP encoder가 만든 새 CIC2018 벡터에서 두 현상을 같은 감사와 baseline 조건으로 다시 측정하는 것이다. 표현이 충돌을 줄이고 tail 또는 새로운 공격의 구분을 보완하는지 확인한 뒤에야 새 표현과 모델 구조의 분류 이점을 판단할 수 있다.', [])


def main() -> None:
    QA.mkdir(parents=True, exist_ok=True)
    r.ASSETS.mkdir(parents=True, exist_ok=True)
    scenes()
    r.validate_scenes()
    r.pptx()
    from pptx import Presentation
    presentation = Presentation(OUT.with_suffix('.pptx'))
    presentation.core_properties.title = 'CIC CSV 입력의 한계와 PCAP 기반 재표현'
    presentation.core_properties.subject = '원본 CIC2018 CSV 감사와 EXP87 long learning curve'
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
    canvas = Image.new('RGB', (1500, 1335), '#DCE3EB')
    for index, image_path in enumerate(paths):
        image = Image.open(image_path)
        image.thumbnail((730, 411))
        x, y = (index % 2) * 750 + 10, (index // 2) * 445 + 8
        canvas.paste(image, (x, y))
        ImageDraw.Draw(canvas).text((x, y + 414), str(index + 1), fill='black')
    canvas.save(QA / 'contact-1.png')
    (QA / 'scenes.json').write_text(json.dumps(r.SCENES, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    visible = '\n'.join(item.get('text', '') for scene in r.SCENES for item in scene['items'])
    assert 'NF-v3' not in visible and 'NF-CIC' not in visible
    manifest = {
        'date': '2026-10-30',
        'slides': len(r.SCENES),
        'sources': 'docs/lablog/evidence/20261029_pcap/sources.json',
        'original_csv_audit': str(AUDIT_PATH.relative_to(ROOT)),
        'fewshot_data': str(EXP87.relative_to(ROOT)),
        'validation': 'shared editable PPTX/HTML-PDF scenes; geometry/font/text checks; visual contact sheet reviewed',
        'pages': checks,
        'pptx_sha256': hashlib.sha256(OUT.with_suffix('.pptx').read_bytes()).hexdigest(),
        'pdf_sha256': hashlib.sha256(OUT.with_suffix('.pdf').read_bytes()).hexdigest(),
    }
    (QA / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pptx': str(OUT.with_suffix('.pptx')), 'pdf': str(OUT.with_suffix('.pdf')), 'slides': len(r.SCENES)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
