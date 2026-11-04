"""Build the Korean deck that follows 20261029_PCAP_연구방향: EXP94 CIC2017 PCAP results.

Scenes stay editable in the PPTX; the two charts are rendered from the saved
EXP94 evidence JSON (CSV reference, PCAP run, three-layer audit).
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

ROOT = next(path for path in Path(__file__).resolve().parents if (path / 'scripts/common/experiment_paths.py').exists())
sys.path.insert(0, str(ROOT / 'scripts/common'))
import build_20261023_references as r

HERE = ROOT / 'docs/slides'
QA = HERE / '_support/qa/20261104_exp94'
OUT = HERE / 'kor/20261104_PCAP_결과'
PCAP_SUMMARY = ROOT / 'docs/lablog/evidence/20261104/exp94_pcap_summary.json'
CSV_SUMMARY = ROOT / 'docs/lablog/evidence/20261103/exp94_csv_summary.json'
CSV_AUDIT = ROOT / 'docs/lablog/evidence/20261103/original_cic2017_csv_audit.json'
SOURCE_DATA = json.loads((ROOT / 'docs/lablog/evidence/20261029_pcap/sources.json').read_text(encoding='utf-8'))['sources']

r.QA = QA
r.ASSETS = HERE / '_support/assets/20261104_exp94'
r.OUT = OUT
r.N_SLIDES = 12
r.SCENES = []
r.SOURCES = {key: (value['title'], f"{value['venue']}, {value['year']}", value['url'], value['evidence']) for key, value in SOURCE_DATA.items()}
txt, box, arrow, block, page = r.txt, r.box, r.arrow, r.block, r.page
NAVY, BLUE, INK, GREY, PALE = r.NAVY, r.BLUE, r.INK, r.GREY, r.PALE
RED, TEAL, AMBER, GREEN = 'B44948', '227B71', '9F6B20', '3C8D4F'
FONT = '/home/user/.local/share/fonts/Pretendard/Pretendard-Regular.otf'
BOLD = '/home/user/.local/share/fonts/Pretendard/Pretendard-Bold.otf'


def asset_image(name: str, x: float, y: float, w: float, h: float) -> dict:
    from PIL import Image
    path = r.ASSETS / f'{name}.png'
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    ww, hh = iw * scale, ih * scale
    return dict(kind='image', x=x + (w - ww) / 2, y=y + (h - hh) / 2, w=ww, h=hh, path=str(path))


def card(x, y, w, h, title, body, *, accent=BLUE, body_size=17, align='center'):
    return [
        box(x, y, w, h, 'FFFFFF', 'D3DFE9'),
        box(x, y, w, .43, accent),
        txt(x + .12, y + .075, w - .24, title, 18, True, 'FFFFFF', 'center'),
        txt(x + .18, y + .60, w - .36, body, body_size, False, INK, align),
    ]


def chip(x, y, w, s, fill=PALE, color=INK, size=14, h=.40, bold=True):
    return [box(x, y, w, h, fill), txt(x + .06, y + (h - size * 1.18 / 72) / 2 - .01, w - .12, s, size, bold, color, 'center')]


def hbar(x, y, label_w, bar_w_max, label, value, vmax, color, value_text, *, h=.34, size=15):
    width = max(.06, bar_w_max * value / vmax)
    return [
        txt(x, y + .04, label_w, label, size, True, INK),
        box(x + label_w, y, width, h, color),
        txt(x + label_w + width + .08, y + .05, 1.1, value_text, size, False, INK),
    ]


# ----------------------------------------------------------------------------- assets
def load():
    return (json.loads(PCAP_SUMMARY.read_text(encoding='utf-8')), json.loads(CSV_SUMMARY.read_text(encoding='utf-8')),
            json.loads(CSV_AUDIT.read_text(encoding='utf-8')))


def make_layer_asset(pcap: dict, audit: dict) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font, bold = FontProperties(fname=FONT), FontProperties(fname=BOLD)
    r.ASSETS.mkdir(parents=True, exist_ok=True)
    layers = pcap['audit3']['layers']
    labelled = pcap['audit3']['labelled_rows']
    rows = [
        ('CSV 78개 통계 (공식 배포본)', audit['nfv3_family_audit']['rows_in_conflicting_groups'] / audit['totals']['rows'], audit['nfv3_family_audit']['rows_in_conflicting_groups'], 14.7, '#617080'),
        ('L1  flow가 가진 정보 전부', layers['L1_flow_signature']['full_labelled']['cross_label_rows'] / labelled, layers['L1_flow_signature']['full_labelled']['cross_label_rows'], 100 * layers['L1_flow_signature']['train_to_test_exact_duplicate_fraction'], '#00306C'),
        ('L3  view 1: 통계 45개', layers['L3_flowstats_vector']['full_labelled']['cross_label_rows'] / labelled, layers['L3_flowstats_vector']['full_labelled']['cross_label_rows'], 100 * layers['L3_flowstats_vector']['train_to_test_exact_duplicate_fraction'], '#3C8D4F'),
        ('L2  netFound 토큰 (view 2 입력)', layers['L2_tokenizer_input']['full_labelled']['cross_label_rows'] / labelled, layers['L2_tokenizer_input']['full_labelled']['cross_label_rows'], 100 * layers['L2_tokenizer_input']['train_to_test_exact_duplicate_fraction'], '#B44948'),
        ('L3  view 2: netFound PCA 64', layers['L3_netfound_pca_vector']['full_labelled']['cross_label_rows'] / labelled, layers['L3_netfound_pca_vector']['full_labelled']['cross_label_rows'], 100 * layers['L3_netfound_pca_vector']['train_to_test_exact_duplicate_fraction'], '#D98B8A'),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.3), dpi=200)
    y = list(range(len(rows)))
    ax.barh(y, [100 * v for _, v, _, _, _ in rows], color=[c for *_, c in rows], height=.58)
    ax.set_yticks(y, [name for name, *_ in rows], fontproperties=bold, fontsize=9.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 9.0)
    ax.set_xlabel('입력이 완전히 같은데 family가 다른 flow의 비율 (%)', fontproperties=font, fontsize=9.4)
    ax.grid(axis='x', color='#D9E1E8', lw=.7)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', labelsize=8.4)
    for yy, (name, frac, rows_n, dup, color) in zip(y, rows):
        ax.text(100 * frac + .08, yy, f'{100 * frac:.2f}%  ({rows_n:,}행)  ·  train과 같은 test {dup:.0f}%', va='center', ha='left', fontsize=8.6, fontproperties=font, color='#20262E')
    fig.subplots_adjust(left=.36, right=.985, top=.97, bottom=.17)
    fig.savefig(r.ASSETS / 'layer_conflicts.png', facecolor='white')
    plt.close(fig)
    return {name: (frac, rows_n, dup) for name, frac, rows_n, dup, _ in rows}


def make_fewshot_asset(pcap: dict) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font, bold = FontProperties(fname=FONT), FontProperties(fname=BOLD)
    views = [('csv', 'CSV 78개 통계'), ('flowstats-v1', 'PCAP 통계 45개 (view 1)'), ('netfound-v2-small', 'PCAP netFound PCA 64 (view 2)')]
    methods = [('xgb', 'XGB', '#B44948'), ('tabpfn', 'TabPFN', '#007BC6')]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.55), dpi=190, sharey=True)
    values: dict = {}
    for ax, (key, title) in zip(axes, views):
        macro = pcap['views'][key]['macro_f1']
        for method, label, color in methods:
            pts = sorted([m for m in macro if m['regime'] == 'with_benign' and m['method'] == method], key=lambda m: m['budget'])
            xs, ys = [p['budget'] for p in pts], [100 * p['mean'] for p in pts]
            ax.errorbar(xs, ys, yerr=[100 * p['sample_sd'] for p in pts], color=color, marker='o', ms=5, lw=2.3, capsize=2.5, label=label)
            ax.annotate(f'{ys[-1]:.1f}', (xs[-1], ys[-1]), xytext=(6, 4 if method == 'tabpfn' else -12), textcoords='offset points', color=color, fontsize=9.5, fontproperties=bold)
            values[(key, method)] = dict(zip(xs, ys))
        ax.set_xscale('log', base=2)
        ax.set_xticks([2, 3, 5, 10, 20, 50], ['2', '3', '5', '10', '20', '50'], fontproperties=font, fontsize=9)
        ax.set_ylim(15, 80)
        ax.set_title(title, fontproperties=bold, fontsize=13, pad=8)
        ax.set_xlabel('클래스당 학습 라벨 수 n', fontproperties=font, fontsize=10)
        ax.grid(axis='y', color='#D9E1E8', lw=.75)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='y', labelsize=9)
    axes[0].set_ylabel('macro-F1 (%) · 정상 포함 7 family', fontproperties=font, fontsize=10)
    axes[2].legend(frameon=False, loc='lower right', prop=font)
    fig.subplots_adjust(left=.055, right=.985, top=.86, bottom=.2, wspace=.12)
    fig.savefig(r.ASSETS / 'fewshot_three_views.png', facecolor='white')
    plt.close(fig)
    return values


# ----------------------------------------------------------------------------- scenes
def table(x, y, widths, header, rows, *, size=13.5, row_h=.40, header_fill=NAVY, align=None, bold_last=False):
    """Simple grid: header row + body rows; numeric columns right-aligned."""
    items = []
    def row(values, yy, fill, bold=False, color=INK):
        xx = x
        for j, (w, v) in enumerate(zip(widths, values)):
            items.append(box(xx, yy, w, row_h, fill, 'D3DFE9'))
            al = (align[j] if align else ('left' if j == 0 else 'right'))
            items.append(txt(xx + .08, yy + (row_h - size * 1.18 / 72) / 2, w - .16, v, size, bold, color, al))
            xx += w
    row(header, y, header_fill, True, 'FFFFFF')
    for i, values in enumerate(rows):
        last = bold_last and i == len(rows) - 1
        row(values, y + row_h * (i + 1) + (.06 if last else 0), ('E6F1F8' if last else (PALE if i % 2 == 0 else 'FFFFFF')), last, NAVY if last else INK)
    return items


def scenes() -> None:
    pcap, csvs, audit = load()
    layer = make_layer_asset(pcap, audit)
    few = make_fewshot_asset(pcap)
    status = pcap['corpus_manifest']['label_status_counts']
    total = sum(status.values())
    bg, conf, unres = status['background_benign'], status['confirmed_attack'], status['unresolved_attempted']

    # 1. What happened since 10/29.
    a = [txt(.65, 1.08, 12.0, '10/29 계획은 CIC2018 PCAP 재표현이었으나 host 1대 추출의 한계로 CIC2017 전체 망으로 전환', 17, False, GREY)]
    a += card(.65, 1.70, 3.45, 3.05, '10/29 계획', 'PCAP에서 인코더로 벡터 생성\n\n충돌·조기 포화를\n같은 조건에서 재점검', accent=BLUE, body_size=17)
    a += [arrow(4.27, 3.05, .40)]
    a += card(4.85, 1.70, 3.90, 3.05, 'EXP93  CIC2018 host 1대', '날짜당 피해 host 1대만 추출\n\n✕ 정상 모집단이 CSV와 다름\n✕ 일부 공격 family 누락\n✕ Bot은 피해 host가 아닌 파일', accent=RED, body_size=16, align='left')
    a += [arrow(8.93, 3.05, .40)]
    a += card(9.50, 1.70, 3.18, 3.05, 'EXP94  CIC2017 전체 망', '5일 전체 망 캡처 51 GB\n\n✓ 정상 모집단이 CSV와 동일\n✓ CSV와 같은 망에서 비교\n✓ 라벨을 세 상태로 재부여', accent=TEAL, body_size=16, align='left')
    steps = [('10/29', '연구 방향 발표'), ('11/02', 'EXP93 host 1대'), ('11/03', 'CSV 기준 실험'), ('11/04', 'PCAP 실행·비교')]
    for i, (d, s_) in enumerate(steps):
        x = .65 + i * 3.05
        a += chip(x, 5.30, 1.05, d, NAVY, 'FFFFFF', 15, .42)
        a.append(txt(x + 1.15, 5.36, 1.85, s_, 15, False, INK))
        if i < 3:
            a.append(arrow(x + 2.72, 5.40, .24, .22))
    a.append(txt(.65, 6.10, 12.0, 'CIC2018은 host별 캡처 약 440개/일, 10일 450 GB  ·  CIC2017은 스위치 캡처 하루 1개 파일, 5일 51 GB', 15, False, GREY))
    page('지난 발표 이후의 진행', a,
         'host 1대 추출은 비교 조건을 깨뜨려 전체 망 캡처인 CIC2017로 전환', '06 진행 경과',
         '10/29 발표의 계획은 CIC2018 PCAP에서 인코더로 새 벡터를 만들어 두 문제를 재점검하는 것이었다. CIC2018은 host별 캡처라 날짜당 피해 host 1대만 받아 실행했는데(EXP93), 정상 트래픽이 그 host 것으로 바뀌어 CSV와 모집단이 달라졌고, 일부 공격이 빠졌으며, Bot은 피해 host가 아닌 파일을 골라 라벨이 틀렸다. 전체 450 GB 대신 스위치에서 전체 망을 캡처한 CIC2017(5개 파일, 51 GB)로 전환해 EXP94를 수행했다.', ['cic18', 'cic17'])

    # 2. Why two views: isolate "fixed flow/label" from "new representation".
    a = [txt(.65, 1.08, 12.0, '가설  CSV의 고정 통계가 정보를 버려 두 문제가 생긴다면, PCAP에서 패킷열로 표현을 만들면 나아진다', 17, False, GREY)]
    cols = [
        (.65, 'CSV (공식 배포본)', GREY, 'flow 정의  CICFlowMeter 원본\n(첫 FIN 종료, RST 무시)\n\n라벨  시간창 + 끝점\n\n표현  통계 78개'),
        (4.85, 'view 1  통계 45개', GREEN, 'flow 정의  고침\n(RST·양쪽 FIN·새 SYN·120초)\n\n라벨  세 상태 (증거 조건)\n\n표현  통계 (CSV와 같은 종류)'),
        (9.05, 'view 2  netFound', RED, 'flow 정의  view 1과 동일\n\n\n라벨  view 1과 동일\n\n표현  패킷 헤더열에서 학습한 벡터'),
    ]
    for x, t, c, b in cols:
        a += card(x, 1.68, 3.63, 2.95, t, b, accent=c, body_size=15, align='left')
    a += [arrow(4.38, 3.05, .38), arrow(8.58, 3.05, .38)]
    a += [
        box(.65, 4.95, 5.85, 1.35, 'F6F9FC', 'D3DFE9'),
        txt(.85, 5.10, 5.5, 'CSV ↔ view 1', 17, True, GREEN),
        txt(.85, 5.50, 5.5, 'flow 정의와 라벨을 고친 효과만 남음\n표현은 같은 종류이므로 대조군', 14.5, False, INK),
        box(6.83, 4.95, 5.85, 1.35, 'F6F9FC', 'D3DFE9'),
        txt(7.03, 5.10, 5.5, 'view 1 ↔ view 2', 17, True, RED),
        txt(7.03, 5.50, 5.5, '표현을 바꾼 효과만 남음\n가설이 참이면 여기서 충돌이 줄고 tail이 올라야 함', 14.5, False, INK),
    ]
    page('두 view를 두는 이유', a,
         'view 1은 flow·라벨 수정의 대조군, view 2가 가설 검증 대상', '07 접근',
         'PCAP으로 가면 두 가지가 동시에 바뀐다. flow 정의와 라벨이 고쳐지고, 표현이 바뀐다. 둘의 효과를 가르기 위해 같은 flow·라벨 위에 통계 view(CSV와 같은 종류의 표현)와 netFound view(패킷 헤더열에서 학습한 표현)를 둔다. CSV와 view 1의 차이는 flow·라벨 수정의 효과, view 1과 view 2의 차이는 표현의 효과다.', [])

    # 3. Pipeline with the encoder as an explicit step.
    a = [txt(.65, 1.08, 12.0, '①~④는 두 view가 공유하고, ⑤에서 갈라진 뒤 ⑥에서 같은 평가를 받는다', 16, False, GREY)]
    blocks = [
        (.65, '① PCAP 5개', '56.4M 패킷\n시각 + 패킷 바이트'),
        (3.30, '② flow로 묶기', '같은 두 끝점의 패킷\nRST·FIN·SYN·120초 종료'),
        (5.95, '③ 라벨 붙이기', '시간창 + 끝점 + 증거\n세 상태로 부여'),
        (8.60, '④ 시간순 나누기', '앞 70% 학습, 뒤 15% 평가\n날짜·family마다'),
    ]
    for x, title, body in blocks:
        a += block(x, 1.60, 2.45, 1.30, title, body, body_size=13.5)
    for x in [3.12, 5.77, 8.42]:
        a.append(arrow(x, 2.15, .20))
    a.append(arrow(9.72, 2.98, .24, .22, down=True))
    a += block(.65, 3.40, 2.30, 1.12, '⑤a 통계 계산', '모든 패킷에서 45개', color=GREEN, body_size=13.5)
    a.append(arrow(3.02, 3.86, .20))
    a += block(3.30, 3.40, 2.00, 1.12, 'view 1', '45차원 벡터', color=GREEN, body_size=13.5)
    a += block(.65, 4.95, 2.30, 1.12, '⑤b 토큰화', '헤더 필드 → 정수\nburst 12 × 패킷 6', color=RED, body_size=13.5)
    a.append(arrow(3.02, 5.41, .20))
    a += block(3.30, 4.95, 2.90, 1.12, 'netFound 인코더 (고정)', '토큰열 → 512차원', color=RED, body_size=13.5)
    a.append(arrow(6.27, 5.41, .20))
    a += block(6.55, 4.95, 1.65, 1.12, 'PCA 64', '학습 데이터로 맞춤', color=RED, body_size=13.5)
    a.append(arrow(8.27, 5.41, .20))
    a += block(8.55, 4.95, 1.55, 1.12, 'view 2', '64차원 벡터', color=RED, body_size=13.5)
    a += block(10.45, 3.40, 2.23, 2.67, '⑥ 평가', '충돌 점검 (3층)\noracle 상한\n\nfew-shot\nXGB · TabPFN\nn = 2 ~ 50', body_size=14)
    a.append(arrow(5.37, 3.86, .20))
    a.append(txt(5.65, 3.72, 4.6, '두 view가 같은 flow·라벨·분할·표본을 받음', 13.5, False, GREY))
    a.append(arrow(10.17, 5.41, .22))
    a.append(arrow(10.12, 3.86, .24))
    page('PCAP에서 벡터까지 여섯 단계', a,
         '공유 단계 ①~④ 위에서 ⑤a 통계와 ⑤b 토큰화·인코더만 다름', '08 파이프라인',
         'PCAP은 시각과 패킷 바이트의 나열이다. 같은 두 끝점 사이의 패킷을 flow로 묶고, 교정 규칙으로 라벨을 세 상태로 붙인 뒤 날짜와 family마다 시간순으로 나눈다. 그 위에서 두 경로로 갈라진다. ⑤a는 모든 패킷에서 통계 45개를 계산하고, ⑤b는 패킷 헤더를 정수 토큰으로 바꿔 고정된 netFound 인코더에 넣어 512차원을 얻은 뒤 PCA로 64차원으로 줄인다. 두 view는 같은 flow·라벨·분할·few-shot 표본으로 ⑥에서 평가된다.', ['netfound'])

    # 4. Step 1 -> 2 in detail: ten packets become one flow.
    a = [txt(.65, 1.08, 12.0, '월요일 11:57:01 UTC  사내 PC 192.168.10.9:1057 ↔ 외부 웹서버 69.31.33.224:80  ·  이 두 끝점의 패킷만 모으면 아래 10개', 15.5, False, GREY)]
    pk = [
        ('1', '0', '→', 'SYN', '52', '0'), ('2', '19,958', '←', 'SYN+ACK', '52', '0'), ('3', '20,008', '→', 'ACK', '40', '0'),
        ('4', '20,083', '→', 'ACK+PSH', '137', '97'), ('5', '40,027', '←', 'ACK', '40', '0'), ('6', '40,640', '←', 'ACK+PSH', '219', '179'),
        ('7', '40,643', '←', 'ACK+FIN', '40', '0'), ('8', '40,685', '→', 'ACK', '40', '0'), ('9', '40,689', '→', 'ACK+FIN', '40', '0'), ('10', '60,574', '←', 'ACK', '40', '0'),
    ]
    a += table(.65, 1.60, [.45, 1.15, .65, 1.35, .85, 1.0], ['#', '시각 (µs)', '방향', 'TCP flag', '길이', 'payload'], pk, size=12.5, row_h=.36, align=['right', 'right', 'center', 'left', 'right', 'right'])
    a.append(txt(.65, 5.65, 5.5, '→ PC가 보냄   ← 서버가 보냄   payload = 헤더 뒤의 실제 데이터(바이트)', 12.5, False, GREY))
    a.append(txt(.65, 5.95, 5.5, '4번이 요청(97 B), 6번이 응답(179 B), 7·9번 FIN으로 양쪽 종료', 13.5, True, NAVY))
    a += card(6.45, 1.60, 6.23, 1.55, '같은 flow로 묶는 조건', '프로토콜 + 두 (IP : 포트) 쌍이 같은 패킷\n어느 쪽이 보냈든 한 flow, 먼저 보낸 쪽이 →', accent=NAVY, body_size=14.5)
    a += card(6.45, 3.30, 6.23, 1.75, 'flow가 끝나는 때', 'RST  ·  양쪽 FIN 뒤 ACK  ·  같은 끝점의 새 SYN\n120초 무통신  ·  시작 후 120초\n이 flow는 10번 패킷에서 종료 (60.6 ms)', accent=NAVY, body_size=14.5)
    a += card(6.45, 5.20, 6.23, 1.45, '같은 규칙이 만드는 특수한 flow', '연결이 끝난 뒤 혼자 오는 RST 1개도 flow 하나\n→ 문제1에서 다시 등장 (패킷 1개짜리 RST flow)', accent=RED, body_size=14.5)
    page('① 패킷 10개가 ② flow 하나가 되기까지', a,
         f'패킷 56.4M개가 같은 규칙으로 flow {total:,}개가 되고, 그중 하나가 이 10개', '08 파이프라인',
         '월요일의 짧은 HTTP 접속 한 건이다. 같은 프로토콜과 두 (IP, 포트) 쌍을 가진 패킷을 시각순으로 모으면 10개이고, 4번이 요청, 6번이 응답, 7번과 9번의 FIN으로 양쪽이 종료를 알린 뒤 10번 ACK에서 flow가 닫힌다. RST, 양쪽 FIN, 새 SYN, 120초 무통신, 시작 후 120초가 종료 조건이다. 연결이 끝난 뒤 혼자 오는 RST도 같은 규칙으로 flow 하나가 되는데, 이것이 문제1의 충돌을 만드는 flow다.', [])

    # 5. One flow into two views.
    a = [txt(.65, 1.08, 12.0, '앞 장의 flow(패킷 10개, payload 97/179 B, 60.6 ms)를 두 view의 입력으로 바꾸면', 16, False, GREY)]
    a += [txt(.65, 1.62, 3.9, '⑤b  첫 패킷(SYN)의 토큰', 18, True, NAVY)]
    fields = [('IP 헤더 길이', '20'), ('TOS', '0'), ('IP 전체 길이', '52'), ('IP flag (DF)', '2'), ('TTL (Windows)', '128'), ('TCP flag (SYN)', '2'), ('window', '8192'), ('상대 seq·ack', '0 0 0 0'), ('urgent', '0')]
    for i, (k, v) in enumerate(fields):
        y = 2.05 + i * .40
        a.append(box(.65, y, 2.55, .36, PALE if i % 2 == 0 else 'FFFFFF', 'D3DFE9'))
        a.append(txt(.75, y + .07, 2.4, k, 13.5, False, INK))
        a.append(box(3.20, y, 1.15, .36, PALE if i % 2 == 0 else 'FFFFFF', 'D3DFE9'))
        a.append(txt(3.25, y + .07, 1.05, v, 13.5, True, NAVY, 'right'))
    a.append(txt(.65, 5.72, 3.9, '포트·IP·시각 없음  ·  패킷 10개 모두 같은 방식', 13, False, GREY))
    a += [txt(4.95, 1.62, 3.6, '⑤a  통계 45개 중 일부', 18, True, NAVY)]
    stats = [('지속 시간', '60,574 µs'), ('보낸·받은 패킷', '5 · 5'), ('보낸 길이 평균·최대', '61.8 · 137'), ('받은 길이 평균·최대', '78.2 · 219'), ('패킷 간격 평균', '6,730 µs'), ('SYN·FIN·PSH 수', '2 · 2 · 2'), ('초기 window', '8192 · 29200'), ('초당 패킷', '165'), ('목적 포트', '80')]
    for i, (k, v) in enumerate(stats):
        y = 2.05 + i * .40
        a.append(box(4.95, y, 2.25, .36, PALE if i % 2 == 0 else 'FFFFFF', 'D3DFE9'))
        a.append(txt(5.05, y + .07, 2.1, k, 13.5, False, INK))
        a.append(box(7.20, y, 1.45, .36, PALE if i % 2 == 0 else 'FFFFFF', 'D3DFE9'))
        a.append(txt(7.25, y + .07, 1.35, v, 13.5, True, GREEN, 'right'))
    a.append(txt(4.95, 5.72, 3.7, '사람이 정한 요약, 포트 포함', 13, False, GREY))
    a += [txt(9.25, 1.62, 3.5, '⑤b  burst → netFound', 18, True, NAVY)]
    bursts = [('→ 1', BLUE), ('← 1', TEAL), ('→ 2', BLUE), ('← 3', TEAL), ('→ 2', BLUE), ('← 1', TEAL)]
    for i, (s_, c) in enumerate(bursts):
        a += chip(9.25 + i * .58, 2.08, .52, s_, c, 'FFFFFF', 12, .40)
    a.append(txt(9.25, 2.58, 3.5, '6 burst (방향 · 패킷 수)\n같은 방향 10 ms 안의 패킷이 한 burst\n간격 0·0·20·20·21·21 ms', 13.5, False, INK))
    a += block(9.25, 3.45, 3.43, 1.05, 'netFound (고정)', '토큰열 → 512차원 벡터', body_size=14.5)
    a.append(arrow(10.85, 4.58, .24, .22, down=True))
    a += block(9.25, 4.92, 3.43, 1.05, 'PCA 64', '학습 데이터로만 맞춤 · 분산 98.1%', body_size=14.5)
    a.append(txt(9.25, 6.12, 3.5, '사전학습  라벨 없는 트래픽의\n가려진 헤더 토큰 맞히기', 13, False, GREY))
    page('② flow 하나를 ⑤ 두 view로', a,
         'view 1은 사람이 정한 요약(포트 포함), view 2는 헤더 토큰에서 배운 벡터(포트·IP·시각 없음)', '08 파이프라인',
         '앞 장의 flow를 두 view로 바꾼다. 토큰은 패킷마다 IP와 TCP 헤더 필드 값을 정수로 옮긴 것이고 IP·포트·시각은 넣지 않는다. 통계 45개는 모든 패킷에서 계산하며 목적 포트를 포함한다. netFound는 같은 방향 10 ms 안의 패킷을 burst로 묶어 처음 12 burst, burst당 6 패킷의 토큰을 받아 512차원 벡터를 내고, 학습 데이터로 맞춘 PCA로 64차원으로 줄인다.', ['netfound'])

    # 6. Labels: decision and what "evidence" means.
    a = [txt(.65, 1.08, 12.0, '교정 규칙 53개(CNS 2022)를 그대로 적용  ·  증거 조건은 공격마다 다르고 저자가 패킷을 보고 정한 것', 16, False, GREY)]
    qs = [('공격자 IP ↔ 피해자 IP?', '아니오 → 정상'), ('공격 시간창 안?', '창 밖 → 정상'), ('공격 행위의 증거?', '없음 → 보류')]
    for i, (q, no) in enumerate(qs):
        y = 1.65 + i * .98
        a += chip(.65, y, 3.35, q, NAVY, 'FFFFFF', 14, .48)
        a.append(txt(.75, y + .55, 2.6, no, 13, False, GREY))
        if i < 2:
            a.append(arrow(3.55, y + .52, .22, .22, down=True))
    a += chip(.65, 4.60, 3.35, '모두 예 → 공격', RED, 'FFFFFF', 14, .48)
    a.append(txt(.65, 5.25, 3.4, 'payload  헤더 뒤의 실제 데이터\nRST  상대가 연결을 거부·중단한 표시\n포트 제외  저자가 사람 조작으로\n확인한 연결의 포트 번호', 12.5, False, INK))
    ev = [
        ('FTP · SSH brute force', '공격자 payload ≥ 1 B (로그인 시도)', 'handshake만 하고 끝난 연결'),
        ('DoS GoldenEye', 'payload ≥ 1, RST 없음 또는 5초 이상', '요청 전에 끊긴 연결'),
        ('Slowloris · Slowhttptest', 'payload ≥ 1, 도구 설정 포트 제외', '서버 무응답 구간의 빈 연결'),
        ('Web Brute Force · XSS', 'payload ≥ 1 (BF는 패킷 ≥ 21)\n브라우저 포트 제외', '설정·종료 때의 정적 파일 요청'),
        ('Bot Ares', 'master와 payload 교환', 'master 종료 뒤\nSYN → RST 재접속'),
        ('Portscan', 'SYN 1개와 무응답 또는 RST\n(payload 없음이 정상)', '해당 없음'),
    ]
    a += table(4.35, 1.65, [2.0, 3.65, 2.68], ['공격', '증거 조건', '증거가 없는 예 → 보류'], ev, size=12.5, row_h=.52, align=['left', 'left', 'left'])
    a.append(txt(4.35, 5.40, 8.3, f'결과  flow {total:,}개 = 정상 {bg:,} · 공격 {conf:,} · 보류 {unres:,} (6.4%) · 경계 15', 14, True, NAVY))
    a.append(txt(4.35, 5.78, 8.3, '보류는 공격에도 정상에도 넣지 않고 학습·채점에서 제외, 수만 집계\n저자 권고: Attempted는 독립 class로 쓰지 말 것', 13, False, GREY))
    page('③ 라벨: 시간창·끝점·증거의 세 조건', a,
         '시간창과 끝점만 맞고 공격 행위의 증거가 없는 flow 154,743개는 보류', '09 라벨',
         '규칙은 시간창, 공격자와 피해자 끝점, 그리고 공격마다 다른 증거 조건으로 되어 있다. payload는 헤더 뒤의 실제 데이터이고, RST는 상대가 연결을 거부하거나 중단한 표시이며, 포트 제외는 저자들이 패킷을 직접 보고 사람 조작으로 확인한 연결의 포트 번호다. 셋을 모두 만족하면 공격, 창과 끝점만 맞고 증거가 없으면 보류, 어느 규칙에도 걸리지 않으면 정상이다. 보류는 저자들이 Attempted라 부른 7범주와 같고, 독립 class로 쓰지 말라는 권고에 따라 제외하고 수만 집계한다.', ['liu', 'attempted'])

    # 7. Final composition per family and subtype, with what is held out.
    a = [txt(.65, 1.08, 12.0, '분할 전 전체 기준  ·  CSV는 공식 라벨 행 수, PCAP은 세 상태로 나눈 flow 수  ·  행 순서는 PCAP 공격 수 내림차순', 15, False, GREY)]
    comp = [
        ('benign', '월요일 전일 + 공격일 배경', '2,273,097', '1,731,862', '–', '○'),
        ('portscan', '금 Portscan 161,202 · 목 NMAP 101,936', '158,930', '263,138', '0', '○'),
        ('dos', 'Hulk 158,571 · GoldenEye 7,567 · Slowloris 3,859 · Slowhttptest 1,741', '252,661', '171,738', '143,383', '○'),
        ('ddos', 'LOIC-HTTP', '128,027', '95,683', '1,203', '○'),
        ('bf', 'FTP 4,021 · SSH 2,961', '13,835', '6,982', '4,090', '○'),
        ('bot', 'Ares', '1,966', '736', '4,067', '○'),
        ('web', 'Brute Force 73 · XSS 18 · SQL Injection 13', '2,180', '104', '1,954', '○ (test 16)'),
        ('inf', 'Communication 32 · CoolDisk 4', '36', '36', '45', '✕ 학습 25'),
        ('heartbleed', 'Heartbleed', '11', '11', '0', '✕ 학습 7'),
        ('합계', '', '2,830,743', '2,270,290', '154,743', '7 family'),
    ]
    a += table(.65, 1.55, [1.2, 5.5, 1.3, 1.35, 1.3, 1.38], ['family', '세부 공격 (PCAP 공격 flow 수)', 'CSV 행', 'PCAP 공격', 'PCAP 보류', '채점'], comp, size=12, row_h=.40, align=['left', 'left', 'right', 'right', 'right', 'center'], bold_last=True)
    a.append(txt(.65, 6.10, 12.0, 'CSV와 PCAP의 수 차이는 flow 정의(RST·FIN 처리)와 보류 분리에서 온다  ·  목요일 NMAP 101,936개는 CSV에서 BENIGN, 교정 규칙에서 portscan', 13.5, False, GREY))
    page('최종 데이터 구성', a,
         '보류 154,743개(6.4%)를 빼고 7 family를 채점, inf·heartbleed는 표본 부족으로 제외', '09 라벨',
         '분할 전 전체 기준의 구성이다. PCAP 공격 수는 세부 공격별로도 적었다. dos의 보류 143,383은 거의 전부 payload 없는 Hulk 연결이고, web은 공격으로 확인된 flow가 104개뿐이다. 목요일 NMAP 스캔 101,936개는 원본 CSV에서 BENIGN이었으나 교정 규칙은 portscan으로 본다. inf와 heartbleed는 학습 표본이 50개에 못 미쳐 채점에서 뺀다.', ['liu'])

    # 8. Problem 1: layers defined, why three, and the verified example.
    a = [txt(.65, 1.08, 12.0, '같은 flow를 세 단계의 입력으로 보는 이유  ·  어느 단계에서 "입력은 같은데 family가 다른" 쌍이 생기는지 보면 라벨 탓인지 표현 탓인지 갈린다', 15, False, GREY)]
    a += card(.65, 1.60, 3.9, 1.70, 'L1  flow가 가진 정보 전부', '통계 45개 + 자르기 전 토큰 전부\n여기서 같으면 가를 정보가 없음\n→ 라벨 쪽의 하한', accent=NAVY, body_size=13.5)
    a += card(4.72, 1.60, 3.9, 1.70, 'L2  netFound 토큰화 뒤', '12 burst × 6 패킷으로 자르고\n간격을 ms로 반올림한 뒤\n→ tokenizer가 버린 정보', accent=RED, body_size=13.5)
    a += card(8.79, 1.60, 3.89, 1.70, 'L3  분류기가 받는 벡터', 'view 1 통계 45차원\nview 2 PCA 64차원\n(CSV 78개 통계도 하나의 L3)', accent=GREEN, body_size=13.5)
    a.append(txt(.65, 3.60, 12.0, '예  둘 다 패킷 1개짜리 RST+ACK, 길이 40 B, payload 0, TTL 128', 15, True, NAVY))
    ex = [
        ('정상  192.168.10.3:88 → 10.9:1031', '목적 포트 1031', '[20,0,40,2,128,20,0,0,0,0,0,0]', '벡터 다름 (포트)'),
        ('스캔 응답  10.50:1720 → 10.8:51700', '목적 포트 51700', '[20,0,40,2,128,20,0,0,0,0,0,0]', '벡터 다름 (포트)'),
    ]
    a += table(.65, 3.97, [3.6, 2.35, 3.5, 2.58], ['flow', 'L1 · 통계', 'L2 · 토큰', 'L3 · view 1'], ex, size=12.5, row_h=.42, align=['left', 'left', 'left', 'left'])
    a.append(txt(.65, 5.35, 12.0, 'L1·통계에서는 포트가 갈라 주지만, 포트가 없는 L2 토큰에서는 두 flow가 완전히 같은 입력 → 정상 11,817 + 스캔 4,363이 한 묶음', 14, True, RED))
    a.append(txt(.65, 5.78, 12.0, '라벨 쪽 하한 0.04%  ·  통계 45 (view 1) 0.28% = CSV 0.25% 수준  ·  netFound 토큰 (view 2 입력) 4.86%  ·  상한(oracle)은 모두 99.8% 이상', 14, False, INK))
    page('문제1  입력이 같아지는 단계', a,
         '충돌은 라벨이 아니라 통계 요약과 토큰 절삭에서 생기며, 포트 없는 netFound 토큰에서 가장 큼', '10 문제1',
         'L1은 flow가 가진 정보 전부이고, 여기서 같으면 파이프라인에 가를 정보가 없으므로 라벨 쪽 하한이다. L2는 netFound tokenizer 출력, L3는 분류기가 받는 벡터다. 같은 flow를 세 단계로 보는 이유는 충돌이 어느 단계에서 생기는지로 라벨 탓과 표현 탓을 가르기 위해서다. 예의 두 flow는 모두 패킷 1개짜리 RST+ACK인데, 통계에는 목적 포트가 있어 다른 벡터가 되지만 포트가 없는 토큰에서는 완전히 같아진다. 그래서 L2 충돌이 4.86%로 가장 크다.', [])

    # 9. Problem 1 numbers: chart.
    a = [txt(.65, 1.08, 12.0, '라벨 있는 flow 2,270,243개 기준  ·  막대는 "입력이 완전히 같은데 family가 다른" flow의 비율', 16, False, GREY)]
    a.append(asset_image('layer_conflicts', .55, 1.50, 7.55, 3.35))
    a += card(8.35, 1.55, 4.33, 3.30, '충돌의 정체  패킷 1개짜리 RST', '닫힌 포트에 보낸 SYN의 RST 응답과\n끝난 연결의 늦은 RST가 같은 모양\n\n토큰  [20, 0, 40, 2, TTL, RST, 0, …]\nLinux  정상 49,204 + 스캔 59\nWindows  정상 11,817 + 스캔 4,363\n\n포트가 없는 토큰은 TTL로만 구분', accent=RED, body_size=14, align='left')
    a += [
        txt(.65, 5.12, 6.0, 'L1  라벨 자체가 못 가르는 양  0.04%', 16, True, NAVY),
        txt(.65, 5.50, 6.0, 'L3 통계 45  CSV와 같은 수준  0.28%', 16, True, GREEN),
        txt(.65, 5.88, 6.0, 'L2 netFound 토큰  절삭과 포트 부재로 4.86%', 16, True, RED),
        txt(7.0, 5.12, 5.7, '어떤 모델도 넘지 못하는 상한(oracle)', 16, True, NAVY),
        txt(7.0, 5.50, 5.7, 'CSV 99.997%  ·  통계 100%  ·  netFound 토큰 99.79%', 15, False, INK),
        txt(7.0, 5.88, 5.7, '충돌이 점수를 깎는 양은 모든 표현에서 작음', 15, False, INK),
    ]
    page('문제1  층별 충돌 비율', a,
         'PCAP으로 가도 충돌은 사라지지 않고 netFound 토큰에서는 CSV보다 커지지만, 상한은 모두 99.8% 이상', '10 문제1',
         '라벨 쪽 하한은 0.04%, 통계 요약은 CSV와 같은 0.28%, netFound 토큰은 4.86%다. 충돌 group의 flow는 거의 전부 패킷 1개짜리 RST flow이며, 포트 스캔의 RST 응답과 정상 트래픽의 늦은 RST는 헤더가 같고 토큰에서는 TTL로만 갈린다. 그래도 세 표현 모두 oracle 상한이 99.8% 이상이라 충돌이 점수를 깎는 양은 작다.', [])

    # 10. Problem 2.
    a = [txt(.65, 1.08, 12.0, '정상 포함 7 family · class당 라벨 n개 · seed 5개 · 전체 평가 집합 · 두 모델에 같은 행', 16, False, GREY)]
    a.append(asset_image('fewshot_three_views', .45, 1.45, 12.45, 3.45))
    n50 = {k: few[(k, 'tabpfn')][50] for k in ('csv', 'flowstats-v1', 'netfound-v2-small')}
    n50x = {k: few[(k, 'xgb')][50] for k in ('csv', 'flowstats-v1', 'netfound-v2-small')}
    a += card(.65, 5.05, 3.95, 1.55, 'head 포화는 세 열 동일', '공격만 n=20  portscan 98~99.6%\nddos 89~99% · dos 85~98%\nn=50까지 1~2 pp 변화', accent=BLUE, body_size=14.5)
    a += card(4.85, 5.05, 3.95, 1.55, 'n=50 macro-F1  XGB / TabPFN', f'CSV  {n50x["csv"]:.1f} / {n50["csv"]:.1f}\n통계 45  {n50x["flowstats-v1"]:.1f} / {n50["flowstats-v1"]:.1f}\nnetFound  {n50x["netfound-v2-small"]:.1f} / {n50["netfound-v2-small"]:.1f}', accent=GREEN, body_size=14.5)
    a += card(9.05, 5.05, 3.63, 1.55, 'netFound 점수의 조건', 'test의 56%가 train과\n같은 토큰열\n외워서 맞힐 수 있는 구간', accent=RED, body_size=14.5)
    page('문제2  라벨 n개로 얼마나 오르나', a,
         'head의 조기 포화는 표현과 무관하고, netFound+TabPFN의 고점은 토큰 중복 56% 조건 위의 수치', '11 문제2',
         '세 열 모두 같은 few-shot 규약이다. 공격만 과제에서 portscan·ddos·dos는 n=20에서 85~99%에 이르고 n=50으로 가도 1~2 pp만 변한다. 정상 포함 macro-F1은 n=50에서 netFound+TabPFN이 가장 높고 netFound+XGB가 가장 낮아 표현보다 표현과 분류기의 조합이 가른다. netFound view는 test의 56%가 train과 같은 토큰열이므로 그 점수의 일부는 암기로 설명될 수 있다.', [])

    # 11. Verdict.
    a = []
    a += card(.65, 1.45, 3.95, 2.55, '문제1  동일 입력·상이 라벨', 'PCAP으로 가도 사라지지 않음\n통계 0.28% = CSV 수준\nnetFound 토큰 4.86%\n\n상한은 모두 99.8% 이상', accent=RED, body_size=15)
    a += card(4.85, 1.45, 3.95, 2.55, '문제2  head 조기 포화', '세 표현에서 동일\n대량·반복 공격의 성질\n\n표현이 바꾸는 것은\ntail(bf·bot)과 모델 차이의 부호', accent=BLUE, body_size=15)
    a += card(9.05, 1.45, 3.63, 2.55, '더 큰 변수  라벨 정책', '보류 154,743 flow를\n공격/정상 어디에 두느냐\n\n같은 corpus·인코딩에서\n변형 실험 진행 중', accent=AMBER, body_size=15)
    a += [
        box(.65, 4.35, 12.03, 1.05, 'F6F9FC', 'D3DFE9'),
        txt(.90, 4.60, 11.5, 'PCAP 전환의 산출물은 새 표현이 아니라 정리된 평가 규약', 19, True, NAVY, 'center'),
        txt(.90, 5.02, 11.5, '세 상태 라벨 · 시간순 분할 · 층별 충돌 점검과 상한 · 중복률을 본 연구의 평가 프로토콜로 채택', 14.5, False, GREY, 'center'),
    ]
    nexts = [('(a) 보류 → 공격 / 정상 두 정책 비교', '실행 중'), ('(b) PCA 128 · 512', '실행 중'), ('(c) 패킷 1개 RST flow 정리, 토큰에 포트 결합', '후보'), ('CIC2018 전체 450 GB 수집', '하지 않음')]
    for i, (s_, state) in enumerate(nexts):
        y = 5.70 + (i % 2) * .50
        x = .65 + (i // 2) * 6.1
        a += chip(x, y, .95, state, NAVY if state == '실행 중' else (TEAL if state == '후보' else GREY), 'FFFFFF', 12.5, .38)
        a.append(txt(x + 1.05, y + .07, 4.9, s_, 14.5, False, INK))
    page('판정과 다음 단계', a,
         '표현 교체로는 두 문제를 풀 수 없고, 라벨 정책과 평가 규약이 남은 변수', '12 판정',
         '문제1은 PCAP 직접 인코딩으로 사라지지 않았고, 문제2는 데이터의 성질이라 표현과 무관하다. 대신 세 상태 라벨, 시간순 분할, 층별 충돌 점검과 oracle 상한, 중복률이 정리됐고 이를 본 연구의 평가 프로토콜로 가져간다. 보류 flow의 라벨 정책 두 가지와 PCA 차원 변형은 같은 corpus와 인코딩 위에서 실행 중이며, 패킷 1개짜리 RST flow 정리와 토큰에 포트를 결합하는 변형이 다음 후보다. CIC2018 전체 수집은 하지 않는다.', [])

    # 12. Appendix table.
    a = [txt(.65, 1.08, 12.0, '정상 포함 n=20 family별 F1 (%) · 5 seed 평균 · 행 순서는 PCAP 평가 집합 수 내림차순', 16, False, GREY)]
    header = ['family', 'PCAP test', 'CSV XGB', 'CSV TabPFN', '통계 XGB', '통계 TabPFN', 'netFound XGB', 'netFound TabPFN']
    widths = [1.55, 1.25, 1.35, 1.45, 1.35, 1.45, 1.75, 1.88]
    def cf(view, method, name):
        return next((c for c in pcap['views'][view]['class_f1'] if c['budget'] == 20 and c['regime'] == 'with_benign' and c['method'] == method and c['name'] == name), None)
    families = ['benign', 'portscan', 'dos', 'ddos', 'bf', 'bot', 'web']
    rows = []
    for name in families:
        c0 = cf('flowstats-v1', 'xgb', name)
        rows.append([name, f"{c0['support']:,}"] + [f"{100 * cf(v, m, name)['mean']:.1f}" for v in ('csv', 'flowstats-v1', 'netfound-v2-small') for m in ('xgb', 'tabpfn')])
    rows.append(['macro', '340,543'] + [f"{100 * next(m['mean'] for m in pcap['views'][v]['macro_f1'] if m['regime'] == 'with_benign' and m['method'] == mm and m['budget'] == 20):.1f}" for v in ('csv', 'flowstats-v1', 'netfound-v2-small') for mm in ('xgb', 'tabpfn')])
    a += table(.65, 1.62, widths, header, rows, size=14, row_h=.44, bold_last=True)
    a.append(txt(.65, 6.05, 12.0, 'bot 0.0(CSV)은 평가 집합의 bot이 전부 master 종료 뒤 재접속 flow인 CSV 라벨 결함  ·  web은 PCAP 평가 집합 16행이라 참고치', 14, False, GREY))
    page('부록  n=20 family별 F1', a,
         '', '부록',
         '문제2 곡선을 family별로 푼 표다. CSV의 bot 0은 평가 집합의 bot 294행이 모두 master 종료 뒤 재접속 flow(저자 분류 Attempted)인데 Bot으로 라벨된 결함 때문이다. PCAP에서는 그 flow가 보류로 빠져 bot이 공격 사이에서는 구분되지만 정상과는 여전히 구분되지 않는다. web은 PCAP 평가 집합이 16행이라 수치를 해석하지 않는다.', [])


def main() -> None:
    QA.mkdir(parents=True, exist_ok=True)
    r.ASSETS.mkdir(parents=True, exist_ok=True)
    scenes()
    r.validate_scenes()
    r.pptx()
    from pptx import Presentation
    presentation = Presentation(OUT.with_suffix('.pptx'))
    presentation.core_properties.title = 'CIC2017 PCAP 전환 결과: CSV·통계·netFound 비교'
    presentation.core_properties.subject = 'EXP94 CSV 기준 실험과 PCAP 실행 결과'
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
    canvas = Image.new('RGB', (1500, 6 * 445), '#DCE3EB')
    for index, image_path in enumerate(paths):
        image = Image.open(image_path)
        image.thumbnail((730, 411))
        x, y = (index % 2) * 750 + 10, (index // 2) * 445 + 8
        canvas.paste(image, (x, y))
        ImageDraw.Draw(canvas).text((x, y + 414), str(index + 1), fill='black')
    canvas.save(QA / 'contact-1.png')
    (QA / 'scenes.json').write_text(json.dumps(r.SCENES, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    manifest = {
        'date': '2026-11-04', 'slides': len(r.SCENES), 'follows': 'kor/20261029_PCAP_연구방향.pptx',
        'pcap_summary': str(PCAP_SUMMARY.relative_to(ROOT)), 'csv_summary': str(CSV_SUMMARY.relative_to(ROOT)), 'csv_audit': str(CSV_AUDIT.relative_to(ROOT)),
        'validation': 'shared editable PPTX/HTML-PDF scenes; geometry/font/text checks; visual contact sheet reviewed',
        'pages': checks,
        'pptx_sha256': hashlib.sha256(OUT.with_suffix('.pptx').read_bytes()).hexdigest(),
        'pdf_sha256': hashlib.sha256(OUT.with_suffix('.pdf').read_bytes()).hexdigest(),
    }
    (QA / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pptx': str(OUT.with_suffix('.pptx')), 'pdf': str(OUT.with_suffix('.pdf')), 'slides': len(r.SCENES)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
