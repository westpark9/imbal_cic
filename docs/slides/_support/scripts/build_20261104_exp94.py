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
BOUNDARY_PATH = ROOT / 'docs/lablog/evidence/20261105/exp94_flow_boundary_analysis.json'
BOUNDARY = json.loads(BOUNDARY_PATH.read_text(encoding='utf-8'))
WALK_PATH = ROOT / 'docs/lablog/evidence/20261105/exp94_packet_walkthrough.json'
WALK = json.loads(WALK_PATH.read_text(encoding='utf-8'))
SOURCE_DATA = json.loads((ROOT / 'docs/lablog/evidence/20261029_pcap/sources.json').read_text(encoding='utf-8'))['sources']

r.QA = QA
r.ASSETS = HERE / '_support/assets/20261104_exp94'
r.OUT = OUT
r.N_SLIDES = 20
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
        ('L1  flow에서 뽑은 정보 전부', layers['L1_flow_signature']['full_labelled']['cross_label_rows'] / labelled, layers['L1_flow_signature']['full_labelled']['cross_label_rows'], 100 * layers['L1_flow_signature']['train_to_test_exact_duplicate_fraction'], '#00306C'),
        ('L3  view 1: 통계 45개', layers['L3_flowstats_vector']['full_labelled']['cross_label_rows'] / labelled, layers['L3_flowstats_vector']['full_labelled']['cross_label_rows'], 100 * layers['L3_flowstats_vector']['train_to_test_exact_duplicate_fraction'], '#3C8D4F'),
        ('L2  netFound 토큰 (view 2 입력)', layers['L2_tokenizer_input']['full_labelled']['cross_label_rows'] / labelled, layers['L2_tokenizer_input']['full_labelled']['cross_label_rows'], 100 * layers['L2_tokenizer_input']['train_to_test_exact_duplicate_fraction'], '#B44948'),
        ('L3  view 2: netFound PCA 64', BOUNDARY['netfound_pca_rounded_3dp']['cross_label_rows'] / labelled, BOUNDARY['netfound_pca_rounded_3dp']['cross_label_rows'], BOUNDARY['netfound_pca_rounded_3dp']['train_to_test_dup_pct'], '#B44948'),
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
def table(x, y, widths, header, rows, *, size=13.5, row_h=.40, header_fill=NAVY, align=None, bold_last=False, styles=None, bold_cells=()):
    """Simple grid: header row + body rows; numeric columns right-aligned."""
    items = []
    def row(values, yy, fill, bold=False, color=INK, i=None):
        xx = x
        for j, (w, v) in enumerate(zip(widths, values)):
            items.append(box(xx, yy, w, row_h, fill, 'D3DFE9'))
            al = (align[j] if align else ('left' if j == 0 else 'right'))
            items.append(txt(xx + .08, yy + (row_h - (str(v).count('\n') + 1) * size * 1.18 / 72) / 2, w - .16, v, size, bold or (i, j) in bold_cells, color, al))
            xx += w
    row(header, y, header_fill, True, 'FFFFFF')
    for i, values in enumerate(rows):
        last = bold_last and i == len(rows) - 1
        fill, bold, color = (styles or {}).get(i, ('E6F1F8' if last else (PALE if i % 2 == 0 else 'FFFFFF'), last, NAVY if last else INK))
        row(values, y + row_h * (i + 1) + (.06 if last else 0), fill, bold, color, i)
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

    # 4. What a packet is: one real packet, byte by byte.
    P, rec = WALK['packet'], WALK['records']
    a = [txt(.65, 1.08, 12.0, f"패킷은 네트워크로 오가는 데이터 조각 하나  ·  아래는 월요일 사내 PC가 웹서버로 보낸 실제 패킷 {P['frame_bytes']}바이트", 16, False, GREY)]
    unit, x = 12.03 / P['frame_bytes'], .65
    for name, size, color in (('이더넷', P['ethernet_bytes'], GREY), ('IP 헤더', P['ip_header_bytes'], NAVY), ('TCP 헤더', P['tcp_header_bytes'], BLUE), ('payload', P['payload_bytes'], GREEN)):
        a += chip(x, 1.60, size * unit - .03, f'{name} {size} B', color, 'FFFFFF', 12.5, .46)
        x += size * unit
    a += card(.65, 2.28, 2.35, 3.05, '이더넷 14 B', '바로 옆 장비까지\n전달하기 위한 표\n\n보낸 장비 번호\n받는 장비 번호', accent=GREY, body_size=13.5)
    a += card(3.12, 2.28, 3.05, 3.05, 'IP 헤더 20 B', f"어느 컴퓨터로 갈지 적는 칸\n\n보낸 주소  {P['src_ip']}\n받는 주소  {P['dst_ip']}\n전체 길이  {P['ip_total_length']} B\nTTL  {P['ttl']}\n프로토콜  {P['protocol']}번 (TCP)", accent=NAVY, body_size=13.5, align='left')
    a += card(6.29, 2.28, 3.05, 3.05, 'TCP 헤더 20 B', f"어느 프로그램으로 갈지 적는 칸\n\n보낸 포트  {P['src_port']}\n받는 포트  {P['dst_port']}\nseq  {P['seq']}\nack  {P['ack']}\n신호  {P['tcp_flags']}  ·  window  {P['window']}", accent=BLUE, body_size=13.5, align='left')
    a += card(9.46, 2.28, 3.22, 3.05, 'payload 97 B', '실제 내용\n\n' + '\n'.join(P['payload_lines']), accent=GREEN, body_size=13.5, align='left')
    a.append(txt(.65, 5.47, 12.0, '택배로 비유하면  헤더는 상자 겉의 송장, payload는 상자 속 물건', 16, True, NAVY))
    a.append(txt(.65, 5.86, 12.0, 'TTL  패킷이 더 지날 수 있는 장비 수, 운영체제가 시작값을 정함   ·   window  지금 더 받을 수 있는 양', 13.5, False, INK))
    a.append(txt(.65, 6.16, 12.0, 'seq  보낸 데이터의 순번   ·   ack  어디까지 받았는지 알리는 번호   ·   TCP 신호는 다음 장들에서 설명', 13.5, False, INK))
    a.append(txt(.65, 6.50, 12.0, 'PCAP에 기록되는 것  도착 시각 + 이 바이트 전체', 14, True, NAVY))
    page('① 패킷의 생김새', a,
         '패킷은 겉 정보인 헤더와 속 내용인 payload로 구성되고, 본 연구는 payload 내용을 읽지 않음', '08 파이프라인',
         '네트워크로 오가는 데이터는 작은 조각으로 나뉘어 전송되고 그 조각 하나가 패킷이다. 예로 든 패킷은 월요일에 사내 PC가 웹서버에 보낸 실제 패킷이고 내용은 Windows가 인터넷 연결을 확인하는 요청이다. 앞쪽 54바이트가 헤더이고 뒤쪽 97바이트가 내용이다. 이더넷 헤더는 바로 옆 장비까지 전달하는 표, IP 헤더는 목적지 컴퓨터, TCP 헤더는 그 컴퓨터 안의 프로그램과 연결 상태를 적는 칸이다. 택배에 비유하면 헤더는 송장이고 payload는 상자 속 물건이다. 통계 view는 내용의 길이만 쓰고, netFound view는 헤더 값만 쓴다.', [])

    # 5. Reading the capture in time order and what is kept.
    fp = WALK['flows_by_protocol']
    a = [txt(.65, 1.08, 12.0, 'PCAP은 스위치를 지나간 패킷을 통째로 녹화한 파일  ·  기록 한 건 = 도착 시각 + 패킷 바이트  ·  5일치 파일 5개, 51 GB', 16, False, GREY)]
    how = [
        ('기록을 한 건씩 읽기', f"파일 앞에서 뒤로 {rec['total']:,}건을 차례로 읽음"),
        ('시각순으로 맞추기', f"녹화 순서가 뒤바뀐 패킷 {rec['out_of_order']:,}건\n최근 {rec['reorder_window_seconds']}초 분량을 모아 두고 시각이 빠른 패킷부터 처리"),
        ('필요한 패킷만 남기기', 'IPv4의 TCP·UDP·ICMP 패킷만 flow로 묶음\n나머지는 오른쪽 표처럼 제외'),
    ]
    for i, (t_, b_) in enumerate(how):
        y = 1.62 + i * .95
        a += chip(.65, y, .44, str(i + 1), NAVY, 'FFFFFF', 15, .44)
        a.append(txt(1.22, y + .02, 5.0, t_, 15.5, True, NAVY))
        a.append(txt(1.22, y + .35, 5.05, b_, 13, False, INK))
    a.append(txt(.65, 4.48, 5.7, '프로토콜 = 주고받는 방식의 약속, IP 헤더의 번호로 구분', 14.5, True, NAVY))
    used = [
        ('전체 기록', f"{rec['total']:,}", '5일치'),
        ('ARP', f"{rec['arp']:,}", 'IP에 맞는 장비 번호 묻기'),
        ('IPv6', f"{rec['ipv6']:,}", '이번 구현 범위 밖'),
        ('기타 장비 신호', f"{rec['llc_8023'] + rec['lldp']:,}", '스위치끼리의 신호'),
        ('조각·기타 프로토콜', f"{rec['ipv4_fragments'] + rec['other_transport']:,}", '포트를 읽을 수 없음'),
        ('닫힌 세션의 순수 ACK', f"{rec['closed_session_ack']:,}", '끝난 연결의 마지막 확인'),
        ('flow에 사용', f"{rec['in_flows']:,}", f"{100 * rec['in_flows'] / rec['total']:.1f}%"),
    ]
    a += table(6.55, 1.62, [2.40, 1.33, 2.40], ['구분', '패킷 수', '설명'], used, size=12.5, row_h=.355, align=['left', 'right', 'left'], bold_last=True)
    a += card(.65, 4.95, 3.85, 1.85, 'TCP  번호 6', f"연결을 맺고 받았는지 확인하며 보냄\n웹 · 메일 · 파일 전송\nflow {fp['tcp']:,}개", accent=NAVY, body_size=13.5)
    a += card(4.72, 4.95, 3.85, 1.85, 'UDP  번호 17', f"확인 없이 바로 보냄\n도메인 조회 · 시간 동기화\nflow {fp['udp']:,}개", accent=BLUE, body_size=13.5)
    a += card(8.79, 4.95, 3.89, 1.85, 'ICMP  번호 1', f"네트워크 상태를 알림\nping · 도달 불가 알림\nflow {fp['icmp']:,}개", accent=TEAL, body_size=13.5)
    page('① PCAP을 시각순으로 읽고 남기기', a,
         f"패킷 56.4M개를 시각순으로 읽어 {100 * rec['in_flows'] / rec['total']:.1f}%를 flow 구성에 사용", '08 파이프라인',
         'PCAP은 스위치를 지나간 패킷을 시각과 함께 그대로 녹화한 파일이다. 한 건씩 읽되 녹화 순서가 살짝 뒤바뀐 패킷이 있어 최근 5초 분량을 모아 두고 시각이 빠른 것부터 처리한다. 그다음 IPv4의 TCP, UDP, ICMP 패킷만 남긴다. ARP는 IP에 맞는 장비 번호를 묻는 신호이고, 스위치끼리의 신호와 함께 프로그램 간 대화가 아니라서 제외한다. IPv6는 이번 구현 범위 밖이다. 프로토콜은 주고받는 방식의 약속이다. TCP는 연결을 맺고 받았는지 확인하며 보내고, UDP는 확인 없이 보내며, ICMP는 네트워크 상태를 알린다.', [])

    # 6. Which packets belong to one conversation.
    a = [txt(.65, 1.08, 12.0, 'flow = 두 컴퓨터의 두 프로그램이 같은 약속으로 주고받은 패킷 묶음, 즉 대화 한 건', 16, False, GREY)]
    a += card(.65, 1.60, 3.85, 1.72, 'IP 주소', '네트워크 안의 컴퓨터 주소\n192.168.10.9 = 사내 PC 한 대', accent=NAVY, body_size=14)
    a += card(4.72, 1.60, 3.85, 1.72, '포트', '컴퓨터 안 프로그램의 문 번호\n80 웹 · 443 보안 웹 · 53 도메인 조회\n접속하는 쪽은 임시 번호 사용', accent=BLUE, body_size=13.5)
    a += card(8.79, 1.60, 3.89, 1.72, '프로토콜', '주고받는 방식의 약속\nTCP와 UDP는 포트 번호를 따로 사용\n같은 53번이어도 다른 통로', accent=TEAL, body_size=13.5)
    a.append(txt(.65, 3.46, 12.0, '다섯 값이 같으면 같은 대화  프로토콜 · IP A · 포트 A · IP B · 포트 B  (보낸 쪽과 받는 쪽이 바뀌어도 같음)', 15, True, NAVY))
    conv = [
        ('요청', 'TCP', '192.168.10.9 : 1057', '69.31.33.224 : 80', 'flow A'),
        ('응답', 'TCP', '69.31.33.224 : 80', '192.168.10.9 : 1057', 'flow A  방향만 반대'),
        ('인증 접속 1', 'TCP', '192.168.10.9 : 1055', '192.168.10.3 : 88', 'flow B'),
        ('인증 접속 2', 'TCP', '192.168.10.9 : 1056', '192.168.10.3 : 88', 'flow C  임시 포트가 다름'),
        ('도메인 조회', 'UDP', '192.168.10.9 : 59285', '192.168.10.3 : 53', 'flow D  프로토콜과 상대가 다름'),
    ]
    a += table(.65, 3.90, [1.80, 1.10, 2.75, 2.75, 3.63], ['패킷', '프로토콜', '보낸 쪽', '받는 쪽', '묶인 결과'], conv, size=13, row_h=.385, align=['left', 'left', 'left', 'left', 'left'])
    a.append(txt(.65, 6.36, 12.0, '묶는 이유  패킷 하나로는 행동을 알 수 없고, 대화 한 건 전체를 봐야 요청·응답·종료를 판단 가능', 14.5, True, NAVY))
    page('② 같은 대화의 패킷을 한 flow로 묶기', a,
         f"패킷 55.8M개가 같은 규칙으로 flow {total:,}개가 됨", '08 파이프라인',
         'IP 주소는 컴퓨터의 주소이고 포트는 그 컴퓨터 안 프로그램의 문 번호다. 서버는 정해진 번호를 쓰고 접속하는 쪽은 접속할 때마다 임시 번호를 받는다. 그래서 프로토콜과 양쪽 IP, 양쪽 포트 다섯 값이 같으면 같은 대화로 볼 수 있다. 프로토콜을 포함하는 이유는 TCP와 UDP가 포트 번호를 따로 쓰기 때문이다. 표의 예에서 응답은 보낸 쪽과 받는 쪽만 뒤집혔으므로 같은 flow이고, 같은 서버의 같은 서비스라도 임시 포트가 다르면 다른 접속이므로 다른 flow다. 패킷 하나로는 무슨 일이 일어나는지 알기 어렵기 때문에 대화 단위로 묶어서 본다.', [])

    # 7. TCP signals, RST, and which packets belong to the flow.
    a = [txt(.65, 1.08, 12.0, '→ PC가 보냄   ← 서버가 보냄   ·   소속 칸이 각 패킷이 들어가는 flow   ·   두 대화 모두 월요일 실제 패킷', 15.5, False, GREY)]
    head7 = ['#', '시각 (µs)', '방향', 'TCP flag', '길이', 'payload', '소속']
    w7 = [.40, 1.02, .52, 1.00, .50, .80, 1.66]
    al7 = ['right', 'right', 'center', 'left', 'right', 'right', 'left']
    pk = [
        ('1', '0', '→', 'SYN', '52', '0', 'flow A'), ('2', '19,958', '←', 'SYN+ACK', '52', '0', 'flow A'), ('3', '20,008', '→', 'ACK', '40', '0', 'flow A'),
        ('4', '20,083', '→', 'ACK+PSH', '137', '97', 'flow A'), ('5', '40,027', '←', 'ACK', '40', '0', 'flow A'), ('6', '40,640', '←', 'ACK+PSH', '219', '179', 'flow A'),
        ('7', '40,643', '←', 'ACK+FIN', '40', '0', 'flow A'), ('8', '40,685', '→', 'ACK', '40', '0', 'flow A'), ('9', '40,689', '→', 'ACK+FIN', '40', '0', 'flow A'), ('10', '60,574', '←', 'ACK', '40', '0', 'flow A 마감'),
    ]
    a.append(txt(.65, 1.49, 5.9, 'FIN으로 끝남  ·  192.168.10.9:1057 ↔ 웹서버 69.31.33.224:80', 13.5, True, NAVY))
    a += table(.65, 1.80, w7, head7, pk, size=12, row_h=.335, align=al7, styles={9: ('D5E8F4', True, NAVY)})
    ex = [
        ('1', '0', '→', 'SYN', '60', '0', 'flow B'), ('2', '42,246', '←', 'SYN+ACK', '60', '0', 'flow B'), ('3', '42,350', '→', 'ACK', '52', '0', 'flow B'),
        ('4', '5,079,690', '→', 'ACK+FIN', '52', '0', 'flow B'), ('5', '5,084,903', '←', 'ACK+FIN', '264', '212', 'flow B'), ('6', '5,084,965', '→', 'RST', '40', '0', 'flow B 마감'),
        ('7', '5,121,931', '←', 'ACK', '52', '0', '제외, 닫힌 뒤 ACK'), ('8', '5,121,962', '→', 'RST', '40', '0', '새 flow, 잔여 flow'),
    ]
    a.append(txt(6.78, 1.49, 5.9, 'RST로 끝남  ·  192.168.10.51:38488 ↔ 웹서버 209.15.36.31:80', 13.5, True, NAVY))
    a += table(6.78, 1.80, w7, head7, ex, size=12, row_h=.335, align=al7, styles={5: ('D5E8F4', True, NAVY), 6: ('EEF1F4', False, GREY), 7: ('F8E1E0', True, RED)})
    a.append(txt(6.78, 4.90, 5.9, 'PC가 4번 FIN으로 먼저 닫아 도착하는 5·7번마다 RST로 응답', 13, True, NAVY))
    a.append(txt(6.78, 5.19, 5.9, '첫 RST에서 마감, 둘째 RST는 패킷 1개짜리 새 flow  ·  잔여 flow 11.7%', 13, True, RED))
    a += card(.65, 5.60, 3.75, 1.30, 'TCP 신호', 'SYN 연결 요청  ·  ACK 받았다는 확인\nPSH 내용 바로 전달  ·  FIN 보낼 것이 끝남\nRST 연결을 버림, 확인 없이 즉시 종료', accent=BLUE, body_size=12.5)
    a += card(4.52, 5.60, 3.75, 1.30, 'RST가 오는 두 경우', '닫힌 문으로 연결 요청이 왔을 때 거절\n이미 닫은 연결로 패킷이 계속 올 때 중단', accent=RED, body_size=12.5)
    a += card(8.39, 5.60, 4.29, 1.30, 'flow가 끝나는 때', 'RST  ·  양쪽 FIN 뒤 ACK  ·  같은 끝점의 새 SYN\n120초 무통신  ·  시작 후 120초\nUDP·ICMP는 시간 조건만 사용', accent=NAVY, body_size=12.5)
    page('② TCP 신호와 flow의 끝', a,
         'FIN 종료는 마지막 ACK까지, RST 종료는 첫 RST까지가 한 flow이고 뒤따르는 RST는 새 flow', '08 파이프라인',
         'TCP는 헤더의 신호 칸으로 연결 상태를 알린다. 왼쪽 flow A는 1번부터 3번까지 연결을 맺고 4번이 요청, 6번이 응답이며, 7번과 9번의 FIN으로 양쪽이 종료를 알린 뒤 10번 ACK에서 닫힌다. 이 대화에는 RST가 없다. 오른쪽 flow B는 PC가 4번 FIN으로 먼저 연결을 닫은 경우다. 서버의 5번과 7번이 뒤늦게 도착하자 PC는 이미 닫은 연결이므로 패킷마다 RST로 답한다. 본 구현은 RST가 오면 flow를 바로 닫으므로 6번까지가 flow B다. 7번 순수 ACK는 닫힌 세션의 꼬리로 버리지만 8번 RST는 버리는 대상이 아니어서 패킷 1개짜리 새 flow가 된다. 이 잔여 flow가 전체의 11.7%이고 문제1 충돌의 주원인이다.', [])

    # 8. Why headers become number tokens.
    tk = WALK['tokens']
    a = [txt(.65, 1.08, 12.0, 'netFound는 문장을 단어 번호로 읽는 언어 모델과 같은 구조  ·  헤더의 칸 하나를 단어 하나처럼 숫자로 변환', 16, False, GREY)]
    head = ['패킷', 'IP 헤더\n길이', 'TOS', '전체\n길이', 'IP\nflag', 'TTL', 'TCP\nflag', 'window', 'seq\n앞', 'seq\n뒤', 'ack\n앞', 'ack\n뒤', 'urgent']
    rows = [['1번 SYN'] + [str(v) for v in tk['first_syn']], ['2번 SYN+ACK'] + [str(v) for v in tk['second_syn_ack']], ['4번 요청'] + [str(v) for v in tk['fourth_request']]]
    a += table(.65, 1.60, [1.71] + [.86] * 12, head, rows, size=12, row_h=.44)
    a.append(txt(.65, 3.42, 12.0, '패킷 1개 = 숫자 12개  ·  같은 자리는 항상 같은 뜻  ·  TCP flag 2 = SYN, 18 = SYN+ACK, 24 = ACK+PSH  ·  이 숫자열이 모델의 입력', 13.5, False, GREY))
    a += card(.65, 3.86, 2.92, 2.90, '왜 숫자인가', '모델은 숫자열만 읽음\n\n칸마다 뜻을 고정해야\n패킷끼리 비교 가능', accent=NAVY, body_size=14)
    a += card(3.69, 3.86, 2.92, 2.90, '왜 두 칸으로 나누나', '토큰 하나는 0~65,535\n\nseq·ack는 더 큰 수라\n앞·뒤 두 칸으로 분할', accent=BLUE, body_size=14)
    a += card(6.73, 3.86, 2.92, 2.90, '왜 상대값인가', f"원본 seq {P['first_packet_seq']}은\n무작위 출발값\n\n첫 패킷을 0으로 맞춰\n주고받은 양만 남김", accent=TEAL, body_size=14)
    a += card(9.77, 3.86, 2.91, 2.90, '왜 IP·포트를 빼나', '주소를 외우지 않고\n행동 모양을 배우도록\n저자가 제외\n\n사전학습도 같은 형식', accent=RED, body_size=14)
    page('⑤b 헤더를 숫자 토큰으로 바꾸는 이유', a,
         '헤더를 숫자열로 바꿔야 netFound가 읽을 수 있고, 이때 IP·포트·시각·payload는 입력에서 빠짐', '08 파이프라인',
         'netFound는 언어 모델처럼 정해진 번호의 나열을 읽는다. 그래서 헤더의 칸마다 값을 숫자 하나로 옮기고 그 숫자를 토큰이라 부른다. 칸의 순서가 고정되어 있어 같은 자리의 숫자끼리 비교할 수 있다. 토큰 하나는 0부터 65,535까지이므로 더 큰 seq와 ack는 앞뒤 두 칸으로 나눈다. seq는 연결마다 무작위 값에서 시작하므로 첫 패킷을 0으로 맞춘 상대값을 쓰고, 그러면 얼마나 주고받았는지만 남는다. IP와 포트는 저자들이 주소를 외우는 지름길을 막기 위해 뺐고, 공개된 가중치도 이 형식으로 사전학습되어 다른 칸을 넣을 수 없다.', ['netfound'])

    # 5. One flow into two views.
    a = [txt(.65, 1.08, 12.0, '앞에서 본 flow(패킷 10개, payload 97/179 B, 60.6 ms)를 두 view의 입력으로 바꾸면', 16, False, GREY)]
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
    a.append(txt(9.25, 2.58, 3.5, '6 burst (방향 · 패킷 수)\n같은 방향 10 ms 안의 패킷이 한 burst\n간격 0·0·20·20·20·20 ms (1 ms 미만 버림)', 13.5, False, INK))
    a += block(9.25, 3.45, 3.43, 1.05, 'netFound (고정)', '토큰열 → 512차원 벡터', body_size=14.5)
    a.append(arrow(10.85, 4.58, .24, .22, down=True))
    a += block(9.25, 4.92, 3.43, 1.05, 'PCA 64', '학습 데이터로만 맞춤 · 분산 98.1%', body_size=14.5)
    a.append(txt(9.25, 6.12, 3.5, '사전학습  라벨 없는 트래픽의\n가려진 헤더 토큰 맞히기', 13, False, GREY))
    page('② flow 하나를 ⑤ 두 view로', a,
         'view 1은 사람이 정한 요약(포트 포함), view 2는 헤더 토큰에서 배운 벡터(포트·IP·시각 없음)', '08 파이프라인',
         '앞에서 본 flow를 두 view로 바꾼다. 토큰은 패킷마다 IP와 TCP 헤더 필드 값을 정수로 옮긴 것이고 IP·포트·시각은 넣지 않는다. 통계 45개는 모든 패킷에서 계산하며 목적 포트를 포함한다. netFound는 같은 방향 10 ms 안의 패킷을 burst로 묶어 처음 12 burst, burst당 6 패킷의 토큰을 받아 512차원 벡터를 내고, 학습 데이터로 맞춘 PCA로 64차원으로 줄인다.', ['netfound'])

    # 10. Why short flows end up identical.
    ov = BOUNDARY['token_overlap_labelled']
    a = [txt(.65, 1.08, 12.0, '같다 = 숫자 토큰, 방향, 바이트, 패킷 수, ms 단위 간격이 전부 일치  ·  IP·포트·시각은 입력에 없어 달라도 같은 입력', 15.5, False, GREY)]
    name = {'3-5': '3~5', '6-20': '6~20', '21+': '21 이상'}
    a += table(.65, 1.62, [1.30, 1.35, 1.55, 1.50], ['패킷 수', 'flow 수', '서로 다른 입력', '같은 입력 %'],
               [(name.get(v['packets'], v['packets']), f"{v['rows']:,}", f"{v['distinct']:,}", f"{v['pct']:.1f}") for v in ov['by_packets']], size=13, row_h=.40)
    a += card(.65, 4.14, 5.70, 2.66, '짧은 flow의 칸이 거의 고정값인 이유', 'TTL  운영체제 기본값 64 · 128\n길이  빈 패킷 40, 연결 요청 52 · 60\nwindow  운영체제 기본값, RST는 0\n상대 seq·ack  0 또는 1\n시간 간격  1 ms 미만은 0', accent=NAVY, body_size=14, align='left')
    sh = WALK['identical_short_flows']
    tok = lambda v: '[' + ', '.join(str(x) for x in v) + ']'
    a += card(6.55, 1.60, 6.13, 1.14, f"패킷 1개  HTTPS 연결 끝의 RST  {sh[0]['flows']:,}개", f"{tok(sh[0]['tokens'][0])}", accent=RED, body_size=13)
    a += card(6.55, 2.82, 6.13, 1.36, f"패킷 2개  스캔 SYN과 거절 RST  {sh[1]['flows']:,}개", f"{tok(sh[1]['tokens'][0])}\n{tok(sh[1]['tokens'][1])}", accent=RED, body_size=13)
    a += card(6.55, 4.26, 6.13, 1.36, f"패킷 4개  도메인 조회  {sh[2]['flows']:,}개", f"질문 {tok(sh[2]['tokens'][0])} 2번  ·  답 {tok(sh[2]['tokens'][2])} 2번\nUDP는 칸이 6개, 이름 길이가 같으면 토큰도 같음", accent=TEAL, body_size=13)
    a.append(txt(6.55, 5.76, 6.13, '패킷이 많아지면 길이와 seq가 쌓여 조합이 갈라짐', 14, True, NAVY))
    a.append(txt(6.55, 6.08, 6.13, '겹침 자체는 문제가 아니고, 라벨까지 같으면 해롭지 않음', 14, False, INK))
    a.append(txt(6.55, 6.40, 6.13, '문제가 되는 것은 라벨이 갈리는 4.86%', 14, True, RED))
    page('② 패킷이 적은 flow가 똑같아지는 이유', a,
         '패킷이 5개 이하인 flow는 칸이 거의 고정값이라 같은 입력이 많고, IP·포트가 달라도 구별되지 않음', '08 파이프라인',
         '여기서 같다는 것은 netFound에 들어가는 입력 전체가 일치한다는 뜻이다. 숫자 토큰, burst의 방향, 바이트, 패킷 수, ms 단위 간격이 모두 같아야 한다. IP, 포트, 시각은 입력에 없으므로 서로 다른 컴퓨터의 다른 시각 flow도 같은 입력이 될 수 있다. 짧은 flow가 잘 겹치는 이유는 칸의 값이 거의 고정이기 때문이다. TTL과 window는 운영체제 기본값이고, 빈 패킷의 길이는 40바이트이며, 상대 seq와 ack는 0이나 1이고, 1 ms 미만의 간격은 0이 된다. 예를 들어 Linux PC가 HTTPS 연결 끝에 보낸 RST는 어느 PC가 어느 서버로 보냈든 숫자 12개가 같다. 도메인 조회는 내부 스위치 녹화에서 같은 패킷이 두 번 기록되어 패킷 4개로 보인다. 패킷이 많아지면 길이와 seq가 쌓여 조합이 갈라진다. 겹침 자체는 라벨까지 같으면 해롭지 않고, 문제는 라벨이 갈리는 4.86%다.', [])

    # 11. Labels: decision and what "evidence" means.
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
    rc = BOUNDARY['residual_counts']
    a.append(txt(4.35, 6.34, 8.3, f"보류의 {100 * rc['by_status']['unresolved_attempted'][0] / unres:.0f}%는 세션이 닫힌 뒤 남은 잔여 flow  ·  Hulk 무payload {rc['hulk_nopayload'][0]:,} / {rc['hulk_nopayload'][1]:,}", 14, True, RED))
    page('③ 라벨: 시간창·끝점·증거의 세 조건', a,
         '시간창과 끝점만 맞고 증거가 없는 flow 154,743개는 보류, 그중 89%는 세션이 닫힌 뒤의 잔여 flow', '09 라벨',
         '규칙은 시간창, 공격자와 피해자 끝점, 그리고 공격마다 다른 증거 조건으로 되어 있다. payload는 헤더 뒤의 실제 데이터이고, RST는 상대가 연결을 거부하거나 중단한 표시이며, 포트 제외는 저자들이 패킷을 직접 보고 사람 조작으로 확인한 연결의 포트 번호다. 셋을 모두 만족하면 공격, 창과 끝점만 맞고 증거가 없으면 보류, 어느 규칙에도 걸리지 않으면 정상이다. 보류는 저자들이 Attempted라 부른 7범주와 같고, 독립 class로 쓰지 말라는 권고에 따라 제외하고 수만 집계한다. 재분석 결과 보류의 89%는 공격 의도가 모호한 연결이 아니라 닫힌 세션 뒤에 남은 잔여 패킷이었고, Hulk 무payload 137,908개 중 133,626개가 여기에 해당한다.', ['liu', 'attempted'])

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
    after = BOUNDARY['residual_counts']['after_removal_status']
    a.append(txt(.65, 6.47, 12.0, f"잔여 flow {BOUNDARY['residual_counts']['all_flows']:,}개를 빼면  정상 {after['background_benign']:,}  ·  공격 {after['confirmed_attack']:,}  ·  보류 {after['unresolved']:,}  ·  경계 {after['boundary']}", 14, True, RED))
    page('최종 데이터 구성', a,
         '보류 154,743개(6.4%)를 빼고 7 family를 채점, inf·heartbleed는 표본 부족으로 제외', '09 라벨',
         '분할 전 전체 기준의 구성이다. PCAP 공격 수는 세부 공격별로도 적었다. dos의 보류 143,383은 대부분 Hulk 세션이 닫힌 뒤의 잔여 flow이고, web은 공격으로 확인된 flow가 104개뿐이다. 목요일 NMAP 스캔 101,936개는 원본 CSV에서 BENIGN이었으나 교정 규칙은 portscan으로 본다. inf와 heartbleed는 학습 표본이 50개에 못 미쳐 채점에서 뺀다.', ['liu'])

    # 8. Problem 1: layers defined, why three, and the verified example.
    a = [txt(.65, 1.08, 12.0, '같은 flow를 세 단계의 입력으로 보는 이유  ·  어느 단계에서 "입력은 같은데 family가 다른" 쌍이 생기는지 보면 정보가 사라지는 단계가 드러남', 15, False, GREY)]
    a += card(.65, 1.60, 3.9, 1.70, 'L1  flow에서 뽑은 정보 전부', '통계 45개 + 자르기 전 토큰 전부\nIP·시각·출발 포트·원본 seq는 없음\n→ 뽑은 정보로 가르지 못하는 양', accent=NAVY, body_size=13.5)
    a += card(4.72, 1.60, 3.9, 1.70, 'L2  netFound 토큰화 뒤', '12 burst × 6 패킷으로 자르고\n간격을 ms 단위로 버림한 뒤\n→ tokenizer가 버린 정보', accent=RED, body_size=13.5)
    a += card(8.79, 1.60, 3.89, 1.70, 'L3  분류기가 받는 벡터', 'view 1 통계 45차원\nview 2 PCA 64차원\n(CSV 78개 통계도 하나의 L3)', accent=GREEN, body_size=13.5)
    a.append(txt(.65, 3.60, 12.0, '예  둘 다 패킷 1개짜리 RST+ACK, 길이 40 B, payload 0, TTL 128', 15, True, NAVY))
    ex = [
        ('정상  192.168.10.3:88 → 10.9:1031', '목적 포트 1031', '[20,0,40,2,128,20,0,0,0,0,0,0]', '벡터 다름 (포트)'),
        ('스캔 응답  10.50:1720 → 10.8:51700', '목적 포트 51700', '[20,0,40,2,128,20,0,0,0,0,0,0]', '벡터 다름 (포트)'),
    ]
    a += table(.65, 3.97, [3.6, 2.35, 3.5, 2.58], ['flow', 'L1 · 통계', 'L2 · 토큰', 'L3 · view 1'], ex, size=12.5, row_h=.42, align=['left', 'left', 'left', 'left'])
    a.append(txt(.65, 5.35, 12.0, 'L1·통계에서는 포트가 갈라 주지만, 포트가 없는 L2 토큰에서는 두 flow가 완전히 같은 입력 → 정상 11,817 + 스캔 4,363이 한 묶음', 14, True, RED))
    a.append(txt(.65, 5.78, 12.0, 'L1 0.04%  ·  통계 45 (view 1) 0.28% = CSV 0.25% 수준  ·  netFound 토큰 (view 2 입력) 4.86%  ·  상한(oracle)은 모두 99.8% 이상', 14, False, INK))
    page('문제1  입력이 같아지는 단계', a,
         '충돌은 flow를 입력으로 바꾸는 단계에서 생기며, 포트 없는 netFound 토큰에서 가장 큼', '10 문제1',
         'L1은 파이프라인이 flow에서 뽑은 정보 전부다. 출발 포트와 원본 seq처럼 뽑지 않은 필드가 있으므로 L1 충돌이 라벨의 하한은 아니다. L2는 netFound tokenizer 출력, L3는 분류기가 받는 벡터다. 같은 flow를 세 단계로 보는 이유는 어느 단계에서 정보가 사라지는지 보기 위해서다. 예의 두 flow는 모두 패킷 1개짜리 RST+ACK인데, 통계에는 목적 포트가 있어 다른 벡터가 되지만 포트가 없는 토큰에서는 완전히 같아진다. 그래서 L2 충돌이 4.86%로 가장 크다.', [])

    # 9. Problem 1 numbers: chart.
    a = [txt(.65, 1.08, 12.0, '라벨 있는 flow 2,270,243개 기준  ·  막대는 "입력이 완전히 같은데 family가 다른" flow의 비율', 16, False, GREY)]
    a.append(asset_image('layer_conflicts', .55, 1.50, 7.55, 3.35))
    a += card(8.35, 1.55, 4.33, 3.30, '충돌의 정체  패킷 1개짜리 RST', '세션이 닫힌 뒤 남은 RST가\n정상과 스캔 양쪽에서 같은 모양\n\n토큰  [20, 0, 40, 2, TTL, RST, 0, …]\nLinux  정상 49,204 + 스캔 59\nWindows  정상 11,817 + 스캔 4,363\n\n포트가 없는 토큰은 TTL로만 구분', accent=RED, body_size=14, align='left')
    a += [
        txt(.65, 5.12, 6.0, 'L1  flow에서 뽑은 정보로 못 가르는 양  0.04%', 16, True, NAVY),
        txt(.65, 5.50, 6.0, 'L3 통계 45  CSV와 같은 수준  0.28%', 16, True, GREEN),
        txt(.65, 5.88, 6.0, 'netFound  토큰과 PCA 벡터 모두 4.86%', 16, True, RED),
        txt(7.0, 5.12, 5.7, '어떤 모델도 넘지 못하는 상한(oracle)', 16, True, NAVY),
        txt(7.0, 5.50, 5.7, 'CSV 99.997%  ·  통계 100%  ·  netFound 토큰 99.79%', 15, False, INK),
        txt(7.0, 5.88, 5.7, '충돌이 점수를 깎는 양은 모든 표현에서 작음', 15, False, INK),
    ]
    page('문제1  층별 충돌 비율', a,
         '현재 flow 규칙에서는 충돌이 남고 netFound에서 CSV보다 커지지만, 상한은 모두 99.8% 이상', '10 문제1',
         'L1은 0.04%, 통계 요약은 CSV와 같은 0.28%, netFound 토큰은 4.86%다. PCA 64 벡터도 소수점 3자리로 비교하면 토큰과 같은 4.86%로, 이전 판의 2.98%는 float32 정확 비교가 GPU 미세 오차로 같은 벡터를 나눈 과소집계였다. 충돌 group의 flow는 거의 전부 패킷 1~2개짜리 RST flow이며, 스캔과 정상의 잔여 RST는 헤더가 같고 토큰에서는 TTL로만 갈린다. 그래도 세 표현 모두 oracle 상한이 99.8% 이상이라 충돌이 점수를 깎는 양은 작다.', [])

    # 10. Problem 1 root cause: the flow boundary.
    b = BOUNDARY
    rc = b['residual_counts']
    ft = b['flow_totals']
    pol3 = b['conflict_rows_by_label_policy']
    a = [txt(.65, 1.08, 12.0, 'RST에서 flow를 바로 닫아 같은 연결의 뒤따르는 패킷이 새 flow가 됨  ·  원본 PCAP 헤더로 대조', 16, False, GREY)]
    a += chip(.65, 1.68, .95, 'SYN', NAVY, 'FFFFFF', 13, .42)
    a += chip(1.66, 1.68, 1.75, '요청 · 응답 패킷', PALE, INK, 13, .42)
    a += chip(3.47, 1.68, .85, 'RST', NAVY, 'FFFFFF', 13, .42)
    a.append(box(4.46, 1.60, .03, .58, RED))
    a += chip(4.63, 1.68, .85, 'RST', RED, 'FFFFFF', 13, .42)
    a.append(txt(.65, 2.18, 3.7, '세션의 flow, 첫 RST에서 종료', 13, False, INK))
    a.append(txt(4.40, 2.18, 1.95, '새 flow, 패킷 1개', 13, True, RED))
    a += card(.65, 2.62, 5.72, 1.32, '잔여 flow 판정 기준', '직전 세션이 RST·양쪽 FIN으로 닫힌 뒤 2초 안\nSYN 없음 · payload 없음 · 패킷 2개 이하', accent=RED, body_size=14)
    facts = [
        ('잔여 flow', f"{rc['all_flows']:,} / {ft['flows']:,}"),
        ('패킷 1개 flow 중 잔여', f"{ft['one_packet_residual']:,} / {ft['one_packet_flows']:,}"),
        ('보류 중 잔여', f"{rc['by_status']['unresolved_attempted'][0]:,} / {rc['by_status']['unresolved_attempted'][1]:,}"),
        ('직전 세션과 라벨 일치', f"{rc['labelled']:,} / {rc['labelled']:,}"),
        ('직전 세션과의 간격 중앙값', f"{1000 * rc['gap_seconds']['median']:.2f} ms"),
    ]
    a += table(.65, 4.12, [3.32, 2.40], ['항목', '수'], facts, size=13, row_h=.38)
    a.append(txt(.65, 6.47, 5.72, f"구성  같은 쪽 연속 RST {rc['composition']['same_sender_rst_after_rst']:,}  ·  상대의 FIN {rc['composition']['peer_fin_after_rst']:,}", 12.5, False, GREY))
    rows = [
        ('L1 flow에서 뽑은 정보', pol3['exclude']['L1']),
        ('L3 통계 45', pol3['exclude']['L3s']),
        ('L2 netFound 토큰', pol3['exclude']['L2']),
        ('보류를 공격으로, 통계 45', pol3['attack']['L3s']),
        ('보류를 정상으로, L1', pol3['benign']['L1']),
    ]
    a += table(6.65, 1.62, [3.25, 1.20, 1.58], ['충돌 행 수', '현재', '잔여 flow 제외'], [(n, f"{v['before']['conflict_rows']:,}", f"{v['after']['conflict_rows']:,}") for n, v in rows], size=13.5, row_h=.42)
    a.append(txt(6.65, 4.20, 6.03, f"남은 netFound 충돌은 거의 전부 스캔 SYN 탐침, 소수 라벨 {pol3['exclude']['L2']['after']['minority']}행", 13.5, True, NAVY))
    a += card(6.65, 4.62, 6.03, 2.05, 'netFound 원 전처리와의 차이', '원 전처리  같은 두 끝점을 새 SYN에서만 분리\nRST·FIN·시간 제한으로 닫지 않아 잔여 flow 없음\n본 구현  RST 즉시 종료, 잔여 패킷은 순수 ACK만 제거\nWTMC 2021이 지적한 종료 패킷 flow가 RST·FIN으로 남음', accent=NAVY, body_size=13.5, align='left')
    page('문제1  flow 경계가 만든 잔여 flow', a,
         '잔여 flow를 세지 않으면 feature 변경 없이 충돌이 대부분 사라지고 보류 정책 간 충돌 차이도 소멸', '10 문제1',
         '충돌의 주원인은 표현이 아니라 flow 경계였다. 현재 규칙은 RST가 오면 flow를 바로 닫는데, 연결을 끊을 때는 RST가 연달아 오거나 상대의 FIN이 뒤따르는 일이 흔하다. 그 패킷이 패킷 1~2개짜리 새 flow가 되고, 세션에서 떨어져 나온 헤더 조각이라 벡터가 같아진다. 패킷 1개 flow의 96%가 이 경우이고 라벨이 붙은 행은 전부 직전 세션과 라벨이 같다. 표의 제외 값은 해당 행을 빼고 다시 센 값이며 재학습 결과가 아니다. 실제 예로 월요일 192.168.10.12가 HTTPS 서버와의 연결을 FIN으로 닫은 뒤 서버 패킷이 두 개 더 도착하자 RST를 0.003 ms 간격으로 두 번 보냈고, 첫 RST가 flow를 닫아 둘째 RST가 패킷 1개짜리 flow가 됐다. 내부 구간에서는 스위치 녹화가 같은 RST+ACK를 두 번 기록해 같은 일이 생긴다. netFound 원 전처리는 새 SYN에서만 flow를 나누므로 이 문제가 없고, EXP92 구현이 WTMC 2021의 지적을 순수 ACK에 대해서만 반영한 것이 원인이다.', ['engelen', 'netfound'])

    # 11. Different inputs are not yet separable inputs.
    cand = {c['name']: c for c in b['candidate_fields_on_L2_conflict_rows']}
    aft = {c['name']: c for c in b['candidate_fields_after_residual_removal']}
    a = [txt(.65, 1.08, 12.0, 'netFound 충돌 묶음의 행에 정보 하나를 더해 train 다수결로 validation을 예측  ·  test 미사용  ·  모델 학습 없는 조회 비교', 15.5, False, GREY)]
    spec = [('없음 (현재 토큰)', 'L2 only', '없음'), ('목적 포트', '+dst_port', '목적 포트'), ('서비스 포트 (두 포트 중 작은 값)', '+service port', '서비스 포트'),
            ('원본 seq가 0인지', '+seq==0,ack==0', '원본 seq가 0인지'), ('같은 IP 쌍의 최근 60초 포트 수', '+pair ports 60s', '같은 IP 쌍의 최근 60초 포트 수')]
    a += table(.65, 1.62, [5.03, 1.90, 2.30, 2.80], ['추가 정보', '충돌 행 수', 'val 정확도 %', '잔여 제외 val 정확도 %'],
               [(n, f"{cand[k]['conflict_rows']:,}", f"{100 * cand[k]['val_acc']:.1f}", f"{100 * aft[j]['val_acc']:.1f}") for n, k, j in spec], size=14, row_h=.44)
    a += card(.65, 4.46, 3.85, 2.20, '목적 포트  충돌 수만 감소', '충돌 99% 제거, 정확도 변화 없음\n스캔 응답의 목적 포트는\n스캐너의 임시 포트', accent=RED, body_size=14.5)
    a += card(4.72, 4.46, 3.85, 2.20, '원본 seq  토큰이 지운 정보', '패킷 1개 flow의 상대 seq는 항상 0\n원본 0이면 닫힌 포트가\nSYN에 답한 RST', accent=AMBER, body_size=14.5)
    auc = b['rank_power_class_vs_benign_validation']['all|portscan']
    a += card(8.79, 4.46, 3.89, 2.20, '맥락  flow 묶음의 성질', f"스캔은 flow 하나로 정의되지 않음\n스캔 대 정상 단일 feature AUC\n{auc['stats'][0][0]:.3f}에서 {auc['ctx'][0][0]:.3f}로\n과거 flow만, IP는 묶는 키로만", accent=TEAL, body_size=14.5)
    page('문제1  다른 입력과 구별되는 입력', a,
         '충돌 수가 줄어도 구별력이 오르는 것은 아니므로 후보는 validation 정확도로 비교', '10 문제1',
         '입력을 다르게 만드는 것만으로는 부족하다. 목적 포트를 넣으면 충돌 행이 99% 줄지만 validation 정확도는 그대로다. 스캔 응답에서 목적 포트는 스캐너의 임시 포트라 train에서 본 값이 다시 나오지 않기 때문이다. 반대로 원본 seq가 0인지 나타내는 값은 충돌을 거의 줄이지 않는데 정확도를 올린다. 패킷 1개 flow는 자기 자신이 기준이라 상대 seq가 항상 0으로 지워지지만, 원본 seq가 0이면 닫힌 포트가 SYN에 답한 RST다. 스캔은 flow 묶음의 성질이므로 같은 IP 쌍이 최근 60초에 접촉한 포트 수가 가장 잘 가른다. 이 값은 과거 flow만 쓰고 IP를 벡터에 넣지 않는다. 표는 다수결 조회 비교이고 val의 스캔 충돌은 대부분 목요일 세션에서 나왔으므로, 다른 공격자에 대한 일반화는 아직 검증하지 않았다.', [])

    # 12. Problem 2.
    a = [txt(.65, 1.08, 12.0, '정상 포함 7 family · class당 라벨 n개 · seed 5개 · 전체 평가 집합 · 두 모델에 같은 행', 16, False, GREY)]
    a.append(asset_image('fewshot_three_views', .45, 1.45, 12.45, 3.45))
    n50 = {k: few[(k, 'tabpfn')][50] for k in ('csv', 'flowstats-v1', 'netfound-v2-small')}
    n50x = {k: few[(k, 'xgb')][50] for k in ('csv', 'flowstats-v1', 'netfound-v2-small')}
    a += card(.65, 5.05, 3.95, 1.55, 'head 포화는 세 열 동일', '공격만 n=20  portscan 98~99.6%\nddos 89~99% · dos 85~98%\nn=50까지 1~2 pp 변화', accent=BLUE, body_size=14.5)
    a += card(4.85, 5.05, 3.95, 1.55, 'n=50 macro-F1  XGB / TabPFN', f'CSV  {n50x["csv"]:.1f} / {n50["csv"]:.1f}\n통계 45  {n50x["flowstats-v1"]:.1f} / {n50["flowstats-v1"]:.1f}\nnetFound  {n50x["netfound-v2-small"]:.1f} / {n50["netfound-v2-small"]:.1f}', accent=GREEN, body_size=14.5)
    ov = BOUNDARY['fewshot_overlap_n50_netfound']
    a += card(9.05, 5.05, 3.63, 1.55, 'netFound 격차는 비중복 행', f"학습 350행과 같은 토큰열 test {ov['test_rows_equal_to_350_train_rows_pct']:.1f}%\n그 행 정확도 두 모델 {ov['acc_on_those']['tabpfn']:.1f}\n나머지  XGB {ov['acc_on_rest']['xgb']:.1f} · TabPFN {ov['acc_on_rest']['tabpfn']:.1f}", accent=RED, body_size=13)
    page('문제2  라벨 n개로 얼마나 오르나', a,
         'head의 조기 포화는 표현과 무관하고, netFound+TabPFN의 우위는 학습과 겹치지 않는 행에서 나옴', '11 문제2',
         '세 열 모두 같은 few-shot 규약이다. 공격만 과제에서 portscan·ddos·dos는 n=20에서 85~99%에 이르고 n=50으로 가도 1~2 pp만 변한다. 정상 포함 macro-F1은 n=50에서 netFound+TabPFN이 가장 높고 netFound+XGB가 가장 낮아 표현보다 표현과 분류기의 조합이 가른다. test의 56%가 train과 같은 토큰열이라는 값은 train 전체와 비교한 것이고, n=50 학습 350행과 같은 토큰열은 test의 8.7%뿐이다. 그 행의 정확도는 두 모델 모두 97.0%이고 우위는 나머지 행의 91.1 대 97.5에서 나온다. n=2에서 두 곡선이 만나는 것은 5 seed 평균뿐이다. seed별로는 최대 7~12 pp 벌어지고 두 모델의 예측 일치율도 34~64%다. 같은 14행으로 만든 1-NN도 같은 수준이고 XGB의 트리는 전부 잎 2개 이하이므로, 출발점은 모델이 아니라 14행과 표현이 정한다.', [])

    # 13. Label-policy variant (a).
    a = [txt(.65, 1.08, 12.0, '같은 corpus·분할·인코딩에서 보류 154,743개의 라벨만 바꿔 다시 측정  ·  exclude = 본 실행, attack = CSV 방식, benign = EXP93 방식', 15.5, False, GREY)]
    pol = [
        ('exclude (본 실행)', '0.04', '0.28', '4.86', '83.5 / 81.4', '65.2 / 69.0', '57.8 / 71.6', '10.4 / 2.9'),
        ('attack (CSV 방식)', '0.04', '5.71', '4.77', '75.1 / 75.8', '68.1 / 70.2', '57.6 / 66.4', '36.8 / 22.1'),
        ('benign (EXP93 방식)', '5.42', '5.65', '10.15', '83.5 / 81.4', '63.7 / 68.9', '57.1 / 69.0', '11.4 / 7.8'),
    ]
    a += table(.65, 1.62, [2.2, .9, 1.2, 1.45, 1.55, 1.5, 1.85, 1.38], ['보류 정책', 'L1 충돌', '통계 충돌', '토큰 충돌', '통계 공격만 n=50', '통계 정상 n=50', 'netFound 정상 n=50', 'bot n=20 통계'], pol, size=12.5, row_h=.50)
    a.append(txt(.65, 3.75, 12.0, '충돌은 라벨 있는 flow 대비 %  ·  macro-F1과 class F1은 XGB / TabPFN', 12.5, False, GREY))
    pv = BOUNDARY['conflict_rows_by_label_policy']
    a += card(.65, 4.20, 3.95, 2.25, 'attack  통계 충돌 급증', f"통계 view 충돌 0.28 → 5.71%\n그중 dos 130,504행 (Hulk 잔여 flow)\n잔여 제외 시 {pv['attack']['L3s']['before']['conflict_rows']:,} → {pv['attack']['L3s']['after']['conflict_rows']}행\n공격만 macro n=50  8 pp 하락\nnetFound+TabPFN 71.6 → 66.4", accent=RED, body_size=14, align='left')
    a += card(4.85, 4.20, 3.95, 2.25, 'benign  L1 충돌 급증', f"L1 정보가 같은데 라벨 다름\n0.04 → 5.42% (Hulk 130,500행이 정상)\n잔여 제외 시 {pv['benign']['L1']['before']['conflict_rows']:,} → {pv['benign']['L1']['after']['conflict_rows']}행\nfew-shot 변화 −0.1 ~ −2.6 pp\n공격만 과제는 영향 없음", accent=AMBER, body_size=14, align='left')
    a += card(9.05, 4.20, 3.63, 2.25, 'exclude  가장 깨끗', f"충돌·상한·곡선 모두 최선\n세 상태 라벨을 기본 규약으로 유지\n잔여 제외 후 보류 {BOUNDARY['residual_counts']['unresolved_remaining']:,}개만\nhost·시간 집계 과제로", accent=TEAL, body_size=14, align='left')
    page('보류 flow를 어디에 두느냐 (변형 a)', a,
         '보류를 공격으로 두면 통계 충돌이 5.71%로 뛰지만 원인은 잔여 flow, 제외하면 정책 간 충돌 차이 소멸', '11 문제2',
         '같은 corpus와 인코딩 위에서 보류 flow의 라벨만 세 가지로 바꿨다. 공격으로 넣는 CSV 방식에서는 Hulk 세션의 잔여 flow 130,504행이 정상 세션의 잔여 flow와 같은 통계 벡터를 가져 통계 view 충돌이 5.71%로 뛰고, 공격만 macro-F1이 8 pp 내려간다. 반면 정상 포함 통계 view n=20은 재접속·무payload 행이 쉬운 표본으로 들어가 bot과 dos가 오른다. 정상으로 넣는 방식은 L1 정보가 같은 확정 dos 25행과 한 묶음을 만들어 L1 충돌이 5.42%가 된다. 재분석에서 잔여 flow를 빼면 attack 정책의 통계 충돌은 138,527행에서 75행으로, benign 정책의 L1 충돌은 131,418행에서 2행으로 줄어 정책 간 충돌 차이가 사라졌다. 표의 macro-F1은 현재 flow 규칙의 측정값이다. 제외가 충돌·상한·곡선 모두에서 가장 깨끗하다.', [])

    # 14. Verdict.
    a = []
    ex = BOUNDARY['conflict_rows_by_label_policy']['exclude']
    a += card(.65, 1.45, 3.95, 2.55, '문제1  동일 입력·상이 라벨', f"주원인은 flow 경계의 잔여 flow\n잔여 제외 시 통계 0.28 → {ex['L3s']['after']['pct']:.3f}%\nnetFound 토큰 4.86 → {ex['L2']['after']['pct']:.2f}%\n\n상한은 모두 99.8% 이상", accent=RED, body_size=15)
    a += card(4.85, 1.45, 3.95, 2.55, '문제2  head 조기 포화', '세 표현에서 동일\n대량·반복 공격의 성질\n\n표현이 바꾸는 것은\ntail(bf·bot)과 모델 차이의 부호', accent=BLUE, body_size=15)
    a += card(9.05, 1.45, 3.63, 2.55, '남은 과제  희소 class 오탐', 'bf·bot·web 오탐은 패킷 중앙값\n11~109개의 정상 세션, 잔여 flow 0%\n\n잔여 제외 test macro-F1\nn=50에서 0.5~1.4 pp 상승', accent=AMBER, body_size=15)
    a += [
        box(.65, 4.35, 12.03, 1.05, 'F6F9FC', 'D3DFE9'),
        txt(.90, 4.60, 11.5, 'PCAP 전환의 산출물은 새 표현이 아니라 정리된 평가 규약', 19, True, NAVY, 'center'),
        txt(.90, 5.02, 11.5, '세 상태 라벨 · 시간순 분할 · 층별 충돌 점검과 상한 · 중복률을 본 연구의 평가 프로토콜로 채택', 14.5, False, GREY, 'center'),
    ]
    nexts = [('(a) 보류 → 공격 / 정상 두 정책 비교', '완료'), ('(b) PCA 128 · 512', '중단'), ('(c) 잔여 flow 제거, 원본 seq 표시, 60초 맥락 feature', '후보'), ('CIC2018 전체 450 GB 수집', '하지 않음')]
    for i, (s_, state) in enumerate(nexts):
        y = 5.70 + (i % 2) * .50
        x = .65 + (i // 2) * 6.1
        a += chip(x, y, .95, state, NAVY if state == '완료' else (TEAL if state == '후보' else GREY), 'FFFFFF', 12.5, .38)
        a.append(txt(x + 1.05, y + .07, 4.9, s_, 14.5, False, INK))
    page('판정과 다음 단계', a,
         '재집계 기준 문제1의 충돌은 flow 경계 수정으로 대부분 사라지고, 남은 과제는 희소 class의 오탐', '12 판정',
         '문제1의 충돌은 대부분 flow 경계가 만든 잔여 flow에서 나왔고, 이를 제외한 재집계에서 통계 충돌은 0.003%, netFound 토큰 충돌은 2.04%로 줄었다. 문제2는 데이터의 성질이라 표현과 무관하다. 대신 세 상태 라벨, 시간순 분할, 층별 충돌 점검과 oracle 상한, 중복률이 정리됐고 이를 본 연구의 평가 프로토콜로 가져간다. 보류 flow의 라벨 정책 변형은 제외가 가장 깨끗함을 보였고, 정책 간 충돌 차이도 잔여 flow를 빼면 사라졌다. PCA 차원 변형은 불필요하다고 판단해 중단했다. 잔여 flow를 test에서 빼도 n=50 macro-F1은 0.5~1.4 pp만 오르고 bf·bot·web 점수는 변하지 않는다. 이 class의 오탐은 잔여 flow가 없고 패킷 중앙값이 11~109개인 정상 세션이기 때문이다. 잔여 flow 제거, 원본 seq 표시, 최근 60초 맥락 feature를 나누어 평가하는 변형이 다음 후보다. CIC2018 전체 수집은 하지 않는다.', [])

    # 15. Appendix table.
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
    canvas = Image.new('RGB', (1500, 10 * 445), '#DCE3EB')
    for index, image_path in enumerate(paths):
        image = Image.open(image_path)
        image.thumbnail((730, 411))
        x, y = (index % 2) * 750 + 10, (index // 2) * 445 + 8
        canvas.paste(image, (x, y))
        ImageDraw.Draw(canvas).text((x, y + 414), str(index + 1), fill='black')
    canvas.save(QA / 'contact-1.png')
    (QA / 'scenes.json').write_text(json.dumps(r.SCENES, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    manifest = {
        'date': '2026-11-04', 'revised': '2026-11-06', 'slides': len(r.SCENES), 'follows': 'kor/20261029_PCAP_연구방향.pptx',
        'pcap_summary': str(PCAP_SUMMARY.relative_to(ROOT)), 'csv_summary': str(CSV_SUMMARY.relative_to(ROOT)), 'csv_audit': str(CSV_AUDIT.relative_to(ROOT)), 'flow_boundary': str(BOUNDARY_PATH.relative_to(ROOT)), 'packet_walkthrough': str(WALK_PATH.relative_to(ROOT)),
        'validation': 'shared editable PPTX/HTML-PDF scenes; geometry/font/text checks; visual contact sheet reviewed',
        'pages': checks,
        'pptx_sha256': hashlib.sha256(OUT.with_suffix('.pptx').read_bytes()).hexdigest(),
        'pdf_sha256': hashlib.sha256(OUT.with_suffix('.pdf').read_bytes()).hexdigest(),
    }
    (QA / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pptx': str(OUT.with_suffix('.pptx')), 'pdf': str(OUT.with_suffix('.pdf')), 'slides': len(r.SCENES)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
