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
r.N_SLIDES = 8
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
def scenes() -> None:
    pcap, csvs, audit = load()
    layer = make_layer_asset(pcap, audit)
    few = make_fewshot_asset(pcap)
    status = pcap['corpus_manifest']['label_status_counts']
    total = sum(status.values())
    bg, conf, unres = status['background_benign'], status['confirmed_attack'], status['unresolved_attempted']

    # 1. What happened since 10/29: plan -> EXP93 limits -> EXP94.
    a = [txt(.65, 1.08, 12.0, '10/29 계획은 CIC2018 PCAP 재표현이었으나 host 1대 추출의 한계로 CIC2017 전체 망으로 전환', 17, False, GREY)]
    a += card(.65, 1.70, 3.45, 3.05, '10/29 계획', 'PCAP에서 인코더로 벡터 생성\n\n충돌·조기 포화를\n같은 조건에서 재점검', accent=BLUE, body_size=17)
    a += [arrow(4.27, 3.05, .40)]
    a += card(4.85, 1.70, 3.90, 3.05, 'EXP93  CIC2018 host 1대', '날짜당 피해 host 1대만 추출\n\n✕ 정상 모집단이 CSV와 다름\n✕ 일부 공격 family 누락\n✕ Bot은 피해 host가 아닌 파일', accent=RED, body_size=16, align='left')
    a += [arrow(8.93, 3.05, .40)]
    a += card(9.50, 1.70, 3.18, 3.05, 'EXP94  CIC2017 전체 망', '5일 전체 망 캡처 51 GB\n\n✓ 정상 모집단이 CSV와 동일\n✓ CSV와 같은 망에서 비교\n✓ 라벨을 세 상태로 재부여', accent=TEAL, body_size=16, align='left')
    steps = [('10/29', '연구 방향 발표'), ('11/02', 'EXP93 host 1대'), ('11/03', 'CSV 기준 실험'), ('11/04', 'PCAP 실행·비교')]
    for i, (d, s) in enumerate(steps):
        x = .65 + i * 3.05
        a += chip(x, 5.30, 1.05, d, NAVY, 'FFFFFF', 15, .42)
        a.append(txt(x + 1.15, 5.36, 1.85, s, 15, False, INK))
        if i < 3:
            a.append(arrow(x + 2.72, 5.40, .24, .22))
    a.append(txt(.65, 6.10, 12.0, 'CIC2018은 host별 캡처 약 440개/일, 10일 450 GB  ·  CIC2017은 스위치 캡처 하루 1개 파일, 5일 51 GB', 15, False, GREY))
    page('지난 발표 이후의 진행', a,
         'host 1대 추출은 비교 조건을 깨뜨려 전체 망 캡처인 CIC2017로 전환', '06 진행 경과',
         '10/29 발표의 계획은 CIC2018 PCAP에서 인코더로 새 벡터를 만들어 두 문제를 재점검하는 것이었다. CIC2018은 host별 캡처라 날짜당 피해 host 1대만 받아 실행했는데(EXP93), 그 결과 정상 트래픽이 그 host 것으로 바뀌어 CSV와 모집단이 달라졌고, 일부 공격이 빠졌으며, Bot은 피해 host가 아닌 파일을 골라 라벨이 틀렸다. 전체 450 GB 대신 스위치에서 전체 망을 캡처한 CIC2017(5개 파일, 51 GB)로 전환해 EXP94를 수행했다.', ['cic18', 'cic17'])

    # 2. Pipeline in six numbered steps.
    a = [txt(.65, 1.08, 12.0, '같은 flow·라벨·분할 위에 두 가지 벡터 표현을 얹어, 표현만 바꿨을 때의 차이를 본다', 17, False, GREY)]
    blocks = [
        (.65, '① PCAP 5개', '56.4M 패킷\n시각 + 패킷 바이트'),
        (3.30, '② flow로 묶기', '같은 두 끝점의 패킷\nRST·FIN·SYN·120초 종료'),
        (5.95, '③ 라벨 붙이기', '시간창 + 끝점 + 증거\n세 상태로 부여'),
        (8.60, '④ 시간순 나누기', '앞 70% 학습, 뒤 15% 평가\n날짜·family마다'),
    ]
    for x, title, body in blocks:
        a += block(x, 1.70, 2.45, 1.42, title, body, body_size=14)
    for x in [3.12, 5.77, 8.42, 11.07]:
        a.append(arrow(x, 2.30, .20))
    a += block(11.28, 1.70, 1.40, 1.42, '⑥ 평가', '충돌 점검\nfew-shot', body_size=14)
    a += card(.65, 3.65, 5.85, 2.15, '⑤ view 1  통계 45개', 'CSV와 같은 종류의 요약을 깨끗한 flow 위에서 다시 계산\n지속 시간, 패킷 수·바이트, 길이·간격 통계, flag, window, 포트\n→ 45차원 그대로 분류기 입력', accent=GREEN, body_size=15)
    a += card(6.83, 3.65, 5.85, 2.15, '⑤ view 2  netFound', '패킷 헤더를 토큰으로 바꿔 사전학습 transformer에 입력\nflow당 512차원 벡터 → 학습 데이터로 맞춘 PCA 64차원\n→ IP·포트·시각은 넣지 않음', accent=RED, body_size=15)
    a.append(txt(.65, 6.15, 12.0, f'결과  flow {total:,}개  ·  학습 XGBoost와 TabPFN, class당 라벨 2~50개, 전체 평가 집합 340,543 flow', 15, False, GREY))
    page('PCAP에서 벡터까지 여섯 단계', a,
         '표현(⑤)만 다르고 나머지 단계는 두 view가 공유', '07 파이프라인',
         'PCAP은 시각과 패킷 바이트의 나열이다. 같은 두 끝점 사이의 패킷을 flow로 묶고(RST, 양쪽 FIN, 새 SYN, 120초로 종료), 교정 규칙으로 라벨을 세 상태로 붙인 뒤 날짜와 family마다 시간순으로 70/15/15로 나눈다. 그 위에 두 view를 얹는다. view 1은 CSV와 같은 종류의 통계 45개, view 2는 패킷 헤더 토큰을 고정된 netFound에 넣어 얻은 512차원을 PCA 64차원으로 줄인 것이다. 두 view는 같은 flow·라벨·분할·few-shot 표본을 받는다.', ['netfound'])

    # 3. One flow followed to the end.
    a = [txt(.65, 1.08, 12.0, '월요일 11:57 UTC, 사내 PC → 외부 웹서버(:80) 접속 1건  ·  패킷 10개, payload 97/179 B, 60.6 ms, 양쪽 FIN으로 종료', 16, False, GREY)]
    a += [txt(.65, 1.62, 3.9, '첫 패킷(SYN)의 토큰', 18, True, NAVY)]
    fields = [('IP 헤더 길이', '20'), ('TOS', '0'), ('IP 전체 길이', '52'), ('IP flag (DF)', '2'), ('TTL (Windows)', '128'), ('TCP flag (SYN)', '2'), ('window', '8192'), ('상대 seq·ack', '0 0 0 0'), ('urgent', '0')]
    for i, (k, v) in enumerate(fields):
        y = 2.05 + i * .40
        a.append(box(.65, y, 2.55, .36, PALE if i % 2 == 0 else 'FFFFFF', 'D3DFE9'))
        a.append(txt(.75, y + .07, 2.4, k, 13.5, False, INK))
        a.append(box(3.20, y, 1.15, .36, PALE if i % 2 == 0 else 'FFFFFF', 'D3DFE9'))
        a.append(txt(3.25, y + .07, 1.05, v, 13.5, True, NAVY, 'right'))
    a.append(txt(.65, 5.72, 3.9, 'IP·포트·시각은 토큰에 없음', 14, False, GREY))
    a += [txt(4.95, 1.62, 3.6, 'view 1  통계 45개 중 일부', 18, True, NAVY)]
    stats = [('지속 시간', '60,574 µs'), ('보낸·받은 패킷', '5 · 5'), ('보낸 길이 평균·최대', '61.8 · 137'), ('받은 길이 평균·최대', '78.2 · 219'), ('패킷 간격 평균', '6,730 µs'), ('SYN·FIN·PSH 수', '2 · 2 · 2'), ('초기 window', '8192 · 29200'), ('초당 패킷', '165'), ('목적 포트', '80')]
    for i, (k, v) in enumerate(stats):
        y = 2.05 + i * .40
        a.append(box(4.95, y, 2.25, .36, PALE if i % 2 == 0 else 'FFFFFF', 'D3DFE9'))
        a.append(txt(5.05, y + .07, 2.1, k, 13.5, False, INK))
        a.append(box(7.20, y, 1.45, .36, PALE if i % 2 == 0 else 'FFFFFF', 'D3DFE9'))
        a.append(txt(7.25, y + .07, 1.35, v, 13.5, True, GREEN, 'right'))
    a += [txt(9.25, 1.62, 3.5, 'view 2  burst와 netFound', 18, True, NAVY)]
    bursts = [('f 1', BLUE), ('b 1', TEAL), ('f 2', BLUE), ('b 3', TEAL), ('f 2', BLUE), ('b 1', TEAL)]
    for i, (s, c) in enumerate(bursts):
        a += chip(9.25 + i * .58, 2.08, .52, s, c, 'FFFFFF', 12, .40)
    a.append(txt(9.25, 2.58, 3.5, '6 burst (방향 · 패킷 수)\n간격 0·0·20·20·21·21 ms\n같은 방향 10 ms 안의 패킷이 한 burst', 13.5, False, INK))
    a += block(9.25, 3.45, 3.43, 1.05, 'netFound (고정)', '토큰열 → 512차원 벡터', body_size=14.5)
    a.append(arrow(10.85, 4.58, .24, .22, down=True))
    a += block(9.25, 4.92, 3.43, 1.05, 'PCA 64', '학습 데이터로만 맞춤 · 분산 98.1%', body_size=14.5)
    a.append(txt(9.25, 6.12, 3.5, '사전학습: 라벨 없는 트래픽의\n가려진 헤더 토큰 맞히기', 13.5, False, GREY))
    page('flow 하나를 두 view로 바꾸면', a,
         'view 1은 사람이 정한 요약(포트 포함), view 2는 헤더 토큰에서 배운 벡터(포트·IP·시각 없음)', '07 파이프라인',
         '월요일의 짧은 HTTP 접속 한 건을 끝까지 따라간다. 패킷 하나는 IP와 TCP 헤더 필드 값을 정수 토큰으로 옮긴 것이며 IP·포트·시각은 넣지 않는다. view 1은 이 flow의 모든 패킷에서 통계 45개를 계산한다(목적 포트 포함). view 2는 같은 방향 10 ms 안의 패킷을 burst로 묶고 처음 12 burst, burst당 6 패킷의 토큰을 고정된 netFound에 넣어 512차원 벡터를 얻은 뒤 PCA 64차원으로 줄인다.', ['netfound'])

    # 4. Labels: three states and what unresolved contains.
    a = [txt(.65, 1.08, 12.0, '교정 규칙 53개(CNS 2022)를 그대로 적용  ·  시간창과 끝점만 맞고 공격 행위가 없는 flow는 보류', 16, False, GREY)]
    qs = [('공격자 IP ↔ 피해자 IP?', '아니오 → 정상(background)'), ('공격 시간창 안?', '창 밖 → 정상(background)'), ('공격 행위의 증거?  payload·RST·포트', '없음 → 보류(unresolved)')]
    for i, (q, no) in enumerate(qs):
        y = 1.72 + i * 1.08
        a += chip(.65, y, 4.35, q, NAVY, 'FFFFFF', 14.5, .50)
        a.append(txt(.75, y + .58, 3.6, no, 13.5, False, GREY))
        if i < 2:
            a.append(arrow(4.55, y + .55, .22, .22, down=True))
    a += chip(.65, 4.96, 4.35, '모두 예 → 공격(confirmed)', RED, 'FFFFFF', 14.5, .50)
    a.append(txt(5.65, 1.62, 7.0, f'flow {total:,}개의 라벨 상태', 18, True, NAVY))
    bw = 7.0
    parts = [(bg, GREEN, '정상'), (conf, RED, '공격'), (unres, AMBER, '보류')]
    x = 5.65
    for n, c, name in parts:
        w = bw * n / total
        a.append(box(x, 2.05, w, .62, c))
        if w > .9:
            a.append(txt(x + .05, 2.21, w - .1, f'{name} {100 * n / total:.1f}%', 14, True, 'FFFFFF', 'center'))
        x += w
    a.append(txt(5.65, 2.78, 7.0, f'정상 {bg:,}  ·  공격 {conf:,}  ·  보류 {unres:,} (6.4%)', 14, False, INK))
    a.append(txt(5.65, 3.35, 7.0, '보류 flow의 내역', 18, True, NAVY))
    items = [('DoS Hulk  payload 없는 연결', 137908), ('Bot  master 종료 뒤 재접속', 4067), ('FTP brute force  payload 없음', 4039), ('Slowloris·Slowhttptest  무응답', 4439), ('Web 공격  payload 없음', 1869), ('DDoS LOIC  payload 없음', 1203)]
    for i, (name, n) in enumerate(items):
        a += hbar(5.65, 3.80 + i * .44, 3.45, 2.35, name, n, 137908, AMBER, f'{n:,}', h=.30, size=13.5)
    a.append(txt(5.65, 6.48, 7.0, '보류는 공격에도 정상에도 넣지 않고 학습·채점에서 제외, 수만 집계', 14, False, GREY))
    page('라벨의 세 상태', a,
         f'창 안이지만 공격 행위가 없는 flow {unres:,}개는 보류로 분리하고 그 89%는 무효였던 DoS Hulk', '08 라벨',
         '규칙은 시간창, 공격자와 피해자 끝점, 그리고 공격마다 다른 증거 조건으로 되어 있다. 셋을 모두 만족하면 공격, 창과 끝점만 맞고 증거가 없으면 보류, 어느 규칙에도 걸리지 않으면 정상이다. 보류는 저자들이 Attempted라 부른 7범주와 같다. 보류 flow를 공격으로 넣으면 정상 실패 접속과 같은 입력에 다른 라벨을 붙이는 셈이고, 정상으로 넣으면 공격자 트래픽이 정상에 섞인다. 그래서 제외하고 수만 집계하며, 두 정책의 영향은 변형 실험으로 따로 잰다.', ['liu', 'attempted'])

    # 5. Problem 1: where inputs become identical.
    a = [txt(.65, 1.08, 12.0, '같은 flow를 세 단계의 입력으로 보고, 각 단계에서 "입력은 같은데 family가 다른" flow를 센다', 16, False, GREY)]
    a.append(asset_image('layer_conflicts', .55, 1.52, 7.55, 3.35))
    a += card(8.35, 1.55, 4.33, 3.30, '충돌의 정체  패킷 1개짜리 RST', '닫힌 포트에 보낸 SYN의 RST 응답과\n끝난 연결의 늦은 RST가 같은 모양\n\n토큰  [20, 0, 40, 2, TTL, RST, 0, …]\nLinux  정상 49,204 + 스캔 59\nWindows  정상 11,817 + 스캔 4,363\n\n포트가 없는 토큰은 TTL로만 구분', accent=RED, body_size=14, align='left')
    a += [
        txt(.65, 5.12, 6.0, 'L1  라벨 자체가 못 가르는 양  0.04%', 16, True, NAVY),
        txt(.65, 5.50, 6.0, 'L3 통계 45  CSV와 같은 수준  0.28%', 16, True, GREEN),
        txt(.65, 5.88, 6.0, 'L2 netFound 토큰  절삭과 포트 부재로 4.86%', 16, True, RED),
        txt(7.0, 5.12, 5.7, '어떤 모델도 넘지 못하는 상한(oracle)', 16, True, NAVY),
        txt(7.0, 5.50, 5.7, 'CSV 99.997%  ·  통계 100%  ·  netFound 토큰 99.79%', 15, False, INK),
        txt(7.0, 5.88, 5.7, '충돌이 점수를 깎는 양은 모든 표현에서 작음', 15, False, INK),
    ]
    page('문제1  입력이 같아지는 단계', a,
         '충돌은 라벨이 아니라 통계 요약과 토큰 절삭에서 생기며, 포트 없는 netFound 토큰에서 가장 큼', '09 문제1',
         'L1은 flow가 가진 정보 전부(통계 45개와 자르기 전 토큰 전부)이고, 여기서 같으면 파이프라인에 가를 정보가 없다. L2는 netFound tokenizer 출력, L3는 분류기가 받는 벡터다. 라벨 쪽 하한은 0.04%, 통계 요약은 CSV와 같은 0.28%, netFound 토큰은 4.86%다. 충돌 group의 flow를 열어 보면 거의 전부 패킷 1개짜리 RST flow다. 포트 스캔의 RST 응답과 정상 트래픽의 늦은 RST는 헤더가 같고, 포트·IP·시각이 없는 토큰에서는 TTL로만 갈린다. 그래도 세 표현 모두 oracle 상한이 99.8% 이상이라 충돌이 점수를 깎는 양은 작다.', [])

    # 6. Problem 2: few-shot curves in three columns.
    a = [txt(.65, 1.08, 12.0, '정상 포함 7 family · class당 라벨 n개 · seed 5개 · 전체 평가 집합 · 두 모델에 같은 행', 16, False, GREY)]
    a.append(asset_image('fewshot_three_views', .45, 1.45, 12.45, 3.45))
    n50 = {k: few[(k, 'tabpfn')][50] for k in ('csv', 'flowstats-v1', 'netfound-v2-small')}
    n50x = {k: few[(k, 'xgb')][50] for k in ('csv', 'flowstats-v1', 'netfound-v2-small')}
    a += card(.65, 5.05, 3.95, 1.55, 'head 포화는 세 열 동일', '공격만 n=20  portscan 98~99.6%\nddos 89~99% · dos 85~98%\nn=50까지 1~2 pp 변화', accent=BLUE, body_size=14.5)
    a += card(4.85, 5.05, 3.95, 1.55, 'n=50 macro-F1  XGB / TabPFN', f'CSV  {n50x["csv"]:.1f} / {n50["csv"]:.1f}\n통계 45  {n50x["flowstats-v1"]:.1f} / {n50["flowstats-v1"]:.1f}\nnetFound  {n50x["netfound-v2-small"]:.1f} / {n50["netfound-v2-small"]:.1f}', accent=GREEN, body_size=14.5)
    a += card(9.05, 5.05, 3.63, 1.55, 'netFound 점수의 조건', 'test의 56%가 train과\n같은 토큰열\n외워서 맞힐 수 있는 구간', accent=RED, body_size=14.5)
    page('문제2  라벨 n개로 얼마나 오르나', a,
         'head의 조기 포화는 표현과 무관하고, netFound+TabPFN의 고점은 토큰 중복 56% 조건 위의 수치', '10 문제2',
         '세 열 모두 같은 few-shot 규약이다. 공격만 과제에서 portscan·ddos·dos는 n=20에서 85~99%에 이르고 n=50으로 가도 1~2 pp만 변한다. 정상 포함 macro-F1은 n=50에서 netFound+TabPFN이 가장 높고 netFound+XGB가 가장 낮아 표현보다 표현과 분류기의 조합이 가른다. netFound view는 test의 56%가 train과 같은 토큰열이므로 그 점수의 일부는 암기로 설명될 수 있다.', [])

    # 7. Verdict and what comes next.
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
    for i, (s, state) in enumerate(nexts):
        y = 5.70 + (i % 2) * .50
        x = .65 + (i // 2) * 6.1
        a += chip(x, y, .95, state, NAVY if state == '실행 중' else (TEAL if state == '후보' else GREY), 'FFFFFF', 12.5, .38)
        a.append(txt(x + 1.05, y + .07, 4.9, s, 14.5, False, INK))
    page('판정과 다음 단계', a,
         '표현 교체로는 두 문제를 풀 수 없고, 라벨 정책과 평가 규약이 남은 변수', '11 판정',
         '문제1은 PCAP 직접 인코딩으로 사라지지 않았고, 문제2는 데이터의 성질이라 표현과 무관하다. 대신 세 상태 라벨, 시간순 분할, 층별 충돌 점검과 oracle 상한, 중복률이 정리됐고 이를 본 연구의 평가 프로토콜로 가져간다. 보류 flow의 라벨 정책 두 가지와 PCA 차원 변형은 같은 corpus와 인코딩 위에서 실행 중이며, 패킷 1개짜리 RST flow 정리와 토큰에 포트를 결합하는 변형이 다음 후보다. CIC2018 전체 수집은 하지 않는다.', [])

    # 8. Appendix: numbers behind slide 6 (n=20 class F1, normal included).
    a = [txt(.65, 1.08, 12.0, '정상 포함 n=20 family별 F1 (%) · 5 seed 평균 · 행 순서는 PCAP 평가 집합 수 내림차순', 16, False, GREY)]
    header = ['family', 'PCAP test', 'CSV XGB', 'CSV TabPFN', '통계 XGB', '통계 TabPFN', 'netFound XGB', 'netFound TabPFN']
    widths = [1.55, 1.25, 1.35, 1.45, 1.35, 1.45, 1.75, 1.88]
    def cf(view, method, name):
        c = next((c for c in pcap['views'][view]['class_f1'] if c['budget'] == 20 and c['regime'] == 'with_benign' and c['method'] == method and c['name'] == name), None)
        return c
    families = ['benign', 'portscan', 'dos', 'ddos', 'bf', 'bot', 'web']
    rows = []
    for name in families:
        c0 = cf('flowstats-v1', 'xgb', name)
        rows.append([name, f"{c0['support']:,}"] + [f"{100 * cf(v, m, name)['mean']:.1f}" for v in ('csv', 'flowstats-v1', 'netfound-v2-small') for m in ('xgb', 'tabpfn')])
    macro = ['macro', '340,543'] + [f"{100 * next(m['mean'] for m in pcap['views'][v]['macro_f1'] if m['regime'] == 'with_benign' and m['method'] == mm and m['budget'] == 20):.1f}" for v in ('csv', 'flowstats-v1', 'netfound-v2-small') for mm in ('xgb', 'tabpfn')]
    y = 1.62
    def row(values, yy, fill, bold=False, color=INK):
        items = []
        xx = .65
        for j, (w, v) in enumerate(zip(widths, values)):
            items.append(box(xx, yy, w, .44, fill, 'D3DFE9'))
            items.append(txt(xx + .08, yy + .10, w - .16, v, 14, bold, color, 'left' if j == 0 else 'right'))
            xx += w
        return items
    a += row(header, y, NAVY, True, 'FFFFFF')
    for i, values in enumerate(rows):
        a += row(values, y + .50 + i * .47, PALE if i % 2 == 0 else 'FFFFFF')
    a += row(macro, y + .50 + len(rows) * .47 + .08, 'E6F1F8', True, NAVY)
    a.append(txt(.65, 6.05, 12.0, 'bot 0.0(CSV)은 평가 집합의 bot이 전부 master 종료 뒤 재접속 flow인 CSV 라벨 결함  ·  web은 PCAP 평가 집합 16행이라 참고치', 14, False, GREY))
    page('부록  n=20 family별 F1', a,
         '', '부록',
         '슬라이드 6의 곡선을 family별로 푼 표다. CSV의 bot 0은 평가 집합의 bot 294행이 모두 master 종료 뒤 재접속 flow(저자 분류 Attempted)인데 Bot으로 라벨된 결함 때문이다. PCAP에서는 그 flow가 보류로 빠져 bot이 공격 사이에서는 구분되지만 정상과는 여전히 구분되지 않는다. web은 PCAP 평가 집합이 16행이라 수치를 해석하지 않는다.', [])


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
    canvas = Image.new('RGB', (1500, 4 * 445), '#DCE3EB')
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
