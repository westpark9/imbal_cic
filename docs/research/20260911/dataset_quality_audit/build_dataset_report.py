"""Build lablog/html_report/dataset_quality_0911.html (v2 structure: problem -> unreachable oracle -> dedup experiment -> time composition -> verdict)."""
import json, sys, html, glob, os, datetime as dt
import numpy as np, pandas as pd

SP, OUT = sys.argv[1], sys.argv[2]
ROOT = '/home/user/Desktop/imbalcic'; AUD = f'{ROOT}/docs/research/20260911/dataset_quality_audit'
twins = json.load(open(f'{SP}/report_twins.json')); tdata = json.load(open(f'{SP}/report_time.json'))
pairs = json.load(open(f'{AUD}/pair_conflicts_unordered.json')); bank_orc = json.load(open(f'{AUD}/cic_bank_realistic_oracle.json')); v2 = pd.read_csv(f'{AUD}/ceilings_v2.csv'); deriv = json.load(open(f'{AUD}/cic_bank_oracle_derivation.json'))
summ = pd.read_csv(f'{SP}/audit_summary.csv').set_index('dataset'); cls = pd.read_csv(f'{SP}/audit_classes.csv')
unsw_xgb = pd.read_csv(f'{SP}/unsw_xgb_plain.csv').set_index('class')
bot_xgb = pd.read_csv(f'{ROOT}/tabpfn/results_past/20260818_111717_nfv3_botiot_exp2_fullxgb/per_class_metrics.csv').set_index('class')
ton_xgb = pd.read_csv(f'{ROOT}/tabpfn/results_past/20260818_113117_nfv3_toniot_exp2_fullxgb/per_class_metrics.csv').set_index('class')
cic_global = deriv['global_per_class_f1']; row_oracle = deriv['row_oracle_per_class']

DS = [('cse_cic_ids2018', 'CIC-IDS2018', 'NF-CSE-CIC-IDS2018-v3'), ('ton_iot', 'ToN-IoT', 'NF-ToN-IoT-v3'), ('bot_iot', 'BoT-IoT', 'NF-BoT-IoT-v3'), ('unsw_nb15', 'UNSW-NB15', 'NF-UNSW-NB15-v3')]
SLOTS = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
KEYF = ['PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS', 'MIN_TTL', 'MAX_TTL', 'SRC_TO_DST_AVG_THROUGHPUT', 'TCP_WIN_MAX_IN']

def esc(s): return html.escape(str(s))
def fmt(n): return f'{int(n):,}'
def pct(x, d=1): return f'{x*100:.{d}f}%'
def day(ms): return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime('%m-%d')
def daytime(ms): return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime('%Y-%m-%d %H:%M')
def v2c(name, c, col): return float(v2[(v2.dataset == name) & (v2.cls == c)][col].iloc[0])
def v2macro(name, col): return float(v2[v2.dataset == name][col].mean())
def cnum(name, c, col): return float(cls[(cls.dataset == name) & (cls.cls == c)][col].iloc[0])
def orc(name, c): return bank_orc['realistic_oracle_bank']['per_class'][c] if name == 'cse_cic_ids2018' else v2c(name, c, 'ceil_majority_test')
def orc_label(name): return '현실적 oracle (EXP39 bank + 다수 라벨)' if name == 'cse_cic_ids2018' else '현실적 oracle (완벽 라우팅 + 다수 라벨)'
def tic(name, c): return v2c(name, c, 'test_rows_in_test_internal_conflict')
def train_ceiling(name, c):
    """Upper bound on F1 for a model that answers each vector with the label TRAIN gives it:
    a fraction p of this class's test rows sit on vectors whose train majority label is a different
    class, so recall <= 1-p and (with perfect precision) F1 <= 2(1-p)/(2-p)."""
    pconf = float(cnum(name, c, 'chrono_test_conflict'))
    return 2 * (1 - pconf) / (2 - pconf) if pconf < 1 else 0.0
def trc(name, c): return v2c(name, c, 'test_rows_conflicting_with_train_label')

def achieved(name):
    if name == 'cse_cic_ids2018': return cic_global, 'TabPFN global, EXP39 test (0909 §3-C)'
    if name == 'bot_iot': return bot_xgb.f1.to_dict(), 'full-pool XGBoost, benign 포함 (0818 run 20260818_111717)'
    if name == 'ton_iot': return ton_xgb.f1.to_dict(), 'full-pool XGBoost, benign 포함 (0818 run 20260818_113117)'
    return unsw_xgb.F1.to_dict(), 'XGBoost 300 trees, family별 시간순 60/20/20 (0911 감사, 단일 seed)'

# ---------------- charts ----------------
def class_colors(classes, shares):
    atk = [c for c in sorted(classes, key=lambda c: -shares[c]) if c != 'benign']; m = {'benign': 'neutral'}
    for i, c in enumerate(atk[:7]): m[c] = SLOTS[i]
    for c in atk[7:]: m[c] = 'other'
    return m

def svg_composition(name):
    d = tdata[name]; classes = d['classes']; bins = d['bins']; n = len(bins)
    total = {c: sum(b['share'][c] * b['rows'] for b in bins) for c in classes}; rows = sum(b['rows'] for b in bins)
    shares = {c: total[c] / rows for c in classes}; cm = class_colors(classes, shares)
    order = ['benign'] + [c for c in sorted(classes, key=lambda c: -shares[c]) if c != 'benign']
    W, H, L, R, T, B = 960, 250, 44, 12, 14, 44; pw, ph = W - L - R, H - T - B; bw = pw / n
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{esc(name)} class share over time-ordered bins">', f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" class="plot"/>']
    for g in (0.25, 0.5, 0.75): out.append(f'<line x1="{L}" x2="{L+pw}" y1="{T+ph*(1-g):.1f}" y2="{T+ph*(1-g):.1f}" class="grid"/>')
    for i, b in enumerate(bins):
        y0 = T + ph; x = L + i * bw
        for c in order:
            s = b['share'][c]
            if s <= 0: continue
            h = s * ph; y0 -= h
            out.append(f'<rect x="{x:.2f}" y="{y0:.2f}" width="{bw-0.6:.2f}" height="{max(h,0.3):.2f}" class="f-{cm[c]}"><title>{esc(c)} {s*100:.2f}% · {day(b["t_start"])}–{day(b["t_end"])} · {fmt(b["rows"])}행</title></rect>')
    for g, lab in ((0, '0'), (0.5, '50'), (1, '100%')): out.append(f'<text x="{L-6}" y="{T+ph*(1-g)+4:.1f}" class="ax" text-anchor="end">{lab}</text>')
    step = max(1, n // 10); last = None
    for i in range(0, n, step):
        lab = day(bins[i]["t_start"])
        if lab == last: continue
        last = lab; out.append(f'<text x="{L+i*bw+bw/2:.1f}" y="{T+ph+16}" class="ax" text-anchor="middle">{lab}</text>')
    out.append(f'<text x="{L+pw/2:.0f}" y="{H-6}" class="ax" text-anchor="middle">시간순 60등분 (각 칸 = 전체 행의 1/60, 칸 시작일 표시)</text></svg>')
    leg = ''.join(f'<span class="lg"><i class="sw f-{cm[c]}"></i>{esc(c)} <b>{pct(shares[c], 2 if shares[c] < 0.01 else 1)}</b></span>' for c in order if cm[c] != 'other')
    other = [c for c in order if cm[c] == 'other']
    if other: leg += f'<span class="lg"><i class="sw f-other"></i>기타: {esc(", ".join(other))} <b>{pct(sum(shares[c] for c in other), 2)}</b></span>'
    return ''.join(out), leg

def svg_timeline(name):
    d = tdata[name]; tl = d['timeline']; classes = sorted(tl, key=lambda c: tl[c]['t0'])
    t0 = min(v['t0'] for v in tl.values()); t1 = max(v['t1'] for v in tl.values()); span = max(t1 - t0, 1)
    W, L, R, T, rh = 960, 200, 12, 10, 24; pw = W - L - R; H = T + rh * len(classes) + 38; X = lambda t: L + (t - t0) / span * pw
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{esc(name)} per-class time span with chronological split">']
    for i, c in enumerate(classes):
        v = tl[c]; y = T + i * rh
        out.append(f'<text x="{L-8}" y="{y+16}" class="lab" text-anchor="end">{esc(c)} <tspan class="ax">{fmt(v["rows"])}</tspan></text><line x1="{L}" x2="{L+pw}" y1="{y+rh-1}" y2="{y+rh-1}" class="grid"/>')
        for a, b, k, lab in [(v['t0'], v['t_train_end'], 'seg-train', 'train 60%'), (v['t_train_end'], v['t_val_end'], 'seg-val', 'val 20%'), (v['t_val_end'], v['t1'], 'seg-test', 'test 20%')]:
            x0, x1 = X(a), X(b); out.append(f'<rect x="{x0:.1f}" y="{y+5}" width="{max(x1-x0,2.5):.1f}" height="{rh-11}" class="{k}"><title>{esc(c)} {lab}: {daytime(a)} → {daytime(b)}</title></rect>')
    ticks = np.linspace(t0, t1, 7)
    for j, t in enumerate(ticks): out.append(f'<text x="{X(t):.1f}" y="{H-14}" class="ax" text-anchor="{"start" if j == 0 else ("end" if j == len(ticks)-1 else "middle")}">{day(t)}</text>')
    out.append(f'<text x="{L+pw/2:.0f}" y="{H-2}" class="ax" text-anchor="middle">클래스별 시간 범위와 family별 시간순 분할 (범례는 아래)</text></svg>')
    return ''.join(out) + '<div class="legend"><span class="lg"><i class="sw seg-train"></i>train (앞 60%)</span><span class="lg"><i class="sw seg-val"></i>val (20%)</span><span class="lg"><i class="sw seg-test"></i>test (뒤 20%)</span></div>'

def svg_oracle(name):
    """달성 F1 / 회수 가능한 여지 / 모순이 막는 구간, 막대의 끝은 row-level oracle.

    row-level oracle = 행마다 따로 채점해 맞힐 수 있으면 맞힌 것으로 세는 예전 정의. CIC-IDS2018은
    EXP39 expert bank에서 실제로 잰 값이고, bank가 없는 나머지 데이터셋은 1.0이다(행 단위로는 언제나
    맞힐 수 있으므로). 그 격차 중 실제로 닿을 수 있는 곳은 두 제약이 함께 정한다 -- 같은 벡터에는 같은
    답을 내야 하고(현실적 oracle), train이 그 벡터에 붙인 라벨을 따를 수밖에 없다(train 일관 천장).
    둘 중 낮은 쪽이 회수 가능한 끝이고, 그 위는 전부 모순이 먹는다."""
    ach, src = achieved(name); c = v2[v2.dataset == name].set_index('cls'); classes = [k for k in c.index if k in ach]
    row_lvl = (lambda k: float(row_oracle[k])) if name == 'cse_cic_ids2018' else (lambda k: 1.0)
    reach = lambda k: min(orc(name, k), train_ceiling(name, k))
    classes = sorted(classes, key=lambda k: row_lvl(k) - float(ach[k]), reverse=True)
    W, L, R, T, rh = 960, 200, 176, 10, 26; pw = W - L - R; H = T + rh * len(classes) + 34; X = lambda v: L + v * pw
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{esc(name)} achieved vs row-level oracle">']
    for g in (0.25, 0.5, 0.75, 1.0): out.append(f'<line x1="{X(g):.1f}" x2="{X(g):.1f}" y1="{T}" y2="{T+rh*len(classes)}" class="grid"/>')
    for i, k in enumerate(classes):
        y = T + i * rh; a = float(ach[k]); r = max(row_lvl(k), a); rc = min(max(reach(k), a), r)
        out.append(f'<text x="{L-8}" y="{y+17}" class="lab" text-anchor="end">{esc(k)}</text>')
        out.append(f'<rect x="{X(0):.1f}" y="{y+7}" width="{max(X(a)-X(0),1):.1f}" height="12" class="hb-ach"><title>{esc(k)} 달성 F1 {a:.3f}</title></rect>')
        if rc > a: out.append(f'<rect x="{X(a):.1f}" y="{y+7}" width="{X(rc)-X(a):.1f}" height="12" class="hb-room"><title>{esc(k)} 회수 가능 {a:.3f}&rarr;{rc:.3f}</title></rect>')
        if r > rc: out.append(f'<rect x="{X(rc):.1f}" y="{y+7}" width="{X(r)-X(rc):.1f}" height="12" class="f-s2"><title>{esc(k)} 모순이 막는 구간 {rc:.3f}&rarr;{r:.3f}: 같은 벡터에 같은 답을 내야 하고(현실적 oracle {orc(name,k):.3f}) train이 붙인 라벨을 따라야 한다(train 일관 천장 {train_ceiling(name,k):.3f})</title></rect>')
        out.append(f'<text x="{X(1)+8}" y="{y+17}" class="ax">{a:.3f} &rarr; {rc:.3f} &rarr; {r:.3f}</text>')
    for g, l in ((0, '0'), (0.5, '0.5'), (1, '1.0')): out.append(f'<text x="{X(g):.1f}" y="{T+rh*len(classes)+14}" class="ax" text-anchor="middle">{l}</text>')
    out.append(f'<text x="{L+pw/2:.0f}" y="{H-2}" class="ax" text-anchor="middle">달성 F1 &rarr; 회수 가능한 끝 &rarr; row-level oracle</text></svg>')
    leg = ('<div class="legend"><span class="lg"><i class="sw hb-ach"></i>달성 F1</span>'
           '<span class="lg"><i class="sw hb-room"></i>회수 가능한 여지</span>'
           '<span class="lg"><i class="sw f-s2"></i>모순이 막는 구간</span></div>')
    rl = ('막대의 끝 = row-level oracle, EXP39 expert bank에서 행마다 채점해 잰 값이다(0821&middot;0902에서 "회수 가능"이라고 부르던 바로 그 수치).'
          if name == 'cse_cic_ids2018' else
          '막대의 끝 = row-level oracle. 이 데이터셋은 expert bank가 없으므로 행마다 따로 채점하면 언제나 맞힐 수 있어 1.0이다.')
    ton = (f' 예를 들어 scanning은 test 안에서는 충돌이 3%뿐이지만 같은 벡터가 train 기간에 benign으로 라벨돼 있어 회수 가능한 끝이 {reach("scanning"):.3f}에 그친다.'
           if name == 'ton_iot' else '')
    return ''.join(out) + leg + (f'<div class="figcap">달성 F1 출처: {esc(src)}. {rl} 회수 가능한 끝은 두 제약 중 낮은 쪽이다 &mdash; '
                                 f'같은 벡터에는 같은 답을 내야 하고(현실적 oracle), train이 그 벡터에 붙인 라벨을 따를 수밖에 없다(train 일관 천장 2(1&minus;p)/(2&minus;p), '
                                 f'p = 그 클래스 test 행 중 train 라벨이 다른 비율). 그 위는 전부 모순이 먹는 몫이다.{ton}</div>')


# ---------------- tables ----------------
def pair_table(name, k=8):
    P = pairs[name]; rows = []
    for q in P['pairs'][:k]:
        sa = ' · '.join(f'{esc(x)} {fmt(v)}' for x, v in q['scen_a'].items()); sb = ' · '.join(f'{esc(x)} {fmt(v)}' for x, v in q['scen_b'].items())
        rows.append(f'<tr><td class="t"><b>{esc(q["a"])}</b> / <b>{esc(q["b"])}</b><div class="sub">시나리오: {sa} / {sb}</div><div class="sub">기간: {esc(q["span_a"])} / {esc(q["span_b"])}</div></td><td align="right">{fmt(q["rows_a"])} <span class="sub">({pct(q["frac_a"])})</span></td><td align="right">{fmt(q["rows_b"])} <span class="sub">({pct(q["frac_b"])})</span></td><td align="right">{fmt(q["vectors"])}</td></tr>')
    return f'<div class="wrap"><table><tr><th>같은 벡터에 붙은 라벨 A / B</th><th>A 행 수 (A 전체 대비)</th><th>B 행 수 (B 전체 대비)</th><th>벡터 수</th></tr>{"".join(rows)}</table></div>'

def oracle_table(name):
    ach, _ = achieved(name); c = v2[v2.dataset == name].set_index('cls'); rows = []
    for k in sorted(c.index, key=lambda k: -int(c.loc[k, 'rows_all'])):
        a = float(ach.get(k, np.nan)); nt = int(c.loc[k, 'rows_test']); ro = f'{float(row_oracle[k]):.3f}' if name == 'cse_cic_ids2018' else '—'
        rows.append(f'<tr><td class="t">{esc(k)}</td><td align="right">{fmt(nt)}</td><td align="right">{a:.3f}</td><td align="right">{ro}</td><td align="right"><b>{orc(name,k):.3f}</b></td><td align="right">{fmt(round(tic(name,k)*nt))} <span class="sub">({pct(tic(name,k))})</span></td><td align="right">{fmt(round(trc(name,k)*nt))} <span class="sub">({pct(trc(name,k))})</span></td></tr>')
    return f'<div class="wrap"><table><tr><th>클래스</th><th>test 행</th><th>달성 F1</th><th>row-level oracle</th><th>{orc_label(name)}</th><th>test 내부 충돌 행</th><th>train 라벨과 충돌 행</th></tr>{"".join(rows)}</table></div>'

def lookup_table():
    rows = []
    for key, short, _ in DS:
        d = tdata[key]; r, c = d['lookup_random'], d['lookup_chrono']; base = max(v['rows'] for v in d['timeline'].values()) / d['rows']
        rows.append(f'<tr><td class="t">{short}</td><td align="right">{pct(base)}</td><td align="right">{pct(r["seen_frac"])}</td><td align="right">{pct(r["acc_on_seen"])}</td><td align="right"><b>{pct(r["acc"])}</b></td><td align="right">{r["macro_f1"]:.3f}</td><td align="right">{pct(c["seen_frac"])}</td><td align="right">{pct(c["acc"])}</td><td align="right">{c["macro_f1"]:.3f}</td></tr>')
    return f'<div class="wrap"><table><tr><th>데이터셋</th><th>최빈 클래스 비율</th><th>랜덤: seen</th><th>랜덤: seen 정확도</th><th>랜덤: 정확도</th><th>랜덤: macro</th><th>시간순: seen</th><th>시간순: 정확도</th><th>시간순: macro</th></tr>{"".join(rows)}</table></div>'

# ---------------- EXP41/42: dataset-level dedup benchmarks + the method comparison ----------------
ARMS = [('original', '원본'), ('relabel_majority', '중복 유지 + 모순은 다수 라벨'), ('drop_vectors', '중복 유지 + 모순 벡터 삭제'),
        ('dedup_xy', '중복 제거<div class="sub">모순은 유지</div>'),
        ('dedup_majority', '중복 제거 + 모순은 다수 라벨'), ('drop_conflict', '중복 제거 + 모순 벡터 삭제')]
CLS7 = ['benign', 'bot', 'brute_force', 'ddos', 'dos', 'infiltration', 'web_attacks']
CLS7H = ['benign', 'bot', 'brute', 'ddos', 'dos', 'inf', 'web']
METHODS = [('realistic_oracle', '현실적 oracle', '완벽 라우팅 + 충돌은 다수 라벨 = 이 test의 천장'),
           ('xgboost_full', 'XGBoost', 'train 풀 전체 (300 trees · depth 8)'),
           ('xgboost_c0', 'XGBoost', '같은 C0 100k행 (정보 예산 짝지음)'),
           ('tabpfn_plain', 'TabPFN-v3', '무작위 100k 컨텍스트 (비율 보존)'),
           ('ours_c0', 'ours', '구성된 C0 100k (benign 0.75 · 공격 균등 · n_est 4)'),
           ('distpfn', 'DistPFN', 'ours(n_est 4) 확률에 prior 보정'),
           ('boostpfn', 'BoostPFN', 'TabPFN-v1 약학습기 T=50 × 500행'),
           ('localpfn_knn', 'LoCalPFN', 'TabPFN-kNN: 행마다 국소 컨텍스트 1,000행')]


def exp42_load():
    """Merge every completed EXP42 run (one per method group) into one record per arm."""
    out = {}
    for d in sorted(glob.glob(f'{ROOT}/tabpfn/results/*_exp42_dedup_methods_*') + glob.glob(f'{ROOT}/tabpfn/results/*_exp44_conflict_only_*')):
        if not os.path.exists(f'{d}/COMPLETE.json'): continue
        a = json.load(open(f'{d}/args.json'))
        if a.get('test_cap_per_class'): continue  # smoke runs
        tm = json.load(open(f'{d}/timings.json'))[0]; arm = a.get('dataset_dedup') or a.get('conflict_only')
        e = out.setdefault(arm, dict(pc=[], sm=[], counts=None, test_rows=tm['test_rows'], train_rows=tm['train_rows'], seed=a['seed'], secs={}))
        e['pc'].append(pd.read_csv(f'{d}/per_class_metrics.csv')); e['sm'].append(pd.read_csv(f'{d}/1b_summary.csv'))
        for k, v in tm.items():
            if k.endswith('_seconds') and isinstance(v, (int, float)): e['secs'][k] = v
        if e['counts'] is None and os.path.exists(f'{d}/0c_class_counts.csv'): e['counts'] = pd.read_csv(f'{d}/0c_class_counts.csv')
    for e in out.values():
        e['pc'] = pd.concat(e['pc']).drop_duplicates(['method', 'bucket', 'cls'], keep='last')
        e['sm'] = pd.concat(e['sm']).drop_duplicates(['method', 'bucket'], keep='last')
    return out


def counts_table(E):
    if not any(e.get('counts') is not None for e in E.values()): return ''
    head = ''.join(f'<th colspan="2" align="right">{lab.split("<")[0]}</th>' for _, lab in ARMS)
    sub = ''.join('<th align="right">train</th><th align="right">test</th>' for _ in ARMS)
    body = []
    for c in CLS7 + ['합계']:
        tds = []
        for arm, _ in ARMS:
            e = E.get(arm)
            if e is None or e['counts'] is None: tds.append('<td align="right">—</td><td align="right">—</td>'); continue
            cc = e['counts']
            for split in ('train', 'test'):
                sel = cc[cc.split == split] if c == '합계' else cc[(cc.cls == c) & (cc.split == split)]
                n = int(sel.rows_arm.sum()); base = int(sel.rows_pipeline.sum())
                keep = '' if arm == 'original' else f'<div class="sub">{100 * n / base:.1f}%</div>' if base else '<div class="sub">—</div>'
                tds.append(f'<td align="right">{fmt(n)}{keep}</td>')
        nm = '<b>합계</b>' if c == '합계' else esc(c)
        body.append(f'<tr><td class="t">{nm}</td>{"".join(tds)}</tr>')
    return (f'<div class="wrap"><table><tr><th rowspan="2">클래스</th>{head}</tr><tr>{sub}</tr>{"".join(body)}</table></div>'
            '<p class="ref">각 처리를 전 행에 적용한 뒤 같은 scenario별 시간순 60/20/20 규칙으로 다시 나눈 결과다(행이 5개 미만인 시나리오는 train 전용). '
            '작은 숫자는 파이프라인 원본 split의 같은 칸 대비 남은 비율이다. val(20%)은 표에서 생략했다. 원본 arm의 test는 재구현 split이라 파이프라인과 4행 차이가 난다.</p>')


def method_tables(E):
    parts = []
    for arm, label in ARMS:
        e = E.get(arm)
        if e is None: continue
        rows, missing = [], []
        for m, name, note in METHODS:
            pc = e['pc'][(e['pc'].method == m) & (e['pc'].bucket == 'all')]
            if pc.empty: missing.append(name); continue
            sm = e['sm'][(e['sm'].method == m) & (e['sm'].bucket == 'all')].iloc[0]
            f1 = {r.cls: r.f1 for r in pc.itertuples()}
            cls_tds = ''.join(f'<td align="right">{f1[c]:.3f}</td>' for c in CLS7)
            tag = ' class="ref"' if m == 'realistic_oracle' else ''
            rows.append(f'<tr{tag}><td class="t">{name}<div class="sub">{note}</div></td>'
                        f'<td align="right"><b>{sm.macro_f1:.4f}</b></td><td align="right">{sm.tail_f1:.4f}</td>{cls_tds}</tr>')
        if not rows: continue
        miss = f' <span class="sub">— {", ".join(missing)} 미실행(EXP44는 LoCalPFN 제외: arm당 ~3.4 h)</span>' if missing else ''
        parts.append(f'<h4>{label.split("<")[0]} — test {fmt(e["test_rows"])}행 · train {fmt(e["train_rows"])}행 · seed {e["seed"]}{miss}</h4>'
                     f'<div class="wrap"><table><tr><th>방법</th><th align="right">macro-F1</th><th align="right">tail F1</th>'
                     + ''.join(f'<th align="right">{h}</th>' for h in CLS7H) + f'</tr>{"".join(rows)}</table></div>')
    return ''.join(parts)


def macro_matrix(E):
    def val(arm, m):
        e = E.get(arm)
        if e is None: return None
        x = e['sm'][(e['sm'].method == m) & (e['sm'].bucket == 'all')]
        return None if x.empty else float(x.macro_f1.iloc[0])
    best = {}
    for arm, _lab in ARMS:
        cand = [(val(arm, m), m) for m, _n, _o in METHODS if m != 'realistic_oracle' and val(arm, m) is not None]
        if cand: best[arm] = max(cand)[1]
    rows = []
    for m, name, note in METHODS:
        tds = []
        for arm, _lab in ARMS:
            v = val(arm, m)
            if v is None: tds.append('<td align="right">—</td>')
            elif best.get(arm) == m: tds.append(f'<td align="right"><b>{v:.4f}</b></td>')
            else: tds.append(f'<td align="right">{v:.4f}</td>')
        if all('—' in t for t in tds): continue
        rows.append(f'<tr><td class="t">{name}<div class="sub">{note}</div></td>{"".join(tds)}</tr>')
    if not rows: return ''
    head = ''.join(f'<th align="right">{lab.split("<")[0]}</th>' for _, lab in ARMS)
    return (f'<div class="wrap"><table><tr><th>방법 (macro-F1)</th>{head}</tr>{"".join(rows)}</table></div>'
            '<p class="ref">열마다 test가 다르므로 가로 비교는 "그 벤치마크에서의 점수"다. 같은 열 안의 세로 비교만 방법 간 비교다. '
            '각 열에서 oracle을 제외한 최고값을 굵게 표시했다. seed 42, XGBoost 300 trees · depth 8. 컨텍스트를 쓰는 다섯 방법은 모두 같은 크기(100k행)를 받는다 — XGBoost C0 · ours · BoostPFN · LoCalPFN은 구성된 C0를, TabPFN-v3는 같은 풀에서 비율을 보존해 무작위로 뽑은 100k를 받는다. 그 차이가 컨텍스트 구성의 몫이다. DistPFN은 ours의 확률에 사후 prior 보정을 건 것인데, exp37 기록(원본 벤치마크)에서는 n_estimators 4·8·16·32 모두 중립~상승(0.7617→0.7609, 0.7635→0.7633, 0.7631→0.7691, 0.7693→0.7692)이었던 것과 달리 EXP42 재구현에서는 원본 arm 0.6681로 내려간다 — n_estimators로는 설명되지 않으며 exp37 플러그인과 EXP42 재구현의 차이는 미해결이다(0915 정정). C0가 꼬리를 일부러 과대표집한 컨텍스트라 &quot;prior shift 보정&quot;이 그 구성을 되돌리는 방향으로 작동하는 것은 두 구현에 공통이다.</p>')


def exp41_seed_table():
    recs = []
    for d in sorted(glob.glob(f'{ROOT}/tabpfn/results/*_exp41_dedup_dataset_tabpfn_*')):
        if not os.path.exists(f'{d}/1b_summary.csv'): continue
        sm = pd.read_csv(f'{d}/1b_summary.csv'); a = json.load(open(f'{d}/args.json'))
        x = sm[(sm.method == 'tabpfn_v3') & (sm.bucket == 'all')]; xc = sm[(sm.method == 'xgboost_c0') & (sm.bucket == 'all')]
        if len(x): recs.append(dict(arm=a['dataset_dedup'], seed=a['seed'], macro=float(x.macro_f1.iloc[0]), xgbc0=float(xc.macro_f1.iloc[0]) if len(xc) else np.nan))
    if not recs: return ''
    df = pd.DataFrame(recs); rows = []; wins_all = []
    for arm, label in ARMS:
        g = df[df.arm == arm]
        if g.empty: continue
        diffs = [(float(g[g.seed == s].macro.iloc[0]) - float(g[g.seed == s].xgbc0.iloc[0])) if len(g[g.seed == s]) else np.nan for s in (42, 43, 44)]
        if all(d > 0 for d in diffs if not np.isnan(d)): wins_all.append(label.split("<")[0])
        dtxt = ' / '.join('—' if np.isnan(d) else f'{d:+.4f}' for d in diffs)
        rows.append(f'<tr><td class="t">{label.split("<")[0]}</td>' + ''.join(
            f'<td align="right">{float(g[g.seed == s].macro.iloc[0]):.4f}</td>' if len(g[g.seed == s]) else '<td align="right">—</td>' for s in (42, 43, 44))
            + f'<td align="right"><b>{g.macro.mean():.4f}</b></td><td align="right">{g.macro.max() - g.macro.min():.4f}</td>'
            + f'<td align="right">{g.xgbc0.mean():.4f}</td><td align="right">{dtxt}</td></tr>')
    return (f'<h4>ours의 seed 변동 — 같은 처리, C0만 다시 추첨 (EXP41), 그리고 같은 C0를 준 XGBoost와의 차이</h4><div class="wrap"><table>'
            '<tr><th>데이터</th><th align="right">ours seed 42</th><th align="right">seed 43</th><th align="right">seed 44</th><th align="right">평균</th><th align="right">폭</th><th align="right">XGB-c0 평균</th><th align="right">ours − XGB-c0 (42 / 43 / 44)</th></tr>'
            f'{"".join(rows)}</table></div><p class="ref">100k C0는 seed에 따라 macro가 0.675–0.765를 오간 기록이 있어(0902 §8ag) 처리별 차이를 읽기 전에 seed 폭을 먼저 본다. '
            '위 방법 비교표는 seed 42 한 번이므로, 처리 간 차이가 이 폭보다 작으면 차이로 읽지 않는다. '
            f'같은 C0를 준 XGBoost와 비교하면 ours가 세 seed 모두 앞서는 arm은 {" · ".join(wins_all) if wins_all else "없음"}뿐이고, 중복 제거 세 arm에서는 seed 42만 ours가 앞선다 — '
            '위 표의 &quot;ours 4/4 최고&quot;는 seed 42 한 번의 결과이며, dedup arm에서 XGB-c0 대비 우위는 seed 폭 안이라 차이로 읽지 않는다(0915 재검). 중복 유지 + 모순 제거 두 arm(EXP44)은 seed 42 단일이라 폭이 없다.</p>')


def exp44_verdict(E):
    def v(arm, m):
        e = E.get(arm)
        if e is None: return None
        x = e['sm'][(e['sm'].method == m) & (e['sm'].bucket == 'all')]
        return None if x.empty else float(x.macro_f1.iloc[0])
    def trows(arm, c):
        e = E.get(arm)
        if e is None or e['counts'] is None: return None
        cc = e['counts']; sel = cc[(cc.cls == c) & (cc.split == 'test')]
        return int(sel.rows_arm.sum()), int(sel.rows_pipeline.sum())
    if v('relabel_majority', 'xgboost_full') is None or v('drop_vectors', 'xgboost_full') is None: return ''
    w = trows('relabel_majority', 'web_attacks'); i = trows('relabel_majority', 'infiltration')
    return (f'<h4>중복은 두고 모순만 없애면 &mdash; EXP44의 판정</h4>'
            f'<p>재라벨 arm에서는 train 풀 전체를 쓴 XGBoost({v("relabel_majority","xgboost_full"):.4f})가 ours({v("relabel_majority","ours_c0"):.4f})와 무작위 컨텍스트 TabPFN-v3({v("relabel_majority","tabpfn_plain"):.4f})를 앞선다. 원본에서는 ours가 앞섰으므로 <strong>모순을 없애면 순위가 뒤집힌다</strong> &mdash; 12M행을 통째로 외우는 쪽이 유리해지고 컨텍스트 구성의 이점이 줄어든다. 벡터 삭제 arm에서는 XGBoost {v("drop_vectors","xgboost_full"):.4f} &middot; ours {v("drop_vectors","ours_c0"):.4f} &middot; XGB-c0 {v("drop_vectors","xgboost_c0"):.4f}가 사실상 동률이다. 즉 <strong>ours의 우위는 모순이 남아 있을 때만 나타난다</strong> &mdash; 우리 주장에 불리하지만 이것이 데이터가 말하는 바다. '
            + (f'그리고 재라벨은 행을 하나도 지우지 않는데도 test web {fmt(w[1])}&rarr;{fmt(w[0])}행({100*w[0]/w[1]:.1f}%), infiltration {fmt(i[1])}&rarr;{fmt(i[0])}행({100*i[0]/i[1]:.1f}%)이 다수 라벨(benign)로 흡수되고 dos의 slowhttptest는 brute_force로 넘어간다. ' if w and i else '')
            + '두 arm 모두 macro 상승분의 상당 부분은 모델이 좋아져서가 아니라 어려운 행이 사라지거나 쉬운 클래스로 옮겨간 몫이다. 다수 라벨은 train+test 전체에서 계산하므로 재라벨 arm과 중복 제거+다수 라벨 arm은 모두 Arp et al.(USENIX Sec\'22)의 selective snooping에 해당하는 진단용 벤치마크이지 배포 가능한 전처리가 아니다. <strong>결론: 모순은 지워도 고쳐도 꼬리를 잃는다.</strong> headline은 원본 벤치마크로 두고, 모순은 §1의 3분해와 클래스별 현실적 oracle을 옆에 두는 방식으로 보고한다(§7). 데이터 감사 실험은 여기서 마친다(09-15).</p>')


def exp42_section():
    E = exp42_load()
    if not E: return '<p class="ref">EXP42 실행 중 — 완료되면 이 자리에 표가 들어간다.</p>'
    return ('<h4>처리가 train·test를 클래스별로 얼마나 바꾸는가</h4>' + counts_table(E)
            + '<h4>그래서 결과는 어떻게 됐는가 — 방법 × 처리</h4>' + macro_matrix(E) + exp44_verdict(E)
            + method_tables(E) + exp41_seed_table())


# ---------------- §1 charts: 모순 / 중복 / 누수 ----------------
def _classes_by_rows(name):
    c = cls[cls.dataset == name].sort_values('rows', ascending=False)
    return [str(x) for x in c.cls]


def svg_conflict_matrix(name):
    """Class x class: what share of class i's rows share a vector with class j."""
    P = pairs[name]; cr = P['class_rows']
    classes = [c for c in _classes_by_rows(name) if c in cr]
    n = len(classes); idx = {c: i for i, c in enumerate(classes)}
    frac = [[0.0] * n for _ in range(n)]; cnt = [[0] * n for _ in range(n)]
    for q in P['pairs']:
        a, b = q['a'], q['b']
        if a not in idx or b not in idx: continue
        frac[idx[a]][idx[b]] = q['frac_a']; cnt[idx[a]][idx[b]] = q['rows_a']
        frac[idx[b]][idx[a]] = q['frac_b']; cnt[idx[b]][idx[a]] = q['rows_b']
    L, T, cw, ch, R = 132, 104, min(78, int(760 / max(n, 1))), 28, 16
    W, H = L + cw * n + R, T + ch * n + 6
    out = [f'<svg viewBox="-58 0 {W+58} {H}" width="100%" role="img" aria-label="{esc(name)} label-sharing matrix">']
    for j, c in enumerate(classes):
        x = L + j * cw + cw / 2
        out.append(f'<text x="{x:.1f}" y="{T-18}" class="ax" text-anchor="end" transform="rotate(-38 {x:.1f} {T-18})">{esc(c)}</text>')
    for i, ci in enumerate(classes):
        y = T + i * ch
        out.append(f'<text x="{L-8}" y="{y+18}" class="lab" text-anchor="end">{esc(ci)}</text>')
        for j, cj in enumerate(classes):
            x = L + j * cw
            if i == j:
                out.append(f'<rect x="{x+1}" y="{y+1}" width="{cw-2}" height="{ch-2}" fill="var(--bg-2)"/>')
                out.append(f'<text x="{x+cw/2:.1f}" y="{y+18}" class="ax" text-anchor="middle">&#183;</text>'); continue
            v = frac[i][j]; op = 0 if v <= 0 else 0.10 + 0.62 * (v ** 0.5)
            out.append(f'<rect x="{x+1}" y="{y+1}" width="{cw-2}" height="{ch-2}" fill="var(--warn)" fill-opacity="{op:.3f}"/>'
                       f'<title>{esc(ci)} 행 {fmt(cnt[i][j])}개가 {esc(cj)}와 같은 벡터를 공유 ({pct(v,2)} of {esc(ci)})</title>')
            if v > 0:
                lab = f'{v*100:.1f}' if v >= 0.001 else '&lt;0.1'
                out.append(f'<text x="{x+cw/2:.1f}" y="{y+18}" class="ax" text-anchor="middle" fill="var(--ink)">{lab}</text>')
    out.append('</svg>')
    return ''.join(out) + '<div class="figcap">세로 = 오염된 클래스, 가로 = 상대 클래스. 값 = 세로 클래스의 행 중 몇 %가 가로 클래스와 같은 벡터를 공유하는가. 대각선은 비워 둔다.</div>'


def svg_dup_bar(name):
    c = cls[cls.dataset == name].sort_values('rows', ascending=False)
    rows = [(str(r.cls), int(r.rows), int(r.unique_vectors)) for r in c.itertuples()]
    W, L, R, T, rh = 960, 132, 210, 8, 26; pw = W - L - R; H = T + rh * len(rows) + 26
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{esc(name)} unique vector share">']
    for g in (0.25, 0.5, 0.75, 1.0): out.append(f'<line x1="{L+pw*g:.1f}" x2="{L+pw*g:.1f}" y1="{T}" y2="{T+rh*len(rows)}" class="grid"/>')
    for i, (k, n_rows, n_uniq) in enumerate(rows):
        y = T + i * rh; f = n_uniq / n_rows if n_rows else 0
        out.append(f'<text x="{L-8}" y="{y+17}" class="lab" text-anchor="end">{esc(k)}</text>')
        out.append(f'<rect x="{L}" y="{y+7}" width="{pw}" height="12" fill="var(--bg-2)"/>')
        out.append(f'<rect x="{L}" y="{y+7}" width="{max(pw*f,1):.1f}" height="12" class="f-s1"><title>{esc(k)}: 고유 벡터 {fmt(n_uniq)} / 전체 {fmt(n_rows)}행</title></rect>')
        out.append(f'<text x="{L+pw+8}" y="{y+17}" class="ax">{pct(f,1)} · 고유 {fmt(n_uniq)} · 행/벡터 {n_rows/max(n_uniq,1):.1f}배</text>')
    out.append(f'<text x="{L+pw/2:.0f}" y="{H-4}" class="ax" text-anchor="middle">클래스 행 중 고유 벡터가 차지하는 비율 (낮을수록 같은 행이 반복됨)</text></svg>')
    return out and ''.join(out)


def svg_leak_bar(name):
    c = cls[cls.dataset == name].sort_values('rows', ascending=False)
    W, L, R, T, rh = 960, 132, 196, 8, 26; pw = W - L - R
    rows = []
    for r in c.itertuples():
        tr = float(r.chrono_test_rows)
        if tr <= 0: continue
        seen = float(r.chrono_test_seen_in_train); conf = float(r.chrono_test_conflict)
        rows.append((str(r.cls), int(tr), max(seen - conf, 0.0), conf, max(1 - seen, 0.0)))
    H = T + rh * len(rows) + 26
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{esc(name)} test decomposition">']
    for i, (k, tr, same, conf, uns) in enumerate(rows):
        y = T + i * rh; x = L
        out.append(f'<text x="{L-8}" y="{y+17}" class="lab" text-anchor="end">{esc(k)}</text>')
        for frac_, klass, lab in ((same, 'f-neutral', '같은 라벨로 train에 있던 벡터'), (conf, 'f-s2', '다른 라벨로 train에 있던 벡터'), (uns, 'f-s1', '처음 보는 벡터')):
            if frac_ <= 0: continue
            w = pw * frac_
            out.append(f'<rect x="{x:.1f}" y="{y+7}" width="{max(w,0.6):.1f}" height="12" class="{klass}"><title>{esc(k)} {esc(lab)} {pct(frac_,1)} ({fmt(tr*frac_)}행)</title></rect>'); x += w
        out.append(f'<text x="{L+pw+8}" y="{y+17}" class="ax">test {fmt(tr)}행 · 모순 {pct(conf,1)}</text>')
    out.append(f'<text x="{L+pw/2:.0f}" y="{H-4}" class="ax" text-anchor="middle">시간순 test 행의 3분해</text></svg>')
    leg = ('<div class="legend"><span class="lg"><i class="sw f-neutral"></i>같은 라벨로 train에 있던 벡터 (암기)</span>'
           '<span class="lg"><i class="sw f-s2"></i>다른 라벨로 train에 있던 벡터 (모순)</span>'
           '<span class="lg"><i class="sw f-s1"></i>처음 보는 벡터 (일반화)</span></div>')
    return ''.join(out) + leg


def sec_conflict(d):
    key, short = d['key'], d['short']; s, t = d['s'], d['t']
    return (f'<h3 id="p-{key}">{esc(short)} <span class="pill">{esc(d["full"])}</span> '
            f'<span class="pill">{fmt(s.rows)}행 · 고유 벡터 {fmt(s.unique_vectors)} · 모순 그룹 {fmt(t["mixed_groups"])}개 · 모순 벡터에 속한 행 {pct(s.frac_mixed)}</span></h3>'
            f'<figure><div class="chart">{svg_conflict_matrix(key)}</div></figure>{pair_table(key, 4)}')


def sec_dup(d):
    return f'<h3>{esc(d["short"])}</h3><figure><div class="chart">{svg_dup_bar(d["key"])}</div></figure>'


def sec_leak(d):
    return f'<h3>{esc(d["short"])}</h3><figure><div class="chart">{svg_leak_bar(d["key"])}</div></figure>'


def lit_section():
    A = [('Tavallaee, Bagheri, Lu, Ghorbani &mdash; A Detailed Analysis of the KDD CUP 99 Data Set', 'IEEE CISDA 2009', 'https://www.ee.torontomu.ca/~bagheri/papers/cisda.pdf', '<b>중복</b>을 지워 NSL-KDD를 만듦. train 4,898,431 &rarr; 1,074,992 (78.05% 감소), test 311,027 &rarr; 77,289.'),
         ('Raskovalov, Gabdullin, Dolmatov &mdash; Investigation and rectification of NIDS datasets&hellip;', 'arXiv 2212.13994 (2022)', 'https://arxiv.org/abs/2212.13994', '<b>모순</b>을 지움. "a number of flows are duplicated and labeled as corresponding to different attacks". ToN-IoT 21,978,630 &rarr; 18,902,360, <b>mitm 1,052 &rarr; 0</b>.'),
         ('Chen, Tran, Thumati, Bhuyan, Ding &mdash; Data Curation and Quality Assurance for ML-based Cyber Intrusion Detection', 'arXiv 2105.10041 (2021)', 'https://arxiv.org/abs/2105.10041', '두 클래스에 동시에 존재하는 시퀀스를 제거(HIDS). "중복 자체는 결함이 아니고 train/test 겹침이 결함"이라는 논지의 출처.')]
    B = [('Jerabek, Luxemburk, Plny, Koumar, Pesek, Hynek &mdash; When Simple Model Just Works: Is Network Traffic Classification in Crisis?', 'arXiv 2506.08655 (2025)', 'https://arxiv.org/abs/2506.08655', '<b>이 보고서의 현실적 oracle과 같은 개념의 선행연구.</b> "over 50% redundant samples&hellip; frequently appear in both training and test sets"; 중복이 "reduce the theoretical maximum accuracy when identical flows have conflicting labels".'),
         ('Xie, Li, Zhang, Sun, Xu &mdash; Analysis and Detection against Network Attacks in the Overlapping Phenomenon of Behavior Attribute', 'Computers &amp; Security 123 (2022)', 'https://arxiv.org/abs/2310.10660', '모순을 삭제 대신 <b>multi-label</b>로 재정의. UNSW-NB15에서 라벨 조합 57종, 샘플당 평균 라벨 1.689개.'),
         ('Flood, Engelen, Aspinall, Desmet &mdash; Bad Design Smells in Benchmark NIDS Datasets', 'IEEE EuroS&amp;P 2024 (distinguished paper)', 'https://distrinet.cs.kuleuven.be/news/bad_smells_euros_p_template-4.pdf', '지우지 않고 지표화. wrong-label CIC-17 infiltration 0.81 / CIC-18 0.65, traffic collapse ToN-IoT backdoor 1.00 / CIC-18 bot 0.99. 권고는 삭제가 아니라 "avoiding using training and test attack data from the same class and dataset".'),
         ('Al-Daweri 외 &mdash; An Analysis of the KDD99 and UNSW-NB15 Datasets for the IDS', 'Symmetry 12(10):1666 (2020)', 'https://www.mdpi.com/2073-8994/12/10/1666', 'UNSW-NB15 공식 train의 42.24%가 중복, test는 0.00%.'),
         ('Engelen, Rimmer, Joosen &mdash; Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study', 'IEEE SPW (WTMC) 2021', 'https://intrusion-detection.distrinet-research.be/WTMC2021/index.html', '<b>모순의 원인</b>: IP와 시간 창만으로 라벨링해 거절된 연결에도 공격 라벨이 붙음.'),
         ('Pekár, Jozsa &mdash; Evaluating ML-Based Anomaly Detection Across Datasets of Varied Integrity', 'Computer Networks (2024)', 'https://arxiv.org/abs/2401.16843', '<b>반증.</b> CICIDS2017에서 중복 패킷을 지우고 flow를 재생성했으나 Random Forest 성능은 거의 불변.')]
    C = [('Apruzzese, Laskov, Schneider &mdash; SoK: Pragmatic Assessment of Machine Learning for NIDS', 'IEEE EuroS&amp;P 2023', 'https://arxiv.org/abs/2305.00550', '상위 venue 30편 중 "none of the 30 papers considered different preprocessing mechanisms".'),
         ('Goldschmidt, Chudá &mdash; Network Intrusion Datasets: A Survey, Limitations, and Recommendations', 'Computers &amp; Security 156 (2025)', 'https://arxiv.org/abs/2502.06688', '89개 데이터셋. 중복 항목 문제가 "frequently unreported". 정작 자신들의 10단계 제작 권고에는 dedup 단계가 없음.'),
         ('Arp 외 &mdash; Dos and Don\'ts of Machine Learning in Computer Security', 'USENIX Security 2022', 'https://arxiv.org/abs/2010.09470', '<b>우리에게 걸리는 경고.</b> "selective snooping describes the cleansing of data based on information not available in practice".')]
    def block(title, items):
        li = ''.join(f'<li><a href="{u}" target="_blank" rel="noopener">{t}</a> <span class="pill">{v}</span><div class="sub">{n}</div></li>' for t, v, u, n in items)
        return f'<h4>{title}</h4><ul class="lit">{li}</ul>'
    return (block('A. 실제로 행을 지운 연구', A) + block('B. 지우지 않고 측정하거나 재라벨한 연구', B)
            + block('C. 표준 관행이 아니라는 근거', C)
            + '<p class="ref">전체 목록과 확인 수준(본문 확인 / 초록만 / 미확인)은 <code>docs/research/20260911/dataset_quality_audit/literature_links.md</code>에 있다. '
              'NetFlow v2에 대한 중복·모순 감사는 Raskovalov 외의 한 문장이 전부이고, <b>v3에 대해서는 전무하다</b> &mdash; Sarhan 외의 표준 feature set 논문과 Luay 외의 temporal 논문 어디에도 duplicate·redundant·identical·conflict가 나오지 않는다.</p>')


# ---------------- page ----------------
sections = []
for key, short, full in DS:
    comp, leg = svg_composition(key); tl = svg_timeline(key); orc_svg = svg_oracle(key); s = summ.loc[key]; t = twins[key]
    sections.append(dict(key=key, short=short, full=full, comp=comp, leg=leg, tl=tl, orc=orc_svg, s=s, t=t))

def sec_problem(d):
    key, short = d['key'], d['short']; s, t = d['s'], d['t']
    return f'''<h3 id="p-{key}">{esc(short)} <span class="pill">{esc(d["full"])}</span> <span class="pill">{fmt(s.rows)}행 · 고유 벡터 {fmt(s.unique_vectors)} · 상반 라벨 그룹 {fmt(t["mixed_groups"])}개 · 충돌 그룹에 속한 행 {pct(s.frac_mixed)}</span></h3>
<h4>동일 벡터를 나눠 갖는 라벨 조합과 행 수 (전체 행 기준, 상위 8쌍)</h4>{pair_table(key)}'''

def sec_oracle(d):
    key, short = d['key'], d['short']
    return f'''<h3 id="o-{key}">{esc(short)}</h3>{oracle_table(key)}<figure><div class="chart">{d["orc"]}</div></figure>'''


def sec_oracle_novtable(d):
    return f'''<h3 id="o-{d["key"]}">{esc(d["short"])}</h3><figure><div class="chart">{d["orc"]}</div></figure>'''

def sec_time(d):
    return f'''<h3 id="t-{d["key"]}">{esc(d["short"])}</h3><figure><div class="chart">{d["comp"]}</div><div class="legend">{d["leg"]}</div></figure><figure><div class="chart">{d["tl"]}</div></figure>'''

cic_orc_macro = v2macro('cse_cic_ids2018', 'ceil_majority_test'); bl = deriv['bank_limited_hash_consistent_maxf1_per_class']
narr_oracle = {
 'cse_cic_ids2018': f'''<p>infiltration test 행의 {pct(tic('cse_cic_ids2018','infiltration'),0)}, web_attacks의 {pct(tic('cse_cic_ids2018','web_attacks'),0)}, dos의 {pct(tic('cse_cic_ids2018','dos'),0)}(slowhttptest=ftp 단일 벡터)가 test 안에서 다른 라벨의 동일 벡터를 갖는다. row-level oracle은 infiltration을 {row_oracle['infiltration']:.3f}까지 "교정 가능"으로 세지만, 같은 bank에 다수 라벨 제약을 건 현실적 oracle은 {orc('cse_cic_ids2018','infiltration'):.3f}이다. macro로는 달성 {deriv['global_macro']:.3f} → 현실적 oracle {bank_orc['realistic_oracle_bank']['macro']:.3f} → row-level {deriv['row_oracle_macro']:.3f}(bank 없이 완벽 라우팅 + 다수 라벨이면 {bank_orc['realistic_oracle_perfect']['macro']:.3f}). EXP39의 교정 가능 infiltration {fmt(deriv['bank_correctable_infiltration'])}행 중 {fmt(deriv['with_benign_twin'])}행이 benign {fmt(deriv['benign_rows_sharing'])}행과 벡터를 나눠 갖는다(<code>cic_bank_oracle_derivation.json</code>).</p>''',
 'ton_iot': f'''<p>ddos test 행의 {pct(tic('ton_iot','ddos'),0)}, benign의 {pct(tic('ton_iot','benign'),0)}가 test 안에서 서로 벡터를 나눠 갖지만 각 벡터에서 다수인 쪽이 분명해 현실적 oracle은 둘 다 {orc('ton_iot','benign'):.2f}·{orc('ton_iot','ddos'):.2f}다. scanning은 test 안에서는 충돌이 {pct(tic('ton_iot','scanning'),0)}뿐이라 현실적 oracle {orc('ton_iot','scanning'):.2f}인데, 같은 벡터가 train 기간에는 benign이라(train 라벨과 충돌 {pct(trc('ton_iot','scanning'),0)}) train으로 학습한 모델은 F1 {ton_xgb.f1['scanning']:.3f}이다. mitm·ransomware는 충돌이 적고 oracle {orc('ton_iot','mitm'):.2f}·{orc('ton_iot','ransomware'):.2f}인데 달성 {ton_xgb.f1['mitm']:.2f}·{ton_xgb.f1['ransomware']:.2f}(정밀도 {ton_xgb.precision['mitm']:.2f}·{ton_xgb.precision['ransomware']:.2f}) — 데이터가 막는 게 아니라 모델이 못 미친 유일한 꼬리다.</p>''',
 'bot_iot': f'''<p>theft test 행의 {pct(tic('bot_iot','theft'),0)}가 test 안에서도 recon과 벡터를 나누지만 그 벡터에서 theft가 다수라 현실적 oracle은 {orc('bot_iot','theft'):.2f}. 그런데 같은 벡터가 train에서는 recon(5월)이라(train 라벨과 충돌 {pct(trc('bot_iot','theft'),0)}) full-XGB theft F1은 {bot_xgb.f1['theft']:.3f}이다. ddos F1 {bot_xgb.f1['ddos']:.3f}는 충돌이 아니라 ddos test 벡터가 train에 {pct(cnum('bot_iot','ddos','chrono_test_seen_in_train'),1)} 존재해 같은 도구의 dos로 분류되는 wave 내부 변화다. 공격 3종의 oracle은 0.99 이상이라 배울 것이 없다.</p>''',
 'unsw_nb15': f'''<p>충돌은 공격끼리만 있다(generic {pct(tic('unsw_nb15','generic'),0)}, backdoor {pct(tic('unsw_nb15','backdoor'),0)}, reconnaissance {pct(tic('unsw_nb15','reconnaissance'),0)}). benign은 0이지만 MIN_TTL/MAX_TTL 두 feature만으로 benign/attack 정확도 0.9999가 나오는 testbed 상관이라(Flood et al. EuroS&P'24, Sarhan et al.) benign 축은 자명하다. 공격 10종의 달성(XGB {unsw_xgb.F1.mean():.3f})과 현실적 oracle 평균 {v2macro('unsw_nb15','ceil_majority_test'):.3f}의 격차는 고유 벡터 위의 카테고리 혼동(backdoor→exploits 315·generic 268·fuzzers 213·reconnaissance 105행; dos→exploits 504행)이고, Flood et al.은 exploits/shellcode/reconnaissance가 "notionally identical"이라 카테고리 간 일반화 평가에 부적합하다고 본다.</p>''',
}

page = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NF-v3 데이터 품질 감사</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+KR:wght@400;500;700&display=swap">
<style>
:root {{ --bg:#F6F7F5; --bg-2:#EDEFEC; --ink:#1B2229; --muted:#5C6770; --line:#D3D8D4; --acc:#1F6F8B; --acc-soft:#DCEBF0; --warn:#B0413E; --warn-soft:#F3E1E0; --ok:#3C7A4A; --ok-soft:#E1EEE3;
  --serif:"IBM Plex Serif",Georgia,serif; --sans:"IBM Plex Sans","Noto Sans KR","Noto Sans CJK KR",system-ui,sans-serif; --mono:"IBM Plex Mono",Menlo,Consolas,monospace;
  --plot:#FCFCFB; --grid:#E1E0D9; --neutral:#C3C2B7; --other:#9E9C95; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
  --seg-train:#1c5cab; --seg-val:#5598e7; --seg-test:#b7d3f6; --hb-ach:#3a4650; --hb-room:#1baf7a; --hb-mirage:#c9c7bf; --hb-row:#eb6834; color-scheme: light dark; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14181B; --bg-2:#1C2226; --ink:#E6E9E7; --muted:#9AA5AC; --line:#2C3338; --acc:#58B4D0; --acc-soft:#1B3540; --warn:#E0736F; --warn-soft:#3D2322; --ok:#7DBB8A; --ok-soft:#1F3324;
  --plot:#1A1A19; --grid:#2C2C2A; --neutral:#4D4D45; --other:#6B6A63; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  --seg-train:#86b6ef; --seg-val:#3987e5; --seg-test:#184f95; --hb-ach:#c3c2b7; --hb-room:#199e70; --hb-mirage:#383835; --hb-row:#d95926; }} }}
:root[data-theme="dark"] {{ --bg:#14181B; --bg-2:#1C2226; --ink:#E6E9E7; --muted:#9AA5AC; --line:#2C3338; --acc:#58B4D0; --acc-soft:#1B3540; --warn:#E0736F; --warn-soft:#3D2322; --ok:#7DBB8A; --ok-soft:#1F3324;
  --plot:#1A1A19; --grid:#2C2C2A; --neutral:#4D4D45; --other:#6B6A63; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  --seg-train:#86b6ef; --seg-val:#3987e5; --seg-test:#184f95; --hb-ach:#c3c2b7; --hb-room:#199e70; --hb-mirage:#383835; --hb-row:#d95926; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans); font-size:15px; line-height:1.65; }}
main {{ max-width:1040px; margin:0 auto; padding:40px 24px 80px; }}
.eyebrow {{ font-family:var(--mono); font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }}
h1 {{ font-family:var(--serif); font-weight:600; font-size:32px; line-height:1.2; margin:6px 0 0; text-wrap:balance; }} .lede {{ max-width:70ch; color:var(--muted); margin:10px 0 0; }}
h2 {{ font-family:var(--serif); font-weight:600; font-size:22px; margin:44px 0 10px; text-wrap:balance; }} h3 {{ font-family:var(--serif); font-weight:600; font-size:19px; margin:34px 0 8px; }}
h4 {{ font-weight:600; font-size:14px; margin:20px 0 6px; color:var(--muted); letter-spacing:.02em; }} p {{ max-width:72ch; margin:8px 0; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin:22px 0 6px; }}
.tile {{ background:var(--bg-2); border-top:2px solid var(--acc); padding:14px 16px; }} .tile.bad {{ border-top-color:var(--warn); }} .tile.warn {{ border-top-color:#c98500; }}
.tile .k {{ font-family:var(--mono); font-size:12px; color:var(--muted); word-break:keep-all; }} .tile .v {{ font-family:var(--mono); font-size:28px; font-weight:500; line-height:1.1; margin:6px 0 4px; font-variant-numeric:tabular-nums; }} .tile .d {{ font-size:12.5px; color:var(--muted); }}
.wrap {{ overflow-x:auto; margin:8px 0 14px; }} table {{ border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top; }} th {{ font-family:var(--mono); font-weight:500; font-size:11.5px; color:var(--muted); letter-spacing:.03em; white-space:nowrap; }}
td[align="right"], th[align="right"] {{ text-align:right; white-space:nowrap; font-family:var(--mono); }} td.t {{ min-width:12ch; }} .sub {{ color:var(--muted); font-size:12px; }} .vec {{ font-size:12px; line-height:1.9; }}
code {{ font-family:var(--mono); font-size:12px; background:var(--bg-2); padding:1px 5px; border-radius:2px; }}
.pill {{ display:inline-block; font-family:var(--mono); font-size:11px; padding:1px 7px; border-radius:10px; background:var(--bg-2); color:var(--muted); border:1px solid var(--line); vertical-align:middle; font-weight:400; }}
figure {{ margin:10px 0 18px; }} .chart {{ background:var(--plot); border:1px solid var(--line); padding:8px; }} .chart svg {{ display:block; width:100%; height:auto; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 16px; padding:8px 4px 0; font-size:12.5px; }} .lg {{ display:inline-flex; align-items:center; gap:6px; }} .lg b {{ font-family:var(--mono); font-weight:500; color:var(--muted); }} .sw {{ width:12px; height:12px; display:inline-block; border-radius:2px; }}
svg text {{ font-family:var(--mono); font-size:11px; fill:var(--ink); }} svg .ax {{ fill:var(--muted); font-size:10.5px; }} svg .lab {{ font-size:11.5px; }} svg .plot {{ fill:var(--plot); }} svg .grid {{ stroke:var(--grid); stroke-width:1; }}
.f-neutral {{ fill:var(--neutral); background:var(--neutral); }} .f-other {{ fill:var(--other); background:var(--other); }}
.f-s1 {{ fill:var(--s1); background:var(--s1); }} .f-s2 {{ fill:var(--s2); background:var(--s2); }} .f-s3 {{ fill:var(--s3); background:var(--s3); }} .f-s4 {{ fill:var(--s4); background:var(--s4); }} .f-s5 {{ fill:var(--s5); background:var(--s5); }} .f-s6 {{ fill:var(--s6); background:var(--s6); }} .f-s7 {{ fill:var(--s7); background:var(--s7); }} .f-s8 {{ fill:var(--s8); background:var(--s8); }}
.seg-train {{ fill:var(--seg-train); background:var(--seg-train); }} .seg-val {{ fill:var(--seg-val); background:var(--seg-val); }} .seg-test {{ fill:var(--seg-test); background:var(--seg-test); }}
.hb-ach {{ fill:var(--hb-ach); background:var(--hb-ach); }} .hb-room {{ fill:var(--hb-room); background:var(--hb-room); }} .hb-mirage {{ fill:var(--hb-mirage); background:var(--hb-mirage); }} .hb-row {{ fill:var(--hb-row); background:var(--hb-row); }}
.callout {{ border-left:3px solid var(--acc); background:var(--acc-soft); padding:12px 16px; margin:16px 0; max-width:80ch; }} .callout.warn {{ border-left-color:var(--warn); background:var(--warn-soft); }}
ol, ul {{ max-width:76ch; padding-left:22px; }} li {{ margin:5px 0; }} ul.lit {{ list-style:none; padding-left:0; max-width:88ch; }} ul.lit li {{ margin:0 0 10px; padding-left:12px; border-left:2px solid var(--line); }} ul.lit a {{ color:var(--acc); text-decoration:none; font-weight:500; }} ul.lit a:hover {{ text-decoration:underline; }} .ref, .figcap {{ font-size:12.5px; color:var(--muted); }} .figcap {{ padding:6px 4px 0; }}
@media (max-width:720px) {{ h1 {{ font-size:26px; }} main {{ padding:28px 16px 60px; }} }}
</style></head><body><main>
<p class="eyebrow">imbalcic · 데이터 감사 · 2026-09-11 (09-15 개정) · 코드 <code>docs/research/20260911/dataset_quality_audit/</code></p>
<h1>NF-v3 4종 데이터 품질 감사 — 모순·중복·누수</h1>
<p class="lede">전 행(66,935,021 × 46)을 hash로 묶어 (1) <b>모순</b>(한 벡터가 두 라벨)·<b>중복</b>(같은 벡터·라벨의 반복)·<b>누수</b>(test 벡터가 train에 존재)가 클래스별로 어떻게 생겼는지, (2) 그중 모순이 만드는 oracle 천장이 어디인지, (3) 다른 연구는 이 셋을 어떻게 처리했는지, (4) 중복과 모순을 분리해 처리하면 방법별·클래스별로 무엇이 달라지는지, (5) 시간순으로 어떤 클래스가 언제 나타나는지를 정리한다.</p>
<div class="tiles">{''.join(f'<div class="tile {("bad" if d["key"]=="bot_iot" else "warn")}"><div class="k">{d["short"]} · {fmt(d["s"].rows)}행</div><div class="v">{pct(d["s"].frac_mixed)}</div><div class="d">상반 라벨 동일 벡터에 속한 행 · 현실적 oracle macro {(bank_orc["realistic_oracle_bank"]["macro"] if d["key"]=="cse_cic_ids2018" else v2macro(d["key"],"ceil_majority_test")):.3f} · 시간순 test의 {pct(d["s"].chrono_test_seen_in_train,0)}가 train에 그대로 있음</div></div>' for d in sections)}</div>

<h2>1. 문제: 모순 &middot; 중복 &middot; 누수</h2>
<p>세 가지를 구분해서 부른다. 지금까지 "동일 벡터&middot;상반 라벨"과 "중복"을 섞어 쓴 탓에 처리 실험의 조건이 흐려졌다.</p>
<div class="wrap"><table>
<tr><th>이름</th><th>정의</th><th>성질</th></tr>
<tr><td class="t"><b>모순</b></td><td class="t">한 feature 벡터가 서로 다른 라벨을 가짐</td><td class="t">데이터셋 고유. 어떤 분할을 써도 남는다</td></tr>
<tr><td class="t"><b>중복</b></td><td class="t">같은 (feature 벡터, 라벨) 쌍이 2회 이상 나타남</td><td class="t">데이터셋 고유. 공격이 반복되면 <b>정상적으로</b> 생긴다</td></tr>
<tr><td class="t"><b>누수</b></td><td class="t">test의 벡터가 train에도 존재함</td><td class="t"><b>분할에 따라 달라진다.</b> 데이터의 성질이 아니라 평가 설계의 결과</td></tr>
</table></div>
<p>두 정의는 직교한다. 모순 벡터도 중복될 수 있고, 중복이 전혀 없는 모순 벡터도 있다. 누수는 층위가 다르다 &mdash; 같은 데이터라도 랜덤 split이면 크고 시간순 split이면 작다. 아래 §4의 처리 실험은 이 직교성 위에서 설계해야 한다. <strong>중복은 따로 도표화하지 않는다</strong> &mdash; volumetric 공격이 정적 표적을 때리면 거의 같은 flow가 수백만 개 나오는 것이 정상이고(Flood et al. EuroS&amp;P 2024), 그 반복은 공격의 서명이자 사전확률이다. 지워야 할 오류가 아니므로 §4에서 처리 축으로만 쓴다.</p>

<h3>1a. 모순 &mdash; 어느 클래스끼리 벡터를 나눠 갖는가</h3>
<p>NetFlow 한 행은 46개 feature로 요약된 흐름이다. IDS 데이터셋의 라벨은 패킷 내용이 아니라 <strong>공격자 IP&middot;대상 포트&middot;시간 창</strong>으로 붙는다(Liu et al. CNS'22; Engelen et al. WTMC'21). 그래서 창 안의 평범한 흐름이 "공격"이 되고, 창 밖의 같은 흐름이 "benign"이 된다. A를 B로 잘못 적었다는 뜻이 아니라, 같은 벡터가 두 라벨을 동시에 갖는다는 뜻이다.</p>
{''.join(sec_conflict(d) for d in sections)}

<h3>1b. 누수 &mdash; test 행을 암기&middot;모순&middot;신규로 나누면</h3>
<p>파이프라인과 같은 시간순 분할에서 test 행을 셋으로 나눈다: train에 <b>같은 라벨</b>로 있던 벡터(암기하면 맞는다), train에 <b>다른 라벨</b>로 있던 벡터(모순이라 무엇을 답해도 한쪽은 틀린다), <b>처음 보는</b> 벡터(여기서만 일반화가 측정된다).</p>
{''.join(sec_leak(d) for d in sections)}
<p class="ref">같은 데이터를 랜덤 80/20으로 자르면 test 벡터의 59&ndash;99%가 train에 그대로 남아, 모델 없이 조회표만 써도 정확도가 84&ndash;98%까지 나온다. 누수는 데이터의 성질이 아니라 분할이 만든다. 위 그림은 파이프라인과 같은 시간순 분할 기준이다.</p>

<h2>2. 모순이 만드는 oracle 천장</h2>
<p>EXP39의 correctness oracle은 "이 행에 대해 어떤 expert가 정답을 냈는가"를 행마다 세는 <strong>row-level oracle</strong>이다. 같은 벡터가 두 라벨을 가지면 row-level oracle은 두 행 모두 맞힌 것으로 세지만, 실제 분류기는 같은 입력에 같은 답을 내야 하므로 둘 중 하나는 반드시 틀린다. 그 제약을 건 값이 <strong>현실적 oracle</strong>이다 &mdash; 라우팅은 완벽하다고 가정하되, 모순 벡터는 그 벡터의 다수 라벨로 답한다고 본다. 클래스별 회수 가능 여지는 아래 그림이 전부이고, 이 격차의 정체는 §1a의 모순이다.</p>
<div class="callout"><strong>0821&middot;0902의 oracle과 왜 다른가.</strong> oracle 값은 어떤 후보 집합(bank) 위에서 재느냐에 따라 달라진다. 0821 §5의 0.964는 exp10 family expert를 완벽 배정했을 때, 0902의 0.81&ndash;0.88은 exp29 bank 기준이며 둘 다 row-level이다. 같은 제약을 걸면 모두 내려간다.</div>
{''.join(sec_oracle_novtable(d) + narr_oracle[d['key']] for d in sections)}

<h2>3. 문헌 조사</h2>
<p>모순&middot;중복&middot;누수를 다른 연구는 어떻게 처리했는가. 아래는 링크 정리이며 <span class="pill">보충 예정</span>이다.</p>
{lit_section()}

<h2>4. 중복과 모순을 분리해서 처리하면 &mdash; EXP41&middot;EXP42</h2>
<p>§1의 직교성 위에 처리를 놓으면 네 칸이 나온다. 중복을 지우는 처리는 "중복을 지운 효과"와 "모순을 없앤 효과"를 한꺼번에 섞으므로, 둘을 떼어 보려면 <b>중복을 남긴 채 모순만 없애는</b> 칸이 필요하다.</p>
<div class="wrap"><table>
<tr><th></th><th>모순 유지</th><th>모순 제거</th></tr>
<tr><td class="t"><b>중복 유지</b></td><td class="t"><b>원본</b><div class="sub">train 12,069,313행</div></td><td class="t"><b>중복 유지 + 모순은 다수 라벨</b><div class="sub">모순 벡터의 행을 그 벡터의 다수 라벨로 바꾼다. <b>행을 하나도 지우지 않으므로</b> train·test 행이 원본과 동일하다. 재라벨 210,728행. train 12,069,313행 (EXP44)</div><br><b>중복 유지 + 모순 벡터 삭제</b><div class="sub">모순 벡터의 행만 제거하고 나머지 중복은 남긴다 (EXP44)</div></td></tr>
<tr><td class="t"><b>중복 제거</b></td><td class="t"><b>중복 제거</b><div class="sub">벡터&middot;라벨 쌍당 1행. 모순 벡터는 라벨별로 1행씩 남으므로 <b>모순은 그대로다</b>. train 5,295,822행</div></td><td class="t"><b>중복 제거 + 모순은 다수 라벨</b><div class="sub">벡터당 1행, 라벨은 그 벡터의 다수 라벨. train 5,283,411행</div><br><b>중복 제거 + 모순 벡터 삭제</b><div class="sub">모순 벡터를 통째로 제거. train 5,271,045행</div></td></tr>
</table></div>
<p>각 처리를 전 행에 적용한 뒤 같은 scenario별 시간순 60/20/20 규칙으로 다시 나눈다. 여섯 처리는 서로 다른 벤치마크가 되므로 먼저 <strong>클래스별로 train&middot;test가 몇 행이 됐는지</strong>를 보이고, 그 위에서 XGBoost(train 전체 / 같은 C0) &middot; TabPFN-v3(무작위 컨텍스트) &middot; ours(구성된 C0) &middot; SOTA 3종을 <strong>클래스별로</strong> 비교한다(EXP42; 중복 유지 + 모순 제거 두 arm은 EXP44이며 LoCalPFN은 arm당 ~3.4 h라 제외). 각 arm의 test로 평가하고, 그 test의 천장은 현실적 oracle이다. 여기서 <strong>ours = 구성된 C0 위의 frozen TabPFN-v3</strong>다 &mdash; 조건부 라우팅 층은 09-09의 EXP38&middot;39에서 6조건 모두 global을 유지해 기여가 0이었으므로(&Delta;global = 0), 남은 기여는 컨텍스트 구성뿐이고 그것을 그대로 비교 대상으로 둔다. 이전 기록: EXP16(0821, train만 중복 제거)은 global까지 붕괴(0.701&rarr;0.599), EXP18(0821, 분할 전 클래스 내 dedup)은 macro 0.7547, exp32/33(0902&ndash;03)은 음성.</p>
{exp42_section()}

<h2>5. ours 개선</h2>
<p>여기서 <strong>ours는 구성된 C0(100k &middot; benign 0.75 &middot; 공격 균등 &middot; n_estimators 4) 위의 frozen TabPFN-v3</strong>이고 그 이상이 아니다 &mdash; scorer&middot;verifier 층은 EXP38&middot;39에서 6조건 모두 global을 선택했다(&Delta;global = 0). 그래서 &quot;ours를 발전시킨다&quot;는 라우팅이 아니라 컨텍스트 구성에서 찾아야 한다.</p>
<p><strong>EXP43 (설계, 실행 보류).</strong> <code>tabpfn/scripts/nfv3_v3_exp43_ours_majfill.py</code>의 knob <code>--attack-fill</code>은 예산&middot;클래스 균형&middot;benign 행을 고정한 채 <strong>공격 클래스 할당량을 어떤 행으로 채울지</strong>만 바꾼다(<code>majority_consistent</code>: 그 벡터의 다수 라벨이 해당 클래스인 행 우선, <code>clean</code>: 다른 라벨이 전혀 없는 벡터 우선). 근거는 §1의 감사다 &mdash; infiltration 행의 75.1%, web의 69.6%가 상반 라벨 트윈이라 균일 추첨은 infiltration 슬롯의 3/4을 benign과 구분 불가능한 벡터에 쓴다. 단 현재 구현은 다수 라벨을 train+val+test 전체에서 계산하므로 그대로는 selective snooping이다 &mdash; 실행한다면 <strong>train 풀 다수 라벨</strong>로 바꾼 새 frozen 스크립트여야 한다. 2026-09-14 사용자 판단으로 실행은 보류했다.</p>
<p><strong>대조군 검증에서 나온 것 (실행함).</strong> <code>--attack-fill natural</code>로 중복 제거 arm을 돌려(run <code>20260914_154808</code>) EXP42의 ours와 같은 행을 뽑는지 확인했다. 클래스별 목표 수(benign 75,000 / bot 841 / brute 5,967 / ddos 5,966 / dos 5,966 / inf 5,965 / web 295)와 추첨 seed가 같아 컨텍스트 행은 동일한데 macro-F1은 0.8055(EXP42) vs 0.8087(EXP43)로 0.0032 어긋난다. 원인은 TabPFN config 추첨의 전역 RNG 상태다(EXP42는 같은 프로세스에서 tabpfn_plain이 먼저 돌았다). <strong>함의: 컨텍스트 knob 비교는 반드시 같은 스크립트&middot;같은 프로세스 순서의 natural 대조군과 해야 한다.</strong> EXP42 값과 비교하면 &plusmn;0.003의 가짜 개선을 볼 수 있다.</p>
<p><strong>방향 (09-15).</strong> 데이터 감사 실험은 EXP44로 마치고 ours 자체의 개선 실험으로 돌아간다. 후보와 우선순위는 <code>lablog/report/0915.md</code>에 기록하며, 실행은 사용자와 합의한 뒤 시작한다.</p>

<h2>6. 시간순 클래스 구성</h2>
<p>CIC-IDS2018·ToN-IoT·BoT-IoT는 공격 family가 <strong>날짜 블록</strong>으로 실행됐다(arXiv 2503.04404 Table 4). benign만 전 구간에 걸쳐 있고 각 공격은 자기 날짜 안에 있다. 파이프라인의 family별 시간순 split은 family 내부의 앞뒤를 자른 것이고, 새 family가 나중에 등장하는 상황은 family 등장 순서로 따로 구성해야 한다(ToN-IoT: scanning → … → backdoor/ransomware → mitm). family 내부에도 변화는 있다: BoT-IoT ddos의 test 벡터는 train에 사실상 없고, ToN-IoT scanning·BoT-IoT theft는 같은 벡터의 라벨이 시점에 따라 바뀐다. UNSW-NB15만 모든 family가 두 수집일에 동시에 존재한다.</p>
{''.join(sec_time(d) for d in sections)}

<h2>7. 판정</h2>
<div class="wrap"><table>
<tr><th>데이터셋</th><th>결함의 본질</th><th>row-level oracle이 부풀리는 곳</th><th>쓸 수 있는 질문</th></tr>
<tr><td class="t">CIC-IDS2018</td><td class="t">시간 창 라벨링 → infiltration·web이 benign과 동일 벡터; slowhttptest=ftp_bruteforce 단일 벡터</td><td class="t">infiltration {row_oracle['infiltration']:.2f} vs 현실적 {orc('cse_cic_ids2018','infiltration'):.2f}, web {row_oracle['web_attacks']:.2f} vs {orc('cse_cic_ids2018','web_attacks'):.2f}</td><td class="t">context 구성(bot·ddos·brute의 깨끗한 부분); 새 family 등장은 bot(고유 벡터 100%, 마지막 날); infiltration은 3분해 보고와 함께만</td></tr>
<tr><td class="t">ToN-IoT</td><td class="t">IP 대역+시간 창 라벨링 → benign 62%·ddos 78%·scanning 97%가 충돌; 응답 없는 SYN 한 개가 수백만 benign 행</td><td class="t">scanning은 test 안에서는 순수(현실적 {orc('ton_iot','scanning'):.2f})하나 train 라벨과 {pct(trc('ton_iot','scanning'),0)} 충돌 → 학습 불가</td><td class="t">mitm·ransomware 정밀도(현실적 oracle {orc('ton_iot','mitm'):.2f}/{orc('ton_iot','ransomware'):.2f} vs XGB {ton_xgb.f1['mitm']:.2f}/{ton_xgb.f1['ransomware']:.2f}); 늦게 등장하는 family + k 라벨 refresh</td></tr>
<tr><td class="t">BoT-IoT</td><td class="t">합성 volumetric, 중복률 96%; benign 0.3%가 recon/dos와 충돌; theft 벡터의 라벨이 5월 recon→6월 theft</td><td class="t">theft 현실적 {orc('bot_iot','theft'):.2f}지만 train 라벨과 {pct(trc('bot_iot','theft'),0)} 충돌; benign {orc('bot_iot','benign'):.2f}</td><td class="t">sanity check 이상 없음</td></tr>
<tr><td class="t">UNSW-NB15</td><td class="t">TTL 지름길로 benign/attack 자명; 공격 카테고리는 IXIA 익스플로잇 목록 라벨이라 서로 근사 동일</td><td class="t">backdoor·dos·analysis의 카테고리 혼동(근사 중복이라 oracle 아래 여지가 학습 가능하다는 보장이 없음)</td><td class="t">공격 간 꼬리(보조, 주의 문구 필수)</td></tr>
</table></div>
<p>답은 데이터 교체가 아니라 평가를 바꾸는 것이다. 이는 새 프로토콜이 아니라 이미 요구되는 관행이다(Arp et al. USENIX Sec'22의 temporal snooping·spurious correlation, Flood et al. EuroS&P'24의 near-duplicate·traffic collapse·wrong label 진단, Liu et al. CNS'22의 라벨 감사, Allamanis Onward!'19의 중복/비중복 test 분해, El Mahdaouy et al. arXiv 2602.05594). 구체적으로: (a) headline macro-F1은 전 클래스로 보고하되 클래스별 현실적 oracle과 train 라벨 충돌률을 옆에 둔다(꼬리 클래스를 빼지 않는다); (b) test를 "train에 같은 라벨로 있던 벡터 / 다른 라벨로 있던 벡터 / 처음 보는 벡터"로 나눠 보고한다; (c) split은 시간순이며 family 등장 순서 프로토콜을 구분한다; (d) 충돌 벡터는 행 단위 학습이 아니라 그룹 단위 비용 정책으로 다루되 다음 시간 구간에서 forward로 검증한다; (e) headroom은 현실적 oracle로만 주장하고 row-level oracle은 진단에만 쓴다; (f) 대상 클래스는 ToN mitm·ransomware, CIC web(꼬리 정밀도), ToN d9–d10과 CIC bot(새 family 등장)으로 미리 고른다.</p>
<p><strong>모순 처리에 대한 판정(09-15).</strong> 모순은 oracle 천장을 낮추는 원인이지만, 지우거나(EXP41 중복 제거&middot;벡터 삭제, EXP44 벡터 삭제) 고치는(EXP41&middot;EXP44 다수 라벨) 어느 처리도 꼬리를 지키지 못했고, 처리 뒤에는 방법 순위까지 뒤집힌다(§4). 다수 라벨 재라벨은 test를 들여다보는 selective snooping이라 배포 전처리가 될 수 없고, 문헌도 IDS 벤치마크의 모순을 지우지 않고 원본 위에서 보고하는 것이 관행이다. 따라서 <strong>headline은 원본 벤치마크</strong>로 두고, 모순은 전처리가 아니라 (a)&middot;(b)의 보고 축 &mdash; 클래스별 현실적 oracle과 test 3분해 &mdash; 로 다룬다. 처리된 벤치마크(§4)는 보충 자료로만 남긴다. 모순에서 얻은 지식은 train 풀 쪽에서만 쓴다(예: 컨텍스트 구성 시 train 다수 라벨이 그 클래스인 행을 우선).</p>

<h2>8. 재현</h2>
<p class="ref">입력 <code>data/nfv3_energy_suite_uncapped_scenarios.pkl</code>. hash = <code>pd.util.hash_pandas_object</code>(nan_to_num float32 행). 감사 코드와 CSV: <code>docs/research/20260911/dataset_quality_audit/</code> (audit_pass1.py, audit_pass2_ceilings_unsw_xgb.py, unsw_probe_ttl_confusion.py, pair_conflicts.py → pair_conflicts.json, ceilings_v2.py → ceilings_v2.csv(현실적 oracle = ceil_majority_test), cic_bank_oracle.py → cic_bank_oracle_derivation.json, report_data_twins/time.py, build_dataset_report.py). EXP41: <code>scripts/exp41_dedup_dataset_xgb.py</code>(results/…_exp41_dedup_dataset_xgb) · <code>tabpfn/scripts/nfv3_v3_exp41_dedup_dataset_tabpfn.py</code>(4 처리 × seed 42·43·44). EXP42(방법 비교): <code>tabpfn/scripts/nfv3_v3_exp42_dedup_methods.py</code> — EXP41과 같은 dedup·재split 코드를 그대로 써서 arm별 train/val/test 행이 동일하고, XGBoost train 전체는 EXP41 run의 예측을 test_idx 일치 확인 후 재사용한다. SOTA는 <code>tabpfn/third_party/{{BoostPFN,LoCalPFN}}</code>의 TabPFN-v1을 쓰며 exp35·exp36과 같은 설정이다. 달성 F1 출처: CIC-IDS2018은 EXP39 test global(0909 §3-C), BoT/ToN은 0818 full-pool XGBoost run, UNSW는 0911 XGB. EXP43(ours v2 설계 &middot; natural 대조군만 실행): <code>tabpfn/scripts/nfv3_v3_exp43_ours_majfill.py</code>. EXP44(중복 유지 + 모순 제거): <code>tabpfn/scripts/nfv3_v3_exp44_conflict_only.py</code>(relabel_majority &middot; drop_vectors, LoCalPFN 제외). 랩 로그 <code>lablog/report/0911.md</code> &middot; <code>0914.md</code> &middot; <code>0915.md</code>.</p>
</main></body></html>'''
open(OUT, 'w', encoding='utf-8').write(page); print('wrote', OUT, len(page))
