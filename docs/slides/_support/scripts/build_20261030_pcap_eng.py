"""Build the English deck on CSV limitations and PCAP-based CIC reconstruction.

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
QA = HERE / '_support/qa/20261030_pcap_eng'
OUT = HERE / 'eng/20261030_PCAP_Research_Direction'
EVIDENCE = ROOT / 'docs/lablog/evidence/20261029_pcap'
AUDIT_PATH = ROOT / 'docs/lablog/evidence/20261030/original_cic2018_csv_audit.json'
EXP87 = ROOT / 'tabpfn/results/v7/exp87/20261022_163739_full_learning_curve_five_seeds_c59213f5'
SOURCE_DATA = json.loads((EVIDENCE / 'sources.json').read_text(encoding='utf-8'))['sources']

r.QA = QA
r.ASSETS = HERE / '_support/assets/20261030_pcap_eng'
r.OUT = OUT
r.N_SLIDES = 6
r.SCENES = []
r.SOURCES = {key: (value['title'], f"{value['venue']}, {value['year']}", value['url'], value['evidence']) for key, value in SOURCE_DATA.items()}
txt, box, arrow, block, page = r.txt, r.box, r.arrow, r.block, r.page
NAVY, BLUE, INK, GREY, PALE = r.NAVY, r.BLUE, r.INK, r.GREY, r.PALE
RED, TEAL = 'B44948', '227B71'

# Keep the source audit on the original CSV, while using the same seven
# target classes as the NF-v3 experiments for all slide-facing labels.
RAW_TO_NFV3 = {
    'Benign': 'benign',
    'DDOS attack-HOIC': 'ddos', 'DDOS attack-LOIC-UDP': 'ddos', 'DDoS attacks-LOIC-HTTP': 'ddos',
    'Bot': 'bot',
    'DoS attacks-GoldenEye': 'dos', 'DoS attacks-Hulk': 'dos',
    'DoS attacks-SlowHTTPTest': 'dos', 'DoS attacks-Slowloris': 'dos',
    'FTP-BruteForce': 'bf', 'SSH-Bruteforce': 'bf',
    'Infilteration': 'inf',
    'Brute Force -Web': 'web', 'Brute Force -XSS': 'web', 'SQL Injection': 'web',
}


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
    ax_curve.set_xlabel('Training labels per class (n)', fontproperties=font, fontsize=10)
    ax_curve.set_title('n = 2 to 450', fontproperties=bold, fontsize=14, pad=10)
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
        ax.set_title(f'Classwise F1 · n = {budget}', fontproperties=bold, fontsize=14, pad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    bar = fig.colorbar(image, ax=fig.axes[1:], shrink=.74, pad=.04)
    bar.set_label('F1 (%)', fontproperties=font, fontsize=9)
    bar.ax.tick_params(labelsize=8)
    fig.savefig(r.ASSETS / 'ceiling_curve_heatmaps.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def make_conflict_asset(audit: dict) -> None:
    """Show source-CSV conflict rates at the NF-v3 target-class granularity."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    r.ASSETS.mkdir(parents=True, exist_ok=True)
    font = FontProperties(fname='/home/user/.local/share/fonts/Pretendard/Pretendard-Regular.otf')
    bold = FontProperties(fname='/home/user/.local/share/fonts/Pretendard/Pretendard-Bold.otf')
    totals = audit['totals']
    family_audit = audit['nfv3_family_audit']
    rows = family_audit['per_family']
    rates = [100 * row['conflicting_fraction'] for row in rows]
    fig, axis = plt.subplots(figsize=(13.0, 2.45), dpi=190)
    axis.barh(range(len(rows)), rates, color='#B44948', height=.55)
    axis.set_yticks(
        range(len(rows)),
        [row['family'] for row in rows],
        fontproperties=bold, fontsize=8.3,
    )
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel('Share of each class in identical-feature, different-class groups (%)', fontproperties=font, fontsize=8.7)
    axis.set_title(
        f"Raw CIC2018 CSV: {totals['rows']:,} rows · seven NF-v3 classes: "
        f"{family_audit['rows_in_conflicting_groups']:,} rows ({100 * family_audit['fraction_rows_in_conflicting_groups']:.2f}%)",
        fontproperties=bold, fontsize=10.6, color='#00306C', pad=7,
    )
    axis.grid(axis='x', color='#D9E1E8', lw=.7)
    axis.spines[['top', 'right', 'left']].set_visible(False)
    axis.tick_params(axis='y', length=0)
    axis.tick_params(axis='x', labelsize=7.5)
    for y, (rate, row) in enumerate(zip(rates, rows)):
        text_x = rate + 1.0 if rate < 95 else rate - 1.0
        axis.text(text_x, y, f"{rate:.1f}% ({row['conflicting_rows']:,})",
                  ha='left' if rate < 95 else 'right', va='center', fontsize=6.8,
                  fontproperties=font, color='#00306C')
    fig.subplots_adjust(left=.25, right=.975, top=.79, bottom=.20)
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
        return [str(row[column]) for column in columns[:5]] + ['… 69 fields …'] + [str(row[column]) for column in columns[-4:]] + [RAW_TO_NFV3[row['label']]]
    label_set = ' · '.join(sorted({RAW_TO_NFV3[item['label']] for item in group['per_label_rows']}))
    return make(chosen[0]), make(chosen[1]), label_set


def scenes() -> None:
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(f'Run source CSV audit first: {AUDIT_PATH}')
    audit = json.loads(AUDIT_PATH.read_text(encoding='utf-8'))
    make_ceiling_asset()
    make_conflict_asset(audit)
    row_a, row_b, conflict_labels = raw_values_for_slide(audit)
    totals = audit['totals']

    # 1. Existing inputs and the CIC-IDS2017 production path.
    a = [
        txt(.65, 1.08, 12.0, 'Prior experiments: XGB · TabPFN · SOTA · proposed models  →  CICFlowMeter CSV vectors', 20, True, NAVY),
        txt(.65, 1.52, 12.0, 'Most IDS tabular-classification studies also train and evaluate on the supplied flow-statistics CSV', 16.5, False, GREY),
        txt(.65, 2.16, 4.45, 'Input to prior classifiers', 21, True, NAVY),
    ]
    a += card(.65, 2.64, 4.45, 1.30, 'One CSV row', '[duration, packets, length, IAT, flags, …]\n→ numeric vector + label', body_size=14.7)
    a += [arrow(2.67, 4.20, .42), txt(.65, 4.78, 4.45, 'Each row treated as an independent sample', 15.5, True, BLUE, 'center')]
    a += [
        txt(5.55, 2.16, 7.05, 'CIC-IDS2017: How PCAP becomes CSV', 21, True, NAVY),
    ]
    stages = [(5.55, 1.45, 'Testbed', 'benign +\nattack'), (7.30, 1.45, 'PCAP', 'raw packets\nin time order'), (9.05, 1.96, 'CICFlowMeter', 'flows + fixed\nstatistics'), (11.52, 1.34, 'CSV', 'vector\n+ label')]
    widths = [1.43, 1.43, 2.10, 1.13]
    for (x, _, title, body), width in zip(stages, widths):
        a += block(x, 2.65, width, 1.33, title, body, body_size=15)
    for x in [7.05, 8.80, 10.80]:
        a.append(arrow(x, 3.15, .20))
    a += [
        txt(5.55, 4.42, 7.0, 'Examples of fixed flow statistics in the CSV', 19, True, NAVY),
    ]
    for x, title, body in [
        (5.55, 'Flow duration', 'first to last packet'),
        (7.92, 'Fwd. packets', 'packets sent'),
        (10.29, 'Packet interval', 'interval statistics'),
    ]:
        a += block(x, 4.92, 2.08, 1.22, title, body, body_size=13.2)
    a += [txt(5.60, 6.40, 6.90, 'A summary of predefined flow statistics, not the packet sequence itself', 14, False, GREY, 'center')]
    page('CICFlowMeter CSV is the standard input for IDS classification', a,
         'Prior models use the supplied flow-statistics CSV', '01 Existing CSV',
         'Prior XGB, TabPFN, SOTA, and proposed-model experiments used CICFlowMeter CSV vectors as classification inputs. Many IDS tabular-classification studies also trained and evaluated with the supplied flow statistics. The right half shows how CIC-IDS2017 produced this CSV: network traffic was captured as PCAP, then CICFlowMeter grouped packets into flows and calculated duration, packet-count, IAT, and flag statistics. A CSV row is a summary of these fixed statistics, not the packet sequence itself.', ['cic17'])

    # 2. State the two observations before their diagnostics.
    a = []
    a += card(.65, 1.88, 5.77, 3.67, 'Issue 1  Early saturation in some attacks', 'Head attacks such as DDoS and Bot\nreach F1 in the 90% range with n = 2 to 20\n\n→ a high data ceiling before adding labels', accent=BLUE, body_size=19)
    a += card(6.91, 1.88, 5.77, 3.67, 'Issue 2  Identical features, different labels', 'Rows with dozens of identical flow statistics\ncan still have different recorded labels\n\n→ one deterministic classifier cannot fit both', accent=RED, body_size=19)
    page('Two issues in CSV-based IDS classification', a,
         'Early saturation and feature–label contradictions require separate diagnosis', '01 Existing CSV',
         'The CSV-based evaluation revealed two issues. First, some head attacks retained high F1 even after their label budget was reduced sharply, so the representation and task difficulty should be checked before collecting more labels. Second, some rows had identical flow statistics but different labels. This does not mean that every label is wrong; it means that the current statistical representation cannot let a deterministic classifier predict both labels correctly. The next two slides show the observations through performance curves and literal source-CSV rows.', [])

    # 3. Long curve and classwise F1 heatmaps.
    a = [
        txt(.65, 1.10, 12.0, 'Macro F1 still changes beyond n = 20, but DDoS and Bot approach high F1 with few labels', 16.5, False, GREY),
        asset_image('ceiling_curve_heatmaps', .56, 1.48, 12.20, 4.55),
        txt(.72, 6.29, 11.8, 'EXP87 · attack-only training and evaluation · benign excluded · 5 seeds · fixed conflict-free test · n = 450 is the balanced-label-budget ceiling', 12.6, False, GREY),
    ]
    page('Some head attacks already have high F1 with few labels', a,
         'Macro F1 changes with n, while classwise saturation is uneven', '02 Data ceiling',
         'This figure extends the earlier CSV-based EXP87 attack-only condition from n = 2 to n = 450. Macro F1 still rises after n = 20, so the entire dataset is not fully saturated. However, head classes such as DDoS and Bot already reach high F1 around n = 20 and change little relative to n = 450. Web attacks remain low even with a larger label budget. Thus, the high average does not mean that every attack is easy: early head saturation and residual tail errors must be examined separately. This is a diagnostic of attack-to-attack classification, not IDS detection with benign traffic included.', [])

    # 4. Literal source rows and the class-level incidence of contradictions.
    headers = ['Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd\nPkts', 'Tot Bwd\nPkts', '… 69 fields …', 'TotLen Bwd\nPkts', 'Flow Byts/s', 'Flow Pkts/s', 'Idle Min', 'NF-v3\nclass']
    a = [
        txt(.65, 1.10, 12.0, 'Exact agreement on 78 common CICFlowMeter statistics, excluding timestamps, identifiers, addresses, ports, and labels', 14.5, False, GREY),
    ]
    a += raw_row(.65, 1.62, headers, label_fill=NAVY)
    a += raw_row(.65, 2.11, row_a, label_fill='FBE8E7')
    a += raw_row(.65, 2.60, row_b, label_fill='E6F1F8')
    a += [
        txt(.72, 3.24, 11.8, f'The two rows above: {conflict_labels}  ·  all other 69 fields are also identical', 15.5, True, RED),
        asset_image('original_csv_conflict_rates', .64, 3.47, 12.04, 2.33),
        txt(.72, 6.05, 11.85, f"Audit of all {totals['rows']:,} raw CSV rows · aggregated to seven NF-v3 classes · bar endpoint: different-class rows", 12.7, False, GREY),
    ]
    page('Identical features, different classes in CIC2018 CSV', a,
         'Literal rows share all 78 common flow statistics but belong to different NF-v3 classes', '03 Label conflicts',
         'This slide audits the ten original CIC2018 CICFlowMeter CSV files, rather than an NF-v3-derived input. Rows were grouped when all 78 statistical fields common to every file agreed after excluding timestamps and flow identifiers. The table shows literal source-CSV values: displayed fields and the omitted 69 fields are identical, but the NF-v3 class differs. The bars retain the order benign, ddos, bot, dos, bf, inf, web. Their length and endpoint show the share and number of rows from each class in identical-feature, different-class groups. These values do not imply a single cause or that every label is wrong; they show why the representation and label-generation evidence must be examined together.', ['cic18', 'liu'])

    # 5. A reusable PCAP-based representation pipeline.
    a = [
        txt(.65, 1.10, 12.0, 'Generate representations from packet sequences, then recheck both issues under the same protocol', 14.5, False, GREY),
    ]
    steps = [
        (.65, 'PCAP + labels', 'packet, time,\nlabel evidence'),
        (3.15, 'Flow linkage', 'traceable traffic\nand labels'),
        (5.65, 'Encoder', 'packet-sequence\nembedding'),
        (8.15, 'Projection', 'fit PCA on the\ntraining split only'),
        (10.65, 'New vectors', 'XGB + TabPFN\nsame evaluation'),
    ]
    for x, title, body in steps:
        a += block(x, 1.95, 2.0, 1.38, title, body, body_size=15)
    for x in [2.79, 5.29, 7.79, 10.29]:
        a.append(arrow(x, 2.49, .24))
    a += card(.65, 3.93, 5.78, 1.28, 'Representation audit', 'identical embeddings with different labels\nand the PCAP evidence for each label', accent=RED, body_size=14.5)
    a += card(6.90, 3.93, 5.78, 1.28, 'Classification evaluation', 'baselines, few-shot curves, and classwise F1\nto test whether early head saturation remains', accent=BLUE, body_size=14.5)
    a += [
        txt(.72, 5.63, 11.9, 'Candidate encoders  netFound (arXiv, 2026) · ET-BERT (WWW, 2022) · NetMamba (ICNP, 2024)', 14.5, False, GREY),
        txt(.72, 6.01, 11.9, 'CICFlowMeter uses specified statistics; encoders learn from packet order, timing, and headers', 14.2, True, NAVY),
    ]
    page('A PCAP-based CIC2018 representation pipeline', a,
         'Keep encoders interchangeable and repeat representation audits with matched classification evaluation', '04 PCAP reconstruction',
         'CICFlowMeter calculates CSV features using pre-specified flow rules and statistics. The new pipeline preserves PCAP, time, and label linkage while a packet-sequence encoder generates embeddings and PCA is fit only on the training split. The encoder interface is interchangeable, so netFound, ET-BERT, and NetMamba can be compared with the same inputs, split, and evaluation. Each new vector space is first audited for identical embeddings with different labels and for PCAP label evidence. XGB and TabPFN baselines, few-shot learning curves, and classwise F1 then test whether early head saturation persists in the new representation.', ['netfound', 'etbert', 'netmamba'])

    # 6. Summary.
    a = []
    a += card(.65, 1.48, 3.55, 2.58, '1  Existing CSV input', 'XGB · TabPFN · SOTA\nCICFlowMeter flow statistics\nrow-level inputs', accent=BLUE, body_size=16)
    a += [arrow(4.43, 2.59, .44)]
    a += card(5.04, 1.48, 3.55, 2.58, '2  Observed data issues', 'early head saturation\nidentical features, different classes\n\naverage score alone\ndoes not establish difficulty', accent=RED, body_size=13.8)
    a += [arrow(8.82, 2.59, .44)]
    a += card(9.43, 1.48, 3.25, 2.58, '3  Next validation', 'new vectors from PCAP\n\nrepeat audits and baselines\nunder the same protocol', accent=TEAL, body_size=13.8)
    a += [
        box(.65, 4.77, 12.03, 1.10, 'F6F9FC', 'D3DFE9'),
        txt(.92, 5.06, 11.5, 'Goal  Test whether new representations reduce the data ceiling and feature–label conflicts', 16.2, True, NAVY, 'center'),
        txt(.92, 5.63, 11.5, 'If reduced, repeat TabPFN and XGB comparisons on the new vectors; if retained, examine PCAP label linkage further', 14.5, False, GREY, 'center'),
    ]
    page('Summary', a,
         'Recheck both CSV observations in a new PCAP-based representation', '05 Summary',
         'The presentation has three steps. Existing IDS classifiers used fixed CICFlowMeter statistics from CSV inputs. Inspection of those inputs identified early saturation for some head attacks and identical-feature, different-class rows in the original CSV. The next step is to measure both phenomena again on new CIC2018 vectors created by PCAP encoders, using the same audit and baseline protocol. Only after verifying whether the representation reduces conflicts and improves separation of tail or emerging attacks can a classification benefit from the new representation or model design be assessed.', [])


def main() -> None:
    QA.mkdir(parents=True, exist_ok=True)
    r.ASSETS.mkdir(parents=True, exist_ok=True)
    scenes()
    r.validate_scenes()
    r.pptx()
    from pptx import Presentation
    presentation = Presentation(OUT.with_suffix('.pptx'))
    presentation.core_properties.title = 'Limits of CIC CSV Inputs and PCAP-Based Representation Learning'
    presentation.core_properties.subject = 'Raw CIC2018 CSV audit and EXP87 learning curves'
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
    assert 'NF-CIC' not in visible
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
    print(json.dumps({'pptx': str(OUT.with_suffix('.pptx')), 'pdf': str(OUT.with_suffix('.pdf')), 'slides': len(r.SCENES)}))


if __name__ == '__main__':
    main()
