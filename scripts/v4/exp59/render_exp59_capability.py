"""Render the two saved-prediction expert capability views for EXP59."""

# Repository layout bootstrap: works in the workspace and portable source snapshots.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / 'scripts/common/experiment_paths.py').is_file())
_layout_sys.path.insert(0, str(_layout_root / 'scripts/common'))
from experiment_paths import bootstrap, repo_root, script_path, resolve_path, result_root, snapshot_path, read_record
bootstrap(_layout_root)

from build_exp56_report_section import Markup, table


def n(x):
    return f'{int(x):,}'


def f(x):
    return f'{float(x):.4f}'


def pct(x):
    return f'{100 * float(x):.2f}%'


def metric_value(m, key):
    if key == 'precision' and not m['predicted']:
        return '—'
    if key in ['recall', 'f1'] and not m['support']:
        return '—'
    return f(m[key])


def paired(c, key):
    g, e = c['global_metrics'], c['expert_metrics']
    if key == 'FP':
        a, b = n(g[key]), n(e[key])
    else:
        a, b = metric_value(g, key), metric_value(e, key)
    direction = 'gain' if (e[key] <= g[key] if key == 'FP' else e[key] >= g[key]) else 'loss'
    if '—' in [a, b]:
        direction = ''
    return Markup(f'<span class="paired">{a} → <b class="{direction}">{b}</b></span>')


def interpretations(ds, expert):
    k = expert['expert']
    own, near = expert['residual'], expert['observable']
    text = {
        ('cic2018', 1): (
            '담당영역의 개선은 주로 benign 오탐 복구다.',
            'Infiltration F1은 0.1368 → 0.8893이다. 실제 infiltration 정답 수는 Global과 같고, benign을 infiltration으로 오인하는 FP를 줄인 효과다. 교정·훼손은 모두 실제 benign 행에서 발생한다.',
            '입력 유사군에서는 infiltration 5,602개를 Global과 e1 모두 benign으로 예측한다. 정답 residual R1에서의 개선을 입력만으로 접근 가능한 infiltration 식별력으로 일반화할 수 없다.'),
        ('cic2018', 2): (
            'Bot 성능은 유지하고, 담당영역의 benign을 복구한다.',
            'R2의 1,475건 교정 중 1,471건은 benign, 4건은 bot이다. Bot F1은 반올림 기준 두 모델 모두 0.9985이므로 “bot 자체가 크게 향상됐다”는 해석과 구분한다.',
            '입력 유사군은 bot 41,417개와 benign 4개뿐이다. Global 대비 교정·훼손 모두 0건이고 benign 4개는 둘 다 bot으로 오인한다. 음성이 4개이므로 혼동 억제 능력을 충분히 검증했다고 볼 수 없다.'),
        ('cic2018', 3): (
            'Infiltration 교정 능력은 있으나 유사한 benign 구분이 병목이다.',
            'R3 infiltration F1은 0.4402 → 0.9674이며 해당 클래스에서 5,364건을 교정한다. R3 안에서 benign 출력에 압도된다는 해석은 맞지 않는다.',
            '입력 유사군에서는 benign 22,165개 중 21,535개를 infiltration으로 오인한다. Infiltration FP는 7,935 → 21,535, F1은 0.4066 → 0.2211이다. 담당 residual의 교정 능력과 benign 경계의 취약성이 동시에 존재한다.'),
        ('cic2018', 4): (
            'Benign 복구에 치우쳐 web attack 정답을 잃는다.',
            'Benign 389건을 교정하지만 web attack 정답 27건을 훼손한다. Web F1은 0.1183 → 0.2286으로 오르더라도 TP는 31 → 4로 줄어든다. FP 감소와 recall 손실을 함께 읽어야 한다. Infiltration 80개는 전부 benign으로 예측한다.',
            '입력 유사군에서도 benign 교정 385건과 web attack 훼손 104건이 함께 발생한다. Dos 322개와 infiltration 82개 모두 benign으로 예측하므로 공격 식별 expert로 사용하려면 이 혼동을 context에서 보완해야 한다.'),
        ('toniot', 1): (
            'Scanning 개선이 두 관점에서 유지되지만 benign 오탐은 남는다.',
            'R1 scanning F1은 0.0397 → 0.9568이다. Scanning 5,412건과 mitm 56건을 교정해, 단순한 주 클래스 출력 비중 이상의 개선이 확인된다.',
            '입력 유사군에서도 scanning F1은 0.0496 → 0.8840이다. 다만 scanning FP가 36 → 1,115로 늘며, 그중 830건은 benign이다. 유용한 후보이지만 benign을 구별하는 사례를 보강해야 한다.'),
        ('toniot', 2): (
            '일부 공격 F1 향상과 benign 오탐 증가가 함께 나타난다.',
            'R2에서는 41,984건 교정에 비해 75,492건 훼손이 발생한다. Mitm은 81개를 모두 맞히지만 FP 54,220개로 precision 0.0015, F1 0.0030이다. 실제 ransomware가 없는데도 11,177개를 ransomware로 예측한다.',
            '입력 유사군에서는 injection F1이 0.9189 → 0.9509로 개선되지만, benign → xss 오인이 106,515 → 114,725로 늘어난다. 담당영역에서도 생기는 오류이므로 라우팅뿐 아니라 context의 음성 구성을 개선해야 한다.'),
        ('toniot', 3): (
            'Injection 교정은 강하지만 mitm·dos로의 오인도 함께 억제해야 한다.',
            'R3 injection F1은 0.1595 → 0.9391, 교정은 2,895건이다. R3 benign의 주 오인은 injection이 아니라 mitm 1,080건이다. 낮은 타 클래스 성능을 전부 injection 탓으로 묶지 않는다.',
            '입력 유사군에서는 benign → mitm 52,481건, xss → injection 12,659건, xss → dos 7,880건이 발생한다. 총 교정 8,262건에 비해 훼손 55,466건이다. 특화 능력은 있으나 이 유사군 전체를 맡기기는 어렵다.'),
        ('toniot', 4): (
            'Ransomware 탐지는 가능하지만 구분 능력의 개선은 작다.',
            'R4는 ransomware 709개만 있다. 57건을 교정하고 recall 0.9958을 얻지만, 다른 클래스가 없어 FP 억제 능력은 확인할 수 없다.',
            '입력 유사군은 ransomware 646개와 다른 클래스 11,658개다. Ransomware recall은 두 모델 모두 1.0000이며 FP는 9,354 → 9,333, F1은 0.1214 → 0.1216이다. Expert precision은 0.0647에 그친다. 이 범위에서는 Global보다 조금 낫지만, 높은 담당영역 recall이 클래스 구분 능력을 뜻하지는 않는다.'),
    }
    return text[(ds, k)]


def confusion_table(ds, k, scope, view):
    rows = []
    for p in view['confusion_pairs'][:5]:
        rows.append([f"{p['true_class']} → {p['predicted_class']}", n(p['rows']),
                     n(p['global_count']), n(p['expert']), pct(p['expert'] / p['rows']), n(p['new_harm'])])
    return table(['실제 → 오인 클래스', '해당 실제 클래스 수', 'Global 오인',
                  'Expert 오인', '해당 클래스 중 비율', '그중 Global 정답 훼손'],
                 rows, f'exp59-{ds}-e{k}-{scope}-confusions')


def render(ds, title, capability):
    experts = capability['experts']
    default = 3 if ds == 'cic2018' else 4
    out = [f'<h3>{title} — expert별 판정</h3>']
    rows = []
    for e in experts:
        own, near = e['residual'], e['observable']
        summary, _, _ = interpretations(ds, e)
        cells = [f"e{e['expert']}"]
        for view in [own, near]:
            cells.append(Markup(f'<span class="paired"><b class="gain">{n(view["fixed"])}</b> / <b class="loss">{n(view["harmed"])}</b></span><small class="count-note">평가 {n(view["rows"])}개</small>'))
        rows.append([*cells, summary])
    out.append(table(['Expert', '담당영역 교정 / 훼손', '입력 유사군 교정 / 훼손', '해석'], rows, f'exp59-{ds}-capability-summary'))
    out.append(f'<label class="expert-picker">세부 expert <select class="capability-select" data-dataset="{ds}">' + ''.join(
        f'<option value="{e["expert"]}"'+(' selected' if e['expert'] == default else '')+f'>e{e["expert"]}</option>' for e in experts)+'</select></label>')
    for e in experts:
        k = e['expert']
        own, near = e['residual'], e['observable']
        summary, own_note, near_note = interpretations(ds, e)
        out.append(f'<article class="capability-panel" data-dataset="{ds}" data-expert="{k}"'+(' hidden' if k != default else '')+'>')
        out.append(f'<p class="expert-verdict"><b>e{k}: {summary}</b></p>')
        out.append(f'<h4>관점 A · 담당 residual 영역 R{k}에서 Global의 오류를 고치는가?</h4>')
        out.append(f'<p class="note">평가 {n(own["rows"])}개 · 교정 {n(own["fixed"])}건 · 훼손 {n(own["harmed"])}건. 클래스 행은 실제 정답 기준이다. F1에는 이 영역의 다른 클래스에서 발생한 FP도 포함한다.</p>')
        rows = []
        for c in own['classes']:
            g, m = c['global_metrics'], c['expert_metrics']
            rows.append([c['name'], n(m['support']), metric_value(g, 'f1'), metric_value(m, 'f1'),
                         n(c['fixed']), n(c['harmed']), n(g['FP']), n(m['FP'])])
        out.append(table(['클래스', '실제 표본', 'Global F1', f'e{k} F1', '정답 교정', '정답 훼손', 'Global FP', f'e{k} FP'],
                         rows, f'exp59-{ds}-e{k}-own-capability'))
        out.append(f'<p class="interpretation"><b>해석.</b> {own_note}</p>')
        out.append('<details><summary>담당영역에서 남는 주요 오분류 5개</summary>'+confusion_table(ds, k, 'residual', own)+'</details>')
        out.append(f'<h4>관점 B · 입력이 가까운 다른 클래스도 구분하는가?</h4>')
        out.append(f'<p class="note">입력 유사군 O{k}: {n(near["rows"])}개 · 교정 {n(near["fixed"])}건 · 훼손 {n(near["harmed"])}건. 아래 화살표는 같은 행의 <b>Global → e{k}</b>다. 양성은 해당 클래스, 음성은 이 유사군의 나머지 클래스다.</p>')
        rows = []
        for c in near['classes']:
            m = c['expert_metrics']
            rows.append([c['name'], Markup(f'{n(m["support"])} / {n(c["negative_rows"])}'),
                         *[paired(c, key) for key in ['precision', 'recall', 'f1', 'FP']]])
        out.append(table(['클래스', '양성 / 음성 수', 'Precision · G → E', 'Recall · G → E', 'F1 · G → E', 'FP · G → E'],
                         rows, f'exp59-{ds}-e{k}-near-capability'))
        out.append(f'<p class="interpretation"><b>해석.</b> {near_note}</p>')
        out.append('<p class="note">어떤 클래스가 무엇으로 오인됐는가? Expert의 오분류 건수 순으로 최대 5개를 표시한다. “Global 정답 훼손”은 Expert 오인 중 Global이 원래 정확히 맞힌 샘플 수다.</p>')
        out.append(confusion_table(ds, k, 'observable', near))
        out.append('</article>')
    return '\n'.join(out)
