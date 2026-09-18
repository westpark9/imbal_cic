#!/usr/bin/env python3
"""Build a self-contained visual report from saved EXP47/48 evidence. No fitting."""
import csv
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np
from explain_exp47_mechanism import explain

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPORT_ID = 'clean_revalidation_0916'
OUTPUT = ROOT / 'lablog/html_report' / (REPORT_ID + '.html')


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    rows = list(csv.DictReader(Path(path).open()))
    for row in rows:
        for key, value in row.items():
            if value in ['True', 'False']:
                row[key] = value == 'True'
            else:
                try:
                    row[key] = float(value) if any(c in value for c in '.eE') else int(value)
                except (ValueError, TypeError):
                    pass
    return rows


def collect(launch):
    run = Path(launch['run_dir'])
    result = {'run': str(run.relative_to(ROOT) if run.is_relative_to(ROOT) else run), 'launch': launch}
    clean = Path(launch['clean_dir'])
    if (clean / 'manifest.json').exists():
        result['clean'] = read_json(clean / 'manifest.json')
        result['class_counts'] = read_csv(clean / 'class_counts.csv')
    result['state'] = 'preparing'
    for state in ['RUNNING', 'STOPPED', 'ERROR', 'COMPLETE']:
        if (run / (state + '.json')).exists():
            result.update(state=state.lower(), status=read_json(run / (state + '.json')))
    if (run / 'trace_progress.json').exists():
        result['progress'] = read_json(run / 'trace_progress.json')
    if result['state'] != 'complete':
        return result
    cache, diag = run / 'frozen_cache', run / 'diagnostics'
    meta = read_json(cache / 'COMPLETE.json')
    quality = [r for r in read_csv(diag / 'expert_quality.csv') if r['split'] == 'eval']
    glob = {r['class']: r for r in quality if r['model'] == 'global'}
    matrices = {}
    for row in quality:
        if row['model'] == 'global':
            continue
        g = glob[row['class']]
        gc = round(g['recall'] * g['support'])
        matrix = {'both_correct': gc - row['harmful'], 'harmful': row['harmful'],
                  'helpful': row['helpful'], 'both_wrong': g['support'] - gc - row['helpful']}
        assert min(matrix.values()) >= 0 and sum(matrix.values()) == row['support']
        assert matrix['both_correct'] + matrix['helpful'] == round(row['recall'] * row['support'])
        matrices.setdefault(row['model'], {})[row['class']] = matrix
    for model, rows in matrices.items():
        rows['all'] = {k: sum(row[k] for row in rows.values()) for k in next(iter(rows.values()))}
    stages = [r for r in read_csv(diag / 'baseline_routing_stage_counts.csv') if r['split'] == 'eval']
    aggregates = {}
    for stage in sorted({r['stage'] for r in stages}):
        aggregates[stage] = {key: sum(r[key] for r in stages if r['stage'] == stage)
                             for key in ['rows', 'helpful', 'harmful', 'changed', 'taken', 'bank_correctable', 'global_errors']}
    with np.load(Path(meta['source_run']) / 'system_dump.npz') as dump:
        native = {'calls': int((dump['proposal'] > 0).sum()), 'accepted': int(dump['accepted'].sum()),
                  'same_as_global': bool(np.array_equal(dump['final'], dump['y_glob'])),
                  'tau_pre': str(float(dump['tau_pre'])), 'tau_post': str(float(dump['tau_post'])),
                  'q_corr': float(dump['q_corr'])}
        # Macro-F1 is recomputed for the final system; expert CSV supplies the global reference.
        y, pred = dump['y_true'], dump['final']
        C = len(meta['class_names'])
        cm = np.bincount(y * C + pred, minlength=C*C).reshape(C, C)
        native['macro_f1'] = float(np.mean(np.divide(2*np.diag(cm), cm.sum(0)+cm.sum(1),
                                                   out=np.zeros(C), where=(cm.sum(0)+cm.sum(1)) > 0)))
        actual = aggregates['native_final_policy']
        assert actual['helpful'] == int(((dump['y_glob'] != y) & (pred == y)).sum())
        assert actual['harmful'] == int(((dump['y_glob'] == y) & (pred != y)).sum())
        assert actual['taken'] == native['accepted'] <= native['calls']
    grid = read_csv(diag / 'baseline_policy_rejection_reasons.csv')
    result.update(meta=meta, quality=quality, matrices=matrices, stages=aggregates,
                  stage_by_class=stages, grid=grid, native=native,
                  macro_global=sum(r['f1'] for r in glob.values()) / len(glob),
                  scope=read_json(diag / 'audit_scope.json'))
    if (diag / 'exp39_legacy').exists():
        summaries = list((run / 'routing').glob('*/summary.csv'))
        if len(summaries) == 1:
            result['legacy_summary'] = read_csv(summaries[0])
    residual = read_csv(diag / 'residual_audit.csv')
    result['residual'] = []
    mass = sum(r['residual_mass'] for r in residual)
    clipped = sum(r['clipped_mass'] for r in residual)
    for name in meta['class_names']:
        rows = [r for r in residual if r['class'] == name]
        result['residual'].append({'class': name, 'rows': sum(r['rows'] for r in rows),
                                  'wrong': sum(r['wrong'] for r in rows),
                                  'raw_fraction': sum(r['residual_mass'] for r in rows) / max(mass, 1e-20),
                                  'clipped_fraction': sum(r['clipped_mass'] for r in rows) / max(clipped, 1e-20)})
    with np.load(diag / 'scorer_targets.npz') as t:
        y = t['y']
        positive = t['utility'] > 0
        g = np.load(cache / 'route_p0.npy', mmap_mode='r').argmax(1)
        pairs = dict(both_correct=0, helpful=0, harmful=0, both_wrong=0)
        for k in range(meta['n_experts']):
            pred = np.load(cache / f'route_p{k+1}.npy', mmap_mode='r').argmax(1)
            take = positive[:, k]
            for key, event in [('both_correct', (g == y) & (pred == y)), ('helpful', (g != y) & (pred == y)),
                               ('harmful', (g == y) & (pred != y)), ('both_wrong', (g != y) & (pred != y))]:
                pairs[key] += int((take & event).sum())
        assert sum(pairs.values()) == int(positive.sum())
        result['positive_pairs'] = pairs
    # Use the original native calibration candidate and mask, not regrouped dense predictions.
    with np.load(diag / 'baseline_cal_scores.npz') as scores:
        y = np.load(cache / 'cal_y.npy')
        g = np.load(cache / 'cal_p0.npy').argmax(1)
        mask = np.load(cache / 'cal_baseline_mask.npy')
        candidate = scores['candidate']
        result['cal_rows'] = int(mask.sum())
        result['cal_stages'] = []
        for name, take in [('Scorer top-1', mask), ('원시 verifier > 0', mask & (scores['q_hat'] > 0)),
                           ('보정 verifier > 0', mask & (scores['g_lower'] > 0))]:
            result['cal_stages'].append({'name': name, 'helpful': int((take & (g != y) & (candidate == y)).sum()),
                                         'harmful': int((take & (g == y) & (candidate != y)).sum())})
    return result


def write_ton_result(ton):
    """Keep a concise source report alongside the visual snapshot after completion."""
    if ton['state'] != 'complete':
        return
    stage, native = ton['stages'], ton['native']
    top, post, actual = [stage[k] for k in ['top1_no_gates', 'raw_verifier_gt0_diagnostic', 'native_final_policy']]
    cal = ton['cal_stages']
    lines = ['# EXP48 — ToN 모순 제거 후 기존 구조 재검증', '',
             f"완료: {ton['status']['completed_kst']}, {ton['status']['seconds']/60:.1f}분, seed 42.", '',
             f"- Global macro-F1 {ton['macro_global']:.6f}; 실제 시스템 {native['macro_f1']:.6f}.",
             f"- 실제 호출 {native['calls']:,}, 채택 {native['accepted']:,}; 교정 {actual['helpful']:,}, 훼손 {actual['harmful']:,}.",
             f"- Global 오답 {top['global_errors']:,}행 중 bank 교정 기회 {top['bank_correctable']:,}행.",
             f"- Gate 생략 test top-1: 교정 {top['helpful']:,} / 훼손 {top['harmful']:,}.",
             f"- 호출 gate 생략 + 원시 verifier 양수 test 진단: 교정 {post['helpful']:,} / 훼손 {post['harmful']:,}.",
             f"- 선택용 cal {ton['cal_rows']:,}행: top-1 {cal[0]['helpful']:,}/{cal[0]['harmful']:,} → 원시 양수 {cal[1]['helpful']:,}/{cal[1]['harmful']:,} → 보정 양수 {cal[2]['helpful']:,}/{cal[2]['harmful']:,}.",
             f"- 임계값 후보 {len(ton['grid'])}개 중 조건 통과 {sum(r['feasible'] for r in ton['grid'])}개.", '',
             '| 클래스 | Global F1 | Test 행 | Bank 교정 기회 | Top-1 교정 | Top-1 훼손 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in ton['quality']:
        if row['model'] != 'global':
            continue
        s = next(r for r in ton['stage_by_class'] if r['stage'] == 'top1_no_gates' and r['class'] == row['class'])
        lines.append(f"| {row['class']} | {row['f1']:.6f} | {row['support']:,} | {s['bank_correctable']:,} | {s['helpful']:,} | {s['harmful']:,} |")
    legacy = next((r for r in ton.get('legacy_summary', []) if r['arm'] == 'legacy' and r['split'] == 'full_test'), None)
    confirm = next((r for r in ton.get('legacy_summary', []) if r['arm'] == 'legacy' and r['split'] == 'cal_confirm'), None)
    if legacy:
        lines += ['', '## 별도 EXP39 legacy 정책', '',
                  f"같은 새 bank에서 학습한 별도 정책이다. Macro-F1 **{legacy['macro_f1']:.6f}** (global 대비 +{legacy['delta_macro_f1']:.6f}), 실제 호출 {legacy['proposed']:,}건, 채택 {legacy['accepted']:,}건, 교정 **{legacy['helpful']:,}** / 훼손 **{legacy['harmful']:,}**이다. 실제 라벨 변경 {legacy['changed']:,}건에는 둘 다 오답인데 다른 라벨로 바뀐 행도 포함한다.", '',
                  f"Test tail F1은 {legacy['tail_f1']:.6f}, global 대비 {legacy['delta_tail_f1']:+.6f}이다. Benign FPR은 {100*legacy['benign_fpr']:.4f}%, global 대비 {100*legacy['delta_benign_fpr']:+.4f}%p다. 전체 macro-F1 개선이 mitm/ransomware 개선까지 의미하지는 않는다."]
    if confirm:
        lines += ['', f"후행 확인 cal: 채택 {confirm['accepted']:,}, 교정 {confirm['helpful']:,} / 훼손 {confirm['harmful']:,}, Δmacro-F1 {confirm['delta_macro_f1']:+.6f}, Δtail-F1 {confirm['delta_tail_f1']:+.6f}. Tail F1 비감소라는 엄격한 후속 채택 기준은 아직 충족하지 못한다."]
    lines += ['', f"재평가 범위: 전체 expert cal 재평가와 native 후보 차이 {ton['scope']['cal_dense_vs_original_candidate_changes']}행, 전체 expert test 재평가로 재구성한 정책과 native 최종 라벨 차이 {ton['scope']['dense_diagnostic_vs_native_final_label_changes']}행. 위 실제 EXP31 성능·호출·채택·H/D는 native system_dump 기준이며, gate 생략 진단과 구분했다."]
    lines += ['', '## 다음 단계', '',
              '우선 저장된 CIC2018/ToN bank에서 기존 scorer를 고정한 verifier target 대조를 수행할 수 있다. 그 다음 기존/변경 scorer × 기존/변경 verifier의 2×2 대조로 효과를 분리한다. 후보 회수와 승인 생존율, benign 훼손을 클래스별로 측정한다. 실제 채택이 있어도 후행 confirmation의 macro-F1·tail F1·benign FPR 조건을 통과해야 주 실험으로 채택한다.', '',
              'Expert 자체의 유용한 교정 영역이 부족하면 같은 context 예산에서 benign 대조 사례와 anchor 구성을 비교한다. ToN C0가 병목이면 정제 데이터의 같은 100k 예산으로 기존 C0/무작위/관측 특성 구성을 별도 검증한다. Test에서 유리한 임계값을 골라 보고하지 않는다.', '',
              f"원본 실행: `{ton['run']}`. 전체 수치와 행렬: `clean_revalidation_visual_data.json`. 시각화: `lablog/html_report/clean_revalidation_0916.html`.", '']
    (HERE / 'exp48_results.md').write_text('\n'.join(lines))
    log = ROOT / 'lablog/report/0916.md'
    if log.exists():
        text = log.read_text()
        block = ('\n\n<!-- EXP48_COMPLETION -->\n## EXP48 완료 결과\n\n' + '\n'.join(lines[2:11])
                 + '\n\n상세: [EXP48 결과](../../docs/research/20260916/exp48_results.md), '
                   '[시각화 HTML](../html_report/clean_revalidation_0916.html). 로컬 원본·통합본 동기화 대상이며 온라인 아티팩트 갱신과 Git push는 수행하지 않았다.\n<!-- /EXP48_COMPLETION -->\n')
        if '<!-- EXP48_COMPLETION -->' in text:
            text = re.sub(r'\n*<!-- EXP48_COMPLETION -->.*?<!-- /EXP48_COMPLETION -->\n*', block, text, flags=re.S)
        else:
            text += block
        log.write_text(text)
    cic_markdown = HERE / 'exp47_results.md'
    cic_markdown.write_text(cic_markdown.read_text().replace(
        'ToN EXP48은 별도로 실행 중이며 완료 결과도 같은 HTML에 반영한다.',
        'ToN EXP48도 완료했으며 결과를 같은 HTML에 반영했다.'))


def brief_case(result):
    """Same three reporting sections for both datasets, using native cal candidates."""
    root = Path(result['launch']['run_dir'])
    base = Path(result['meta']['source_run'])
    cache, diag = root / 'frozen_cache', root / 'diagnostics'
    contexts = read_csv(base / '2a_expert_contexts.csv')
    anchor = next(r for r in contexts if r['expert'] == 'anchor(shared)')
    experts = [dict(r) for r in contexts if isinstance(r['expert'], int)]
    assert len(experts) == result['meta']['n_experts']
    with np.load(diag / 'residual_inputs.npz') as source, np.load(base / 'context_rows.npz') as saved:
        ids, y, pred = source['row_id'], source['y'], source['p0'].argmax(1)
        order = ids.argsort()
        for expert in experts:
            block = saved[f"expert{expert['expert']}_block"]
            positions = order[np.searchsorted(ids[order], block)]
            assert np.array_equal(ids[positions], block)
            assert len(block) == expert['block_rows']
            assert sum(expert[name] for name in result['meta']['class_names']) == len(block)
            expert['global_wrong'] = int((pred[positions] != y[positions]).sum())
            expert['total_context'] = expert['anchor_rows'] + expert['block_rows']
    if result['native']['calls'] == 0:
        # A rejected candidate with only the decided-support failure explains the bottleneck.
        example = max((r for r in result['grid'] if r['reason'] == 'decided_support'),
                      key=lambda r: r['net_gain'])
    else:
        example = next(r for r in result['grid']
                       if np.isclose(r['tau_pre'], float(result['native']['tau_pre']), rtol=0, atol=1e-12)
                       and r['tau_post'] == float(result['native']['tau_post']))
    with np.load(diag / 'baseline_cal_scores.npz') as scores:
        y = np.load(cache / 'cal_y.npy')
        glob = np.load(cache / 'cal_p0.npy').argmax(1)
        candidate = scores['candidate']
        mask = np.load(cache / 'cal_baseline_mask.npy')
        called = mask & (scores['score'].max(1) > example['tau_pre'])
        accepted = called & (scores['g_lower'] > example['tau_post'])
        stages = []
        for name, take in [('Scorer 추천', mask), ('호출 기준 통과', called), ('Verifier 승인', accepted)]:
            h = int((take & (glob != y) & (candidate == y)).sum())
            d = int((take & (glob == y) & (candidate != y)).sum())
            cc = int((take & (glob == y) & (candidate == y)).sum())
            ww = int((take & (glob != y) & (candidate != y)).sum())
            assert h + d + cc + ww == int(take.sum())
            stages.append(dict(name=name, rows=int(take.sum()), helpful=h, harmful=d,
                               both_correct=cc, both_wrong=ww))
        assert (stages[-1]['rows'], stages[-1]['helpful'], stages[-1]['harmful']) == (
            example['accepted'], example['helpful'], example['harmful'])
    for split in ['train', 'test']:
        assert sum(r['after'] for r in result['class_counts'] if r['split'] == split) == result['clean']['split_rows'][split]
    candidates = read_csv(base / '2d_k_selection.csv')
    return {'anchor': anchor, 'experts': experts, 'cal_example': example, 'cal_funnel': stages,
            'chosen_k_before_pruning': max(candidates, key=lambda r: r['tune_oracle_macro'])['K'],
            'feasible_count': sum(bool(r['feasible']) for r in result['grid'])}


def main():
    cic = collect(read_json(ROOT / 'docs/research/20260915/exp47_launch.json'))
    ton = collect(read_json(HERE / 'exp48_launch.json'))
    write_ton_result(ton)
    evidence = read_json(HERE / 'exp47_result_evidence.json')
    assert cic['native']['calls'] == 0 and cic['native']['accepted'] == 0
    assert abs(cic['macro_global'] - evidence['native']['macro_f1']) < 1e-12
    assert cic['matrices']['expert6']['all']['helpful'] == 5720
    assert cic['matrices']['expert6']['benign']['harmful'] == 804855
    assert cic['stages']['top1_no_gates']['helpful'] == 4685
    assert cic['stages']['top1_no_gates']['harmful'] == 553246
    assert cic['positive_pairs']['both_correct'] == evidence['positive_scorer_training_pairs']['both_correct']
    data = {'asof': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds'),
            'cic': cic, 'ton': ton, 'evidence': evidence,
            'residual': read_csv(HERE / 'exp47_residual_context_evidence.csv')}
    data['mechanism'] = explain(cic, read_csv)
    data['brief'] = {'cic': brief_case(cic), 'ton': brief_case(ton)}
    (HERE / 'exp47_mechanism_explainer.json').write_text(json.dumps(data['mechanism'], ensure_ascii=False, indent=2)+'\n')
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('</', r'<\/')
    template = (HERE / 'clean_revalidation_template.html').read_text()
    assert template.count('__REPORT_DATA__') == 1
    OUTPUT.write_text(template.replace('__REPORT_DATA__', encoded))
    (HERE / 'clean_revalidation_visual_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    index = ROOT / 'lablog/html_report/post_tabpfn.html'
    text = index.read_text()
    match = re.search(r'const REPORTS = (.*?);\n', text)
    reports = json.loads(match.group(1))
    report = next((r for r in reports if r['id'] == REPORT_ID), None)
    if report is None:
        report = {'id': REPORT_ID, 'type': 'iframe', 'group': 'report', 'b64': ''}
        reports.insert(1, report)
    report.update(date='2026-09-16', titleKr='동일벡터상반라벨 제거 후 모델 병목 지점 관찰',
                  titleEn='CIC2018 / ToN · Data, experts & routing bottlenecks',
                  verdict='CIC2018 승인 0 · ToN 승인 271,715',
                  verdictClass='warn')
    for old in reports:
        if old['id'] == 'dataset_quality_0911':
            old['verdict'] = '과거 처리 비교 · 후속 정제 재검증은 09-16'
    text = text[:match.start(1)] + json.dumps(reports, ensure_ascii=False) + text[match.end(1):]
    if ton['state'] == 'complete':
        line = f"완료. Global macro-F1 {ton['macro_global']:.6f}, 실제 시스템 {ton['native']['macro_f1']:.6f}. 실제 expert 채택 {ton['native']['accepted']:,}건. 상세 진단은 09-16 보고서에 반영했다."
        text = text.replace('CIC2018 EXP47 완료, ToN EXP48 실행.', 'CIC2018 EXP47·ToN EXP48 모두 완료.')
        text = text.replace('ToN도 같은 정제로 전체 구조를 재검증한다.', 'ToN도 같은 정제로 전체 구조를 재검증했다.')
    else:
        state = {'running': '실행 중', 'preparing': '준비 중', 'error': '오류 확인 필요', 'stopped': '중단'}.get(ton['state'], ton['state'])
        line = f"{state}. 모순 제거 후 {ton['clean']['rows_after']:,}행 유지, test {ton['clean']['split_rows']['test']:,}행. 성능은 완료 후 기록한다."
    status_html = f'<div class="status-cell warn"><p class="label">ToN · EXP48</p><p class="value">{line}</p></div>'
    text = re.sub(r'(?<=<!-- CLEAN_REVALIDATION_STATUS -->).*?(?=<!-- /CLEAN_REVALIDATION_STATUS -->)',
                  status_html, text, flags=re.S)
    text = re.sub(r'<p class="lede">09-15 미팅 후.*?</p>',
                  '<p class="lede">동일 벡터에 상반 라벨이 있는 그룹의 모든 행을 제거한 뒤 기존 구조를 다시 학습했다. '
                  '<strong>CIC2018은 호출·승인 0건, ToN은 호출 288,684건·승인 271,715건</strong>이다. '
                  '보고서는 두 데이터를 같은 순서로 제시한다: <strong>정제 후 클래스별 train/test 수 → expert 구성 → 호출·승인 결과와 병목</strong>.</p>',
                  text, flags=re.S)
    text = re.sub(r'(<button class="jump" data-jump="clean_revalidation_0916"><h3>).*?(</h3></button><p>).*?(</p>)',
                  r'\g<1>동일벡터상반라벨 제거 후 모델 병목 지점 관찰\g<2>'
                  r'CIC2018·ToN 각각의 데이터 규모, 공통·전용 expert 사례 구성, scorer 추천부터 verifier 승인까지의 경로를 같은 형식으로 확인한다.\g<3>',
                  text, flags=re.S)
    index.write_text(text)
    print(f'Built {OUTPUT}; ToN state={ton["state"]}. Run sync_html_reports.py next.')


if __name__ == '__main__':
    main()
