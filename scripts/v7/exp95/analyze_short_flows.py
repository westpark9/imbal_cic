"""Profile flows of two packets or fewer and split the side-column gain by class; no model is fitted.

Run from the repository root: python scripts/v7/exp95/analyze_short_flows.py OUTPUT.json
"""
import json
import sys

import numpy as np

RUN = 'tabpfn/results/v7/exp95/20261106_051907_all_class_0ecc759e'
SUMMARY = 'docs/lablog/evidence/20261106/exp95_drop_fewshot_summary.json'
NAMES = ['benign', 'dos', 'portscan', 'ddos', 'bf', 'bot', 'web', 'inf', 'heartbleed']
TCP, PURE_SYN = 6, 2  # token 5 is a TCP flag only on TCP rows


def tcp_profile(mask, a):
    t = mask & (a['proto'] == TCP)
    return dict(tcp=int(t.sum()), pure_syn_first=int((a['fl'][t] == PURE_SYN).sum()), peer_rst=int((a['brst'][t] > 0).sum()),
                syn_then_peer_rst=int(((a['fl'][t] == PURE_SYN) & (a['brst'][t] > 0)).sum()), own_rst=int((a['frst'][t] > 0).sum()))


def counts(values):
    return {int(k): int(v) for k, v in zip(*np.unique(values, return_counts=True))}


def main(output):
    arrays = np.load(f'{RUN}/base/arrays.npz')
    row = np.load(f'{RUN}/side.npz')['row']
    y = np.load(f'{RUN}/y9.npy')[row]
    a = {k: arrays[k][row] for k in ('packets', 'fl', 'proto', 'frst', 'brst', 'L2')}
    labelled = y >= 0
    short = labelled & (a['packets'] <= 2)

    classes = []
    for c, name in enumerate(NAMES):
        members = y == c
        classes.append(dict(name=name, flows=int(members.sum()), le2=int((members & short).sum()), le2_pct=round(100 * (members & short).sum() / members.sum(), 1),
                            le2_protocol=counts(a['proto'][members & short])))

    idx = np.flatnonzero(labelled)
    order = np.argsort(a['L2'][idx], kind='stable')
    keys, labels = a['L2'][idx][order], y[idx][order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1, len(keys)]
    conflict = np.zeros(len(y), bool)
    groups = 0
    for s, e in zip(starts[:-1], starts[1:]):
        if labels[s] != labels[e - 1] or len(np.unique(labels[s:e])) > 1:
            conflict[idx[order[s:e]]] = True
            groups += 1

    table = {(r['arm'], r['method'], r['budget']): r for r in json.load(open(SUMMARY, encoding='utf-8'))['all_class_table']}
    base, side = table['netfound', 'tabpfn', 50]['class_f1'], table['netfound+ports', 'tabpfn', 50]['class_f1']
    total = sum(side[c] - base[c] for c in NAMES)

    json.dump(dict(
        schema=1, run=RUN, scope='Counts from saved arrays and the saved class F1 table. No model was fitted.',
        labelled=int(labelled.sum()), le2=int(short.sum()), le2_pct=round(100 * short.sum() / labelled.sum(), 1), classes=classes,
        conflict=dict(rows=int(conflict.sum()), groups=groups, le2=int((conflict & short).sum()), le2_pct=round(100 * (conflict & short).sum() / conflict.sum(), 1),
                      packets=counts(a['packets'][conflict]),
                      portscan=dict(rows=int((conflict & (y == 2)).sum()), **tcp_profile(conflict & (y == 2), a)),
                      benign=dict(rows=int((conflict & (y == 0)).sum()), packets=counts(a['packets'][conflict & (y == 0)]), **tcp_profile(conflict & (y == 0), a))),
        le2_portscan=tcp_profile(short & (y == 2), a), le2_benign=tcp_profile(short & (y == 0), a),
        ports_gain_tabpfn_n50=dict(macro_pp=round(100 * total / len(NAMES), 2), by_class={c: dict(f1_pp=round(100 * (side[c] - base[c]), 2), share_pct=round(100 * (side[c] - base[c]) / total, 1)) for c in NAMES}),
    ), open(output, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
