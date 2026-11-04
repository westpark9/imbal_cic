"""EXP94 PCAP side: whole-network CIC2017 captures -> flows -> three-state labels
-> time-forward split -> two views (flow statistics, frozen netFound) -> audits,
oracle ceilings and paired few-shot curves.

Differences from EXP93 (``scripts/v7/exp93/run_pcap_benchmark.py``):

* Labels have three states.  ``confirmed`` flows satisfy window, endpoint and
  the rule's evidence condition.  ``unresolved`` flows satisfy window and
  endpoint only (Attempted categories, or evidence condition not met); they
  are kept in the corpus with ``label=None`` and never mapped to benign.
  ``background`` flows match no rule endpoint/window and become benign.
* The split is chronological within each capture and family (first 70% of a
  family's flows by start time -> train, next 15% validation, last 15% test),
  not a 5-tuple hash.  Every split keeps every flow unless a train cap is set.
* Every flow also gets a ``flowstats-v1`` view: CICFlowMeter-like statistics
  computed from all packets of the flow, written as a second adapter so XGB and
  TabPFN can be run on statistics and on the netFound embedding over the same
  flow IDs, split and labels (the EXP92 2x2 plan).
* The identical-input audit has three layers (full flow signature, tokenizer
  output, projected vector) and reports the deterministic oracle ceiling per
  layer on the test split.

The netFound encoder, PCA projection and the paired few-shot loop are reused
unchanged from EXP93 so the two experiments stay comparable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'scripts/common/experiment_paths.py').is_file())
for rel in ('scripts/common', 'scripts/v7/exp92', 'scripts/v7/exp93'):
    sys.path.insert(0, str(ROOT / rel))
from experiment_paths import resolve_path
from fetch_assets import sha256
import flows as flowlib
from run_pcap_benchmark import CorrectedRuleSet, _f1, encode_netfound, fewshot, read, write

import numpy as np


# ----------------------------------------------------------------------------
# Flow statistics view
# ----------------------------------------------------------------------------
class _Stat:
    __slots__ = ('n', 's', 'ss', 'lo', 'hi')

    def __init__(self):
        self.n = 0; self.s = 0.0; self.ss = 0.0; self.lo = math.inf; self.hi = -math.inf

    def add(self, v: float) -> None:
        self.n += 1; self.s += v; self.ss += v * v
        if v < self.lo: self.lo = v
        if v > self.hi: self.hi = v

    def values(self) -> list[float]:
        if not self.n:
            return [0.0, 0.0, 0.0, 0.0]
        mean = self.s / self.n
        var = max(self.ss / self.n - mean * mean, 0.0)
        return [mean, math.sqrt(var), self.lo, self.hi]


STATS_COLUMNS = (
    ['duration_us', 'fwd_packets', 'bwd_packets', 'fwd_payload_bytes', 'bwd_payload_bytes', 'fwd_header_bytes', 'bwd_header_bytes']
    + [f'fwd_len_{k}' for k in ('mean', 'std', 'min', 'max')] + [f'bwd_len_{k}' for k in ('mean', 'std', 'min', 'max')]
    + [f'flow_iat_{k}' for k in ('mean', 'std', 'min', 'max')] + [f'fwd_iat_{k}' for k in ('mean', 'std', 'min', 'max')]
    + [f'bwd_iat_{k}' for k in ('mean', 'std', 'min', 'max')]
    + ['fin_count', 'syn_count', 'rst_count', 'psh_count', 'ack_count', 'urg_count', 'fwd_psh', 'bwd_psh', 'fwd_urg', 'bwd_urg']
    + ['init_win_fwd', 'init_win_bwd', 'down_up_ratio', 'avg_packet_size', 'bytes_per_s', 'packets_per_s', 'protocol', 'dst_port']
)


class StatsFlow(flowlib.Flow):
    """EXP92 flow plus incrementally computed CICFlowMeter-like statistics."""

    def __post_init__(self):
        super().__post_init__()
        self.len_stat = [_Stat(), _Stat()]
        self.iat_stat = [_Stat(), _Stat()]
        self.flow_iat = _Stat()
        self.header_bytes = [0, 0]
        self.flag_counts = Counter()
        self.dir_psh = [0, 0]; self.dir_urg = [0, 0]
        self.init_win = [None, None]
        self.payload_max = [0, 0]
        self.last_pkt_time = None
        self.last_dir_pkt_time = [None, None]

    def add(self, p):
        d = int((p.src, p.sport) != (self.first.src, self.first.sport))
        size = p.ip_header[2]
        self.len_stat[d].add(float(size))
        if p.payload_len > self.payload_max[d]: self.payload_max[d] = p.payload_len
        if self.last_pkt_time is not None:
            self.flow_iat.add((p.ns - self.last_pkt_time) / 1000.0)
        if self.last_dir_pkt_time[d] is not None:
            self.iat_stat[d].add((p.ns - self.last_dir_pkt_time[d]) / 1000.0)
        self.last_pkt_time = p.ns; self.last_dir_pkt_time[d] = p.ns
        self.header_bytes[d] += p.ip_header[0] + (20 if p.proto == 6 else 8)
        if p.proto == 6:
            for bit, name in ((1, 'fin'), (2, 'syn'), (4, 'rst'), (8, 'psh'), (16, 'ack'), (32, 'urg')):
                if p.flags & bit:
                    self.flag_counts[name] += 1
            if p.flags & 8: self.dir_psh[d] += 1
            if p.flags & 32: self.dir_urg[d] += 1
            if self.init_win[d] is None: self.init_win[d] = p.window
        super().add(p)

    def stats(self) -> list[float]:
        duration = (self.end - self.start) / 1000.0
        total_bytes = self.payload[0] + self.payload[1]
        seconds = duration / 1e6
        return ([duration, self.direction_count[0], self.direction_count[1], self.payload[0], self.payload[1],
                 self.header_bytes[0], self.header_bytes[1]]
                + self.len_stat[0].values() + self.len_stat[1].values()
                + self.flow_iat.values() + self.iat_stat[0].values() + self.iat_stat[1].values()
                + [self.flag_counts[k] for k in ('fin', 'syn', 'rst', 'psh', 'ack', 'urg')]
                + [self.dir_psh[0], self.dir_psh[1], self.dir_urg[0], self.dir_urg[1]]
                + [self.init_win[0] or 0, self.init_win[1] or 0,
                   self.direction_count[1] / self.direction_count[0] if self.direction_count[0] else 0.0,
                   (self.len_stat[0].s + self.len_stat[1].s) / self.count if self.count else 0.0,
                   total_bytes / seconds if seconds > 0 else 0.0,
                   self.count / seconds if seconds > 0 else 0.0,
                   self.first.proto, self.first.dport])

    def row(self, reason):
        r = super().row(reason)
        r['stats'] = self.stats()
        r['fwd_payload_max'], r['bwd_payload_max'] = self.payload_max
        return r


def iter_stats_flows(path, capture, group, stats, idle_seconds=120, active_seconds=120, reorder_seconds=1, allow_truncated=False):
    """Copy of ``flows.iter_flows`` that builds ``StatsFlow`` objects (EXP92 code is left untouched)."""
    active, expiry, tombstone = {}, [], {}
    idle_ns, active_ns = int(idle_seconds * 1e9), int(active_seconds * 1e9)
    for p in flowlib.packets(path, stats, reorder_seconds, allow_truncated):
        while expiry and expiry[0][0] < p.ns:
            _, key, start, version = heapq.heappop(expiry)
            f = active.get(key)
            if f is not None and f.start == start and f.count == version:
                yield f.row('timeout'); del active[key]
        key, f = p.key, active.get(p.key)
        syn = p.proto == 6 and p.flags & 2 and not p.flags & 16
        if f and ((syn and (f.first.seq != p.seq or f.fin)) or p.ns - f.start > active_ns):
            yield f.row('new_syn' if syn else 'active_timeout'); del active[key]; f = None
        if f is None:
            if p.proto == 6 and p.flags == 16 and p.payload_len == 0 and p.ns <= tombstone.get(key, -1):
                stats['closed_session_ack_tail'] += 1; continue
            f = active[key] = StatsFlow(p, capture, group)
        f.add(p)
        closed = p.proto == 6 and (p.flags & 4 or (len(f.fin) == 2 and p.flags & 16 and not p.flags & 1 and not p.payload_len))
        if closed:
            yield f.row('rst' if p.flags & 4 else 'fin_handshake')
            del active[key]; tombstone[key] = p.ns + 2_000_000_000
        else:
            heapq.heappush(expiry, (min(f.start + active_ns, f.end + idle_ns), key, f.start, f.count))
        if len(expiry) > max(4096, 4 * len(active)):
            expiry = [(min(v.start + active_ns, v.end + idle_ns), k, v.start, v.count) for k, v in active.items()]
            heapq.heapify(expiry)
        if stats['raw_packets'] % 10000 == 0:
            tombstone = {k: v for k, v in tombstone.items() if v >= p.ns}
        stats['peak_active_flows'] = max(stats['peak_active_flows'], len(active))
    for f in active.values():
        row = f.row('capture_end')
        row['capture_truncated'] = bool(stats['truncated_capture_tail'])
        yield row


# ----------------------------------------------------------------------------
# Three-state labels
# ----------------------------------------------------------------------------
class EvidenceRuleSet(CorrectedRuleSet):
    """EXP93 matcher with an explicit ``unresolved`` state.

    * attempted rule fits            -> unresolved_attempted (label None)
    * endpoint+window fit, condition fails -> unresolved_no_evidence (label None)
    * confirmed rule fits            -> confirmed_attack
    * crosses a window boundary      -> boundary (label None)
    * nothing fits                   -> background benign
    """

    EXTRA_FIELDS = ('max_peer_payload_bytes', 'min_peer_packets', 'max_peer_packets',
                    'exclude_attack_payload_max_eq', 'exclude_attack_payload_total_in')

    @staticmethod
    def _condition(row, rule, reverse):
        """EXP93 condition plus peer-side bounds and attack-side exclusions (CNS2022 2017 notebook)."""
        if not CorrectedRuleSet._condition(row, rule, reverse):
            return False
        attack, peer = ('bwd_', 'fwd_') if reverse else ('fwd_', 'bwd_')
        c = rule
        if c.get('max_peer_payload_bytes') is not None and row[peer + 'payload_bytes'] > c['max_peer_payload_bytes']:
            return False
        if c.get('min_peer_packets') is not None and row[peer + 'packets'] < c['min_peer_packets']:
            return False
        if c.get('max_peer_packets') is not None and row[peer + 'packets'] > c['max_peer_packets']:
            return False
        return True

    @staticmethod
    def _excluded(row, rule, reverse) -> bool:
        attack = 'bwd_' if reverse else 'fwd_'
        if rule.get('exclude_attack_payload_max_eq') is not None and row.get(attack + 'payload_max') == rule['exclude_attack_payload_max_eq']:
            return True
        if rule.get('exclude_attack_payload_total_in') and row[attack + 'payload_bytes'] in set(rule['exclude_attack_payload_total_in']):
            return True
        return False

    def apply(self, row):
        fits, boundary, near = [], [], []
        for rule in self.rules:
            ok, reverse = self._endpoint_match(row, rule)
            if not ok:
                continue
            start, end = int(rule['start_ns']), int(rule['end_ns'])
            if row['end_ns'] < start or row['start_ns'] > end:
                continue
            if row['start_ns'] < start or row['end_ns'] > end:
                boundary.append(rule['id']); continue
            if self._excluded(row, rule, reverse):
                # The notebook's explicit exclusions name benign background
                # inside an attack window; such a flow is simply not matched.
                continue
            if self._condition(row, rule, reverse):
                fits.append(rule)
            else:
                near.append(rule['id'])
        confirmed = [r for r in fits if not r.get('attempted', False)]
        attempted = [r for r in fits if r.get('attempted', False)]
        labels = {r['label'] for r in confirmed if r.get('label')}
        src = lambda rs: sorted({r.get('source', self.default_source) for r in rs})
        if len(labels) > 1:
            return dict(label=None, annotation_label=None, label_status='conflict', label_rule_ids=[r['id'] for r in confirmed], label_sources=src(confirmed))
        if confirmed:
            # Attempted rules carry higher priority in the protocol; if one fits
            # together with a confirmed rule, the attempted annotation wins
            # because it encodes the authors' more specific finding.
            if attempted and max(int(r.get('priority', 0)) for r in attempted) > max(int(r.get('priority', 0)) for r in confirmed):
                rule = attempted[0]
                return dict(label=None, annotation_label=rule.get('annotation'), label_status='unresolved_attempted',
                            label_rule_ids=[rule['id']], label_sources=src([rule]))
            rule = confirmed[0]
            return dict(label=rule['label'], annotation_label=rule.get('annotation', rule['label']), label_status='confirmed_attack',
                        label_rule_ids=[r['id'] for r in confirmed], label_sources=src(confirmed))
        if attempted:
            rule = attempted[0]
            return dict(label=None, annotation_label=rule.get('annotation'), label_status='unresolved_attempted',
                        label_rule_ids=[r['id'] for r in attempted], label_sources=src(attempted))
        if near:
            return dict(label=None, annotation_label=None, label_status='unresolved_no_evidence', label_rule_ids=near, label_sources=[self.default_source])
        if boundary:
            return dict(label=None, annotation_label=None, label_status='boundary', label_rule_ids=boundary, label_sources=[self.default_source])
        if self.background:
            return dict(label=self.background, annotation_label='background', label_status='background_benign', label_rule_ids=[], label_sources=[self.default_source])
        return dict(label=None, annotation_label=None, label_status='unmatched', label_rule_ids=[], label_sources=[self.default_source])


# ----------------------------------------------------------------------------
# Corpus: two passes (stream flows, then assign a time-forward split)
# ----------------------------------------------------------------------------
def _fam(config: dict[str, Any], annotation: str | None, label: str | None) -> str | None:
    """Rules may carry a raw annotation; map it through family_mapping when present."""
    mapping = config.get('family_mapping') or {}
    if label in config['class_order']:
        return label
    if annotation in mapping:
        return mapping[annotation]
    if label in mapping:
        return mapping[label]
    return None


def build_corpus(config: dict[str, Any], out: Path) -> dict[str, Any]:
    corpus = out / 'corpus'
    manifest_path = corpus / 'manifest.json'
    if manifest_path.exists():
        return read(manifest_path)
    corpus.mkdir(parents=True, exist_ok=False)
    fractions = config['split']['fractions']
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError('split fractions must sum to one')
    source = config['label_source']
    raw = corpus / '.pass1.jsonl.tmp'
    index: list[tuple[str, str, int, int]] = []   # (capture, family-or-None, start_ns, byte offset)
    status_counts: Counter = Counter()
    capture_meta, raw_stats = [], {}
    with raw.open('w') as sink:
        for cap_cfg in config['captures']:
            pcap = resolve_path(cap_cfg['path'], ROOT)
            if not pcap.exists():
                raise FileNotFoundError(f'Missing PCAP: {pcap}')
            engine = EvidenceRuleSet(cap_cfg, source)
            stats: Counter = Counter()
            n, next_progress = 0, 50000
            print(f'corpus: {cap_cfg["id"]} ({pcap.name})', flush=True)
            for row in iter_stats_flows(pcap, cap_cfg['id'], cap_cfg['group_namespace'], stats, **config['flow']):
                n += 1
                labelled = engine.apply(row)
                if row.get('capture_truncated'):
                    labelled = dict(label=None, annotation_label=None, label_status='incomplete_capture', label_rule_ids=[], label_sources=[source])
                row.update(labelled)
                row['family'] = (_fam(config, row['annotation_label'], row['label'])
                                 if row['label_status'] in ('confirmed_attack', 'background_benign') else None)
                if row['label_status'] == 'confirmed_attack' and row['family'] is None:
                    raise ValueError(f'Confirmed label without family mapping: {row["label"]} / {row["annotation_label"]}')
                row['label'] = row['family'] if row['label_status'] in ('confirmed_attack', 'background_benign') else None
                status_counts[row['label_status']] += 1
                index.append((cap_cfg['id'], row['label'] or '__UNRESOLVED__', int(row['start_ns']), sink.tell()))
                sink.write(json.dumps(row, separators=(',', ':')) + '\n')
                if n >= next_progress:
                    print(f'corpus: {cap_cfg["id"]}: {n:,} flows; {stats["raw_packets"]:,} packets', flush=True)
                    next_progress += 50000
            raw_stats[cap_cfg['id']] = dict(stats)
            capture_meta.append(dict(id=cap_cfg['id'], path=cap_cfg['path'], bytes=pcap.stat().st_size, sha256=sha256(pcap),
                                     group_namespace=cap_cfg['group_namespace'], flows=n))
            print(f'corpus: complete {cap_cfg["id"]}: {n:,} flows', flush=True)
    # time-forward split within (capture, family); unresolved rows follow the same rule so they can be audited per split
    split_of: dict[int, str] = {}
    by_key: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for cap, fam, start, off in index:
        by_key[(cap, fam)].append((start, off))
    for key, items in by_key.items():
        items.sort()
        a, b = int(len(items) * fractions[0]), int(len(items) * (fractions[0] + fractions[1]))
        for k, (_, off) in enumerate(items):
            split_of[off] = 'train' if k < a else 'validation' if k < b else 'test'
    cap_cfg_policy = config['corpus'].get('retention_per_label_split') or {}
    rng = np.random.default_rng(config['seed'])
    keep: dict[int, bool] = {}
    retained: Counter = Counter(); source_counts: Counter = Counter()
    for key, items in by_key.items():
        for off_split in ('train', 'validation', 'test'):
            offs = [off for _, off in items if split_of[off] == off_split]
            source_counts[(key[1], off_split)] += len(offs)
            part = cap_cfg_policy.get(off_split, {})
            cap = part.get('labelled' if key[1] != '__UNRESOLVED__' else 'unresolved')
            chosen = offs if cap is None or len(offs) <= int(cap) else sorted(rng.choice(offs, int(cap), replace=False).tolist())
            for off in chosen:
                keep[off] = True
            retained[(key[1], off_split)] += len(chosen)
    flows_path = corpus / 'flows.jsonl'
    with raw.open() as src_f, flows_path.open('w') as dst:
        while True:
            off = src_f.tell()
            line = src_f.readline()
            if not line:
                break
            if not keep.get(off):
                continue
            row = json.loads(line)
            row['split'] = split_of[off]
            dst.write(json.dumps(row, separators=(',', ':')) + '\n')
    raw.unlink()
    manifest = dict(schema=2, dataset=config['dataset'], seed=config['seed'], captures=capture_meta, raw_stats=raw_stats,
                    label_source=source, label_rule_note=config.get('label_rule_note'), label_status_counts=dict(status_counts),
                    label_states=dict(confirmed='window + endpoint + evidence condition -> family label',
                                      unresolved='window + endpoint, no evidence (attempted categories or failed condition); label None, never benign',
                                      background='no rule endpoint/window -> benign', boundary='crosses a rule window; label None'),
                    class_order=config['class_order'], family_mapping=config.get('family_mapping'),
                    split=dict(method='time_forward_within_capture_and_family', fractions=fractions,
                               note='flows of each (capture, family) sorted by start time; first 70% train, next 15% validation, last 15% test'),
                    source_counts=[dict(label=k[0], split=k[1], rows=v) for k, v in sorted(source_counts.items())],
                    retained_counts=[dict(label=k[0], split=k[1], rows=v) for k, v in sorted(retained.items())],
                    retention_per_label_split=cap_cfg_policy or None, flows_sha256=sha256(flows_path),
                    stats_columns=STATS_COLUMNS)
    write(manifest_path, manifest)
    return manifest


# ----------------------------------------------------------------------------
# Flow-statistics adapter: dataset.npz in the EXP93 layout so ``fewshot`` can run on it
# ----------------------------------------------------------------------------
def write_stats_view(config: dict[str, Any], corpus: Path, out: Path) -> Path:
    adapter = 'flowstats-v1'
    target = out / 'encoders' / adapter
    target.mkdir(parents=True, exist_ok=True)
    if (target / 'dataset.npz').exists():
        return target
    rows = [json.loads(l) for l in (corpus / 'flows.jsonl').open()]
    names = list(config['class_order'])
    X = np.asarray([r['stats'] for r in rows], dtype=np.float32)
    if not np.isfinite(X).all():
        raise FloatingPointError('non-finite flow statistics')
    y = np.asarray([names.index(r['label']) if r['label'] in names else -1 for r in rows], dtype=np.int16)
    split = np.asarray([r['split'] for r in rows]); ids = np.asarray([r['flow_id'] for r in rows])
    np.savez_compressed(target / 'dataset.npz', X=X, y=y, split=split, flow_id=ids, class_names=np.asarray(names))
    write(target / 'encoder.json', dict(adapter=adapter, rows=len(rows), columns=STATS_COLUMNS, feature_count=len(STATS_COLUMNS),
                                        corpus_sha256=read(corpus / 'manifest.json')['flows_sha256'],
                                        note='CICFlowMeter-like per-flow statistics from every packet of the flow; no PCA; not a reproduction of CICFlowMeter rows'))
    write(target / 'projection.json', dict(adapter=adapter, rows=len(rows), input_dimensions=X.shape[1], pca_dimensions=None,
                                           classifier_evaluated=False))
    return target


# ----------------------------------------------------------------------------
# Three-layer identical-input audit and oracle ceilings
# ----------------------------------------------------------------------------
def _groups(keys: list[str], idx: np.ndarray) -> dict[str, list[int]]:
    g: dict[str, list[int]] = defaultdict(list)
    for i in idx:
        g[keys[i]].append(int(i))
    return g


def _layer(keys: list[str], rows: list[dict], names: list[str], idx: np.ndarray) -> dict[str, Any]:
    groups = _groups(keys, idx)
    conflicts = {d: m for d, m in groups.items() if len({rows[i]['label'] for i in m if rows[i]['label'] in names}) > 1}
    by_family: Counter = Counter()
    for m in conflicts.values():
        for i in m:
            if rows[i]['label'] in names:
                by_family[rows[i]['label']] += 1
    return dict(rows=int(len(idx)), identical_groups=sum(len(m) > 1 for m in groups.values()),
                cross_label_groups=len(conflicts), cross_label_rows=sum(len(m) for m in conflicts.values()),
                cross_label_rows_by_family=dict(by_family),
                examples=[dict(key=d, rows=len(m), labels=dict(Counter(rows[i]['label'] for i in m)), flow_ids=[rows[i]['flow_id'] for i in m[:10]])
                          for d, m in sorted(conflicts.items(), key=lambda kv: -len(kv[1]))[:50]])


def _oracle(keys: list[str], rows: list[dict], names: list[str], test_idx: np.ndarray, classes: list[int]) -> dict[str, Any]:
    y = np.asarray([names.index(rows[i]['label']) for i in test_idx], dtype=np.int16)
    mask = np.isin(y, classes)
    groups = _groups(keys, test_idx[mask])
    pos = {int(i): k for k, i in enumerate(test_idx[mask])}
    pred = np.empty(int(mask.sum()), dtype=np.int16)
    for m in groups.values():
        maj = Counter(names.index(rows[i]['label']) for i in m).most_common(1)[0][0]
        for i in m:
            pred[pos[i]] = maj
    r = _f1(y[mask], pred, classes, names)
    return dict(macro_f1=r['macro_f1'], classes=[(c['name'], c['f1']) for c in r['classes']])


def project_view(config: dict[str, Any], corpus: Path, encoder_dir: Path) -> dict[str, Any]:
    """EXP93 ``project`` without the 5-tuple group-separation assertion.

    The EXP94 split is chronological within (capture, family), so one 5-tuple
    group legitimately spans train and test; the overlap is recorded, not
    forbidden.  PCA is still fitted on training rows only.
    """
    from sklearn.decomposition import PCA
    result_file = encoder_dir / 'projection.json'
    if result_file.exists():
        return read(result_file)
    rows = [json.loads(l) for l in (corpus / 'flows.jsonl').open()]
    info = read(encoder_dir / 'encoder.json')
    x = np.load(encoder_dir / 'embeddings.npy', mmap_mode='r')
    if sha256(encoder_dir / 'embeddings.npy') != info['embeddings_sha256']:
        raise RuntimeError('Embedding artifact changed after encoding')
    ids = np.asarray([r['flow_id'] for r in rows]); split = np.asarray([r['split'] for r in rows])
    names = list(config['class_order'])
    y = np.asarray([names.index(r['label']) if r['label'] in names else -1 for r in rows], dtype=np.int16)
    train = split == 'train'
    dim = min(int(config['projection']['components']), int(train.sum()) - 1, x.shape[1])
    pca = PCA(n_components=dim, svd_solver='full', random_state=config['seed']).fit(x[train])
    z = pca.transform(x).astype('float32')
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite projection')
    np.savez_compressed(encoder_dir / 'pca.npz', mean=pca.mean_, components=pca.components_,
                        explained_variance_ratio=pca.explained_variance_ratio_, fit_flow_ids=ids[train])
    np.savez_compressed(encoder_dir / 'dataset.npz', X=z, y=y, split=split, flow_id=ids, class_names=np.asarray(names))
    groups = {part: {r['group_id'] for r in rows if r['split'] == part} for part in ('train', 'validation', 'test')}
    result = dict(adapter=info['adapter'], rows=len(rows), input_dimensions=int(x.shape[1]), pca_dimensions=dim,
                  pca_fit_rows=int(train.sum()), pca_fit_scope='all corpus train rows including unresolved; no validation/test',
                  pca_explained_variance=float(pca.explained_variance_ratio_.sum()), class_order=names,
                  split_counts={k: int((split == k).sum()) for k in groups},
                  labelled_counts={part: {name: int(((split == part) & (y == i)).sum()) for i, name in enumerate(names)} for part in groups},
                  five_tuple_groups_shared_train_test=len(groups['train'] & groups['test']),
                  split_note='time-forward within capture and family; shared 5-tuple groups across splits are expected and reported',
                  classifier_evaluated=False)
    write(result_file, result)
    return result


def coverage_filter(config: dict[str, Any], corpus: Path, out: Path) -> list[str]:
    """Keep only families with >= max budget train flows and >= 1 test flow; record the rest instead of failing."""
    rows = [json.loads(l) for l in (corpus / 'flows.jsonl').open()]
    max_budget = max(int(v) for v in config['fewshot']['budgets'])
    cov, included, excluded = {}, [], []
    for fam in config['class_order']:
        c = Counter(r['split'] for r in rows if r.get('label') == fam)
        cov[fam] = dict(train=int(c['train']), validation=int(c['validation']), test=int(c['test']))
        (included if c['train'] >= max_budget and c['test'] >= 1 else excluded).append(fam)
    write(out / 'label_coverage.json', dict(passed=bool(included), max_fewshot_budget=max_budget, coverage=cov,
                                            included=included, excluded=excluded,
                                            rule='families below the budget are excluded from training and macro-F1; their flows stay in the corpus with label kept for audits'))
    if not included:
        raise ValueError('No family satisfies the few-shot coverage rule')
    return included


def audit3(config: dict[str, Any], corpus: Path, out: Path, netfound_dir: Path | None) -> dict[str, Any]:
    rows = [json.loads(l) for l in (corpus / 'flows.jsonl').open()]
    names = list(config['class_order'])
    labelled = np.asarray([i for i, r in enumerate(rows) if r['label'] in names])
    test = np.asarray([i for i in labelled if rows[i]['split'] == 'test'])
    train_keys_cache: dict[str, set] = {}
    layers: dict[str, list[str]] = {}
    # L1: full flow signature = statistics + full bounded token sequence (what the pipeline knows about the flow)
    layers['L1_flow_signature'] = [hashlib.sha256(json.dumps({k: r[k] for k in ('stats', 'burst_tokens', 'directions', 'bytes', 'iats', 'counts', 'protocol')},
                                                            sort_keys=True, separators=(',', ':')).encode()).hexdigest() for r in rows]
    # L2: netFound tokenizer input as the model sees it (ms iats, truncated bursts; flow_duration is not a model input)
    def tok(r):
        return json.dumps(dict(b=[b[:6 * 19] for b in r['burst_tokens'][:12]], d=r['directions'][:12], y=r['bytes'][:12],
                               i=[int(v * 1e-3) for v in r['iats'][:12]], c=r['counts'][:12], p=r['protocol']), separators=(',', ':'))
    layers['L2_tokenizer_input'] = [hashlib.sha256(tok(r).encode()).hexdigest() for r in rows]
    layers['L3_flowstats_vector'] = [hashlib.sha256(np.asarray(r['stats'], dtype=np.float32).tobytes()).hexdigest() for r in rows]
    if netfound_dir and (netfound_dir / 'dataset.npz').exists():
        with np.load(netfound_dir / 'dataset.npz') as z:
            X = z['X']; assert len(X) == len(rows)
        layers['L3_netfound_pca_vector'] = [hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in X]
    result: dict[str, Any] = dict(schema=1, rows=len(rows), labelled_rows=int(len(labelled)), test_rows=int(len(test)), layers={})
    included = list(range(len(names)))
    for name, keys in layers.items():
        train_set = {keys[i] for i in labelled if rows[i]['split'] == 'train'}
        dup = np.asarray([keys[i] in train_set for i in test])
        result['layers'][name] = dict(
            full_labelled=_layer(keys, rows, names, labelled), test=_layer(keys, rows, names, test),
            train_to_test_exact_duplicate_fraction=float(dup.mean()) if len(dup) else None,
            oracle_ceiling={regime: _oracle(keys, rows, names, test, included if regime == 'with_benign' else [c for c in included if names[c] != 'benign'])
                            for regime in config['fewshot']['regimes']})
    result['interpretation'] = ('L1 counts flows the pipeline cannot tell apart at all (label-side floor). The increase from L1 to L2/L3 is '
                                'information discarded by the representation. Oracle ceiling = majority label per identical group on test.')
    write(out / 'audit3.json', result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--stage', choices=['corpus', 'views', 'audit', 'fewshot', 'all'], default='all')
    p.add_argument('--views', default='flowstats-v1,netfound-v2-small')
    p.add_argument('--device', default='cuda'); p.add_argument('--batch-size', type=int, default=32)
    a = p.parse_args()
    config = read(resolve_path(a.config, ROOT))
    if 'label_rules_file' in config:
        rules = read(resolve_path(config['label_rules_file'], ROOT))
        config['captures'] = rules['captures']
        config.setdefault('family_mapping', rules.get('family_mapping'))
        config.setdefault('label_source', rules.get('label_source'))
    out = resolve_path(a.output, ROOT); out.mkdir(parents=True, exist_ok=True)
    write(out / 'protocol_snapshot.json', config)
    corpus = out / 'corpus'
    if a.stage in ('corpus', 'all'):
        build_corpus(config, out)
    config['class_order'] = coverage_filter(config, corpus, out)
    views = a.views.split(',')
    dirs: dict[str, Path] = {}
    if a.stage in ('views', 'all'):
        if 'flowstats-v1' in views:
            dirs['flowstats-v1'] = write_stats_view(config, corpus, out)
        if 'netfound-v2-small' in views:
            d = encode_netfound(config, corpus, out, a.device, a.batch_size)
            project_view(config, corpus, d); dirs['netfound-v2-small'] = d
    for v in views:
        dirs.setdefault(v, out / 'encoders' / v)
    if a.stage in ('audit', 'all'):
        audit3(config, corpus, out, dirs.get('netfound-v2-small'))
    if a.stage in ('fewshot', 'all'):
        for v in views:
            fewshot(config, dirs[v], out, v)
    write(out / 'status.json', dict(status='complete', stage=a.stage, views=views))


if __name__ == '__main__':
    main()
