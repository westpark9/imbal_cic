"""EXP95: behavioural side features on all labelled flows, and residual-flow handling.

Part 1 tests the slide 16 side features without few-shot sampling: the same lookup
on every labelled flow, the same lookup between scan sessions, and XGB trained on
every labelled train flow.  Part 2 compares four treatments of residual flows with
the EXP94 few-shot protocol and tracks how often netFound inputs stay identical.

Flow IDs, labels, splits, seeds, budgets and classifiers come from the EXP94 base
run, which is only read.  ``merge`` replays the captures with the EXP94 assembly
and appends each residual packet to its root parent.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
import gc
import hashlib
import heapq
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
from typing import Any

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/common/experiment_paths.py').is_file())
for rel in ('scripts/common', 'scripts/v7/exp92', 'scripts/v7/exp93', 'scripts/v7/exp94'):
    sys.path.insert(0, str(ROOT / rel))
from experiment_paths import resolve_path
import flows as flowlib
from netfound_encoder import NetFoundEncoder
from run_pcap_benchmark import _f1, _make_model, _predict_proba_batched, _timed, _weights_digest, read, write
from run_pcap_cic2017 import StatsFlow

import numpy as np
import pandas as pd

SEQ = ['first_seq_zero', 'first_ack_zero']
CTX_BASE = ['cli_flows', 'cli_targets', 'cli_fail', 'pair_flows', 'pair_ports', 'pair_fail', 'srv_flows', 'srv_clients', 'srv_fail', 'sig_rep']
TAIL = ['tail_packets', 'tail_rst', 'tail_fin']
GAP_S = 2.0


def h64(b: bytes) -> int:
    return int.from_bytes(hashlib.sha256(b).digest()[:8], 'little')


def keys_of(r: dict[str, Any]) -> tuple[int, int, int]:
    """The EXP94 L1, L2 and L3-statistics identity keys."""
    l1 = json.dumps({k: r[k] for k in ('stats', 'burst_tokens', 'directions', 'bytes', 'iats', 'counts', 'protocol')}, sort_keys=True, separators=(',', ':'))
    l2 = json.dumps(dict(b=[b[:6 * 19] for b in r['burst_tokens'][:12]], d=r['directions'][:12], y=r['bytes'][:12],
                         i=[int(v * 1e-3) for v in r['iats'][:12]], c=r['counts'][:12], p=r['protocol']), separators=(',', ':'))
    return h64(l1.encode()), h64(l2.encode()), h64(np.asarray(r['stats'], dtype=np.float32).tobytes())


def base_run(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg['base_run'], ROOT)


def snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    return read(base_run(cfg) / 'protocol_snapshot.json')


# ---------------------------------------------------------------------------
# Stage base: light arrays, identity keys and the residual mask
# ---------------------------------------------------------------------------
def stage_base(cfg: dict[str, Any], out: Path) -> None:
    target = out / 'base'
    if (target / 'arrays.npz').exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    src = base_run(cfg)
    with np.load(src / 'encoders/flowstats-v1/dataset.npz') as z:
        y, split, ids, names = z['y'], z['split'], z['flow_id'], z['class_names'].tolist()
    N = len(y)
    fin_col = read(src / 'corpus/manifest.json')['stats_columns'].index('fin_count')
    int32 = {k: np.zeros(N, np.int32) for k in ('group', 'sport', 'dport', 'packets', 'fl', 'fin', 'kept')}
    int64 = {k: np.zeros(N, np.int64) for k in ('st', 'en', 'fpay', 'bpay', 'first_idx', 'last_idx')}
    small = {k: np.zeros(N, np.int16) for k in ('cap', 'proto', 'er', 'frst', 'brst')}
    ips = {k: np.zeros(N, np.uint32) for k in ('sip', 'dip')}
    keys = {k: np.zeros(N, np.uint64) for k in ('L1', 'L2', 'L3s')}
    groups: dict[str, int] = {}; caps: dict[str, int] = {}; reasons: dict[str, int] = {}
    ip = lambda s: struct.unpack('!I', socket.inet_aton(s))[0]
    t0 = time.time()
    with (src / 'corpus/flows.jsonl').open() as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if r['flow_id'] != str(ids[i]):
                raise RuntimeError('corpus and dataset rows are not aligned')
            int32['group'][i] = groups.setdefault(r['group_id'], len(groups))
            small['cap'][i] = caps.setdefault(r['capture_id'], len(caps))
            small['er'][i] = reasons.setdefault(r['end_reason'], len(reasons))
            small['proto'][i] = r['protocol']; small['frst'][i] = r['fwd_rst_packets']; small['brst'][i] = r['bwd_rst_packets']
            ips['sip'][i] = ip(r['src_ip']); ips['dip'][i] = ip(r['dst_ip'])
            int32['sport'][i] = r['src_port']; int32['dport'][i] = r['dst_port']; int32['packets'][i] = r['packets']
            int32['fl'][i] = r['burst_tokens'][0][5]; int32['fin'][i] = r['stats'][fin_col]; int32['kept'][i] = sum(r['counts'])
            int64['st'][i] = r['start_ns']; int64['en'][i] = r['end_ns']; int64['fpay'][i] = r['fwd_payload_bytes']; int64['bpay'][i] = r['bwd_payload_bytes']
            int64['first_idx'][i] = r['first_packet_index']; int64['last_idx'][i] = r['last_packet_index']
            keys['L1'][i], keys['L2'][i], keys['L3s'][i] = keys_of(r)
            if (i + 1) % 500000 == 0:
                print(f'base: {i + 1:,} rows {time.time() - t0:.0f}s', flush=True)
    np.savez(target / 'arrays.npz', y=y, split=split, flow_id=ids, **int32, **int64, **small, **ips, **keys)
    write(target / 'meta.json', dict(class_names=names, captures=caps, end_reasons=reasons, rows=N))
    a = dict(np.load(target / 'arrays.npz'))
    tail, root, prev = residual_mask(a, reasons)
    expected = 284600
    if int(tail.sum()) != expected:
        raise RuntimeError(f'residual rule gave {int(tail.sum())}, EXP94 analysis gave {expected}')
    np.savez(target / 'residual.npz', tail=tail, root=root, prev=prev)
    for name, c in caps.items():
        m = np.flatnonzero(tail & (a['cap'] == c))
        two = m[a['packets'][m] == 2]
        packets = np.r_[a['first_idx'][m], a['last_idx'][two]]
        parents = a['first_idx'][root[np.r_[m, two]]]
        np.savez(target / f'tail_{name}.npz', packet=packets, root_first=parents)
    print('base: residual', int(tail.sum()), 'roots', len(np.unique(root[tail])), flush=True)


def residual_mask(a: dict[str, np.ndarray], reasons: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(a['y'])
    order = np.lexsort((np.arange(N), a['st'], a['group']))
    prev = np.full(N, -1, np.int64)
    same = a['group'][order[1:]] == a['group'][order[:-1]]
    prev[order[1:][same]] = order[:-1][same]
    has = prev >= 0
    gap = np.full(N, np.nan)
    gap[has] = (a['st'][has] - a['en'][prev[has]]) / 1e9
    closed = np.zeros(N, bool)
    closed[has] = np.isin(a['er'][prev[has]], [reasons['rst'], reasons['fin_handshake']])
    with np.errstate(invalid='ignore'):
        tail = closed & (gap >= 0) & (gap <= GAP_S) & (a['packets'] <= 2) & ((a['fl'] & 2) == 0) & ((a['fpay'] + a['bpay']) == 0)
    root = prev.copy()
    for i in order[tail[order]].tolist():
        if tail[prev[i]]:
            root[i] = root[prev[i]]
    if tail[root[tail]].any():
        raise RuntimeError('a residual root is itself residual')
    return tail, root, prev


# ---------------------------------------------------------------------------
# Stage merge: replay the captures, appending residual packets to the root parent
# ---------------------------------------------------------------------------
def iter_merged(path, capture, group, stats, tail_root, remaining, raw, idle_seconds=120, active_seconds=120, reorder_seconds=1, allow_truncated=False):
    """``iter_stats_flows`` of EXP94 with one change: a residual packet joins its held root parent."""
    active, expiry, tombstone, held = {}, [], {}, {}
    idle_ns, active_ns = int(idle_seconds * 1e9), int(active_seconds * 1e9)
    for p in flowlib.packets(path, stats, reorder_seconds, allow_truncated):
        while expiry and expiry[0][0] < p.ns:
            _, key, start, version = heapq.heappop(expiry)
            f = active.get(key)
            if f is not None and f.start == start and f.count == version:
                yield f.row('timeout'); del active[key]
        root = tail_root.get(p.index)
        if root is not None:
            f, reason = held[root]
            raw[p.index] = (p.seq, p.ack)
            f.add(p); stats['merged_tail_packets'] += 1
            if p.flags & 4:
                tombstone[p.key] = p.ns + 2_000_000_000
            remaining[root] -= 1
            if remaining[root] == 0:
                yield f.row(reason); del held[root]
            continue
        key, f = p.key, active.get(p.key)
        syn = p.proto == 6 and p.flags & 2 and not p.flags & 16
        if f and ((syn and (f.first.seq != p.seq or f.fin)) or p.ns - f.start > active_ns):
            yield f.row('new_syn' if syn else 'active_timeout'); del active[key]; f = None
        if f is None:
            if p.proto == 6 and p.flags == 16 and p.payload_len == 0 and p.ns <= tombstone.get(key, -1):
                stats['closed_session_ack_tail'] += 1; continue
            f = active[key] = StatsFlow(p, capture, group)
            raw[p.index] = (p.seq, p.ack)
        f.add(p)
        closed = p.proto == 6 and (p.flags & 4 or (len(f.fin) == 2 and p.flags & 16 and not p.flags & 1 and not p.payload_len))
        if closed:
            reason = 'rst' if p.flags & 4 else 'fin_handshake'
            if f.first.index in remaining:
                held[f.first.index] = (f, reason)
            else:
                yield f.row(reason)
            del active[key]; tombstone[key] = p.ns + 2_000_000_000
        else:
            heapq.heappush(expiry, (min(f.start + active_ns, f.end + idle_ns), key, f.start, f.count))
        if len(expiry) > max(4096, 4 * len(active)):
            expiry = [(min(v.start + active_ns, v.end + idle_ns), k, v.start, v.count) for k, v in active.items()]
            heapq.heapify(expiry)
        if stats['raw_packets'] % 10000 == 0:
            tombstone = {k: v for k, v in tombstone.items() if v >= p.ns}
    if held:
        raise RuntimeError(f'{len(held)} root parents still held at capture end')
    for f in active.values():
        row = f.row('capture_end')
        row['capture_truncated'] = bool(stats['truncated_capture_tail'])
        yield row


def merge_one(cfg: dict[str, Any], out: Path, capture: str) -> None:
    snap = snapshot(cfg)
    cap_cfg = next(c for c in snap['captures'] if c['id'] == capture)
    target = out / 'merge'; target.mkdir(exist_ok=True)
    with np.load(out / 'base' / f'tail_{capture}.npz') as z:
        packet, parents = z['packet'].tolist(), z['root_first'].tolist()
    tail_root = dict(zip(packet, parents))
    remaining = dict(Counter(parents))
    if len(tail_root) != len(packet):
        raise RuntimeError('a residual packet is listed twice')
    stats: Counter = Counter(); raw: dict[int, tuple[int, int]] = {}
    t0 = time.time(); n = 0
    with (target / f'{capture}.parents.jsonl').open('w') as sink, (target / f'{capture}.ids.txt').open('w') as ids:
        for row in iter_merged(resolve_path(cap_cfg['path'], ROOT), capture, cap_cfg['group_namespace'], stats, tail_root, remaining, raw, **snap['flow']):
            n += 1
            ids.write(row['flow_id'] + '\n')
            if row['first_packet_index'] in remaining:
                sink.write(json.dumps(row, separators=(',', ':')) + '\n')
    index = np.fromiter(raw.keys(), np.int64, len(raw)); values = np.asarray(list(raw.values()), dtype=np.int64)
    np.savez(target / f'{capture}.raw.npz', index=index, seq=values[:, 0], ack=values[:, 1])
    write(target / f'{capture}.stats.json', dict(stats=dict(stats), flows=n, seconds=round(time.time() - t0)))
    print(f'merge: {capture} {n:,} flows, {stats["merged_tail_packets"]:,} tail packets, {time.time() - t0:.0f}s', flush=True)


def stage_merge(cfg: dict[str, Any], out: Path, config_path: str) -> None:
    target = out / 'merge'
    if (target / 'parents.npz').exists():
        return
    target.mkdir(exist_ok=True)
    meta = read(out / 'base/meta.json')
    procs = [subprocess.Popen([sys.executable, __file__, '--config', config_path, '--output', str(out), '--stage', 'merge_one', '--capture', c])
             for c in meta['captures'] if not (target / f'{c}.stats.json').exists()]
    if any(p.wait() != 0 for p in procs):
        raise RuntimeError('a capture replay failed')
    a = np.load(out / 'base/arrays.npz'); r = np.load(out / 'base/residual.npz')
    tail, root = r['tail'], r['root']
    ids = a['flow_id']; caps = a['cap']; first_idx = a['first_idx']
    manifest = read(base_run(cfg) / 'corpus/manifest.json')['raw_stats']
    check = {}
    first_seq = np.full(len(ids), -1, np.int64); first_ack = np.full(len(ids), -1, np.int64)
    for name, c in meta['captures'].items():
        got = (target / f'{name}.ids.txt').read_text().split()
        want = ids[(caps == c) & ~tail]
        same = len(got) == len(want) and set(got) == set(map(str, want))
        st = read(target / f'{name}.stats.json')['stats']
        tail_expected = int(a['packets'][(caps == c) & tail].sum())
        ack_ok = st.get('closed_session_ack_tail', 0) == manifest[name].get('closed_session_ack_tail', 0)
        check[name] = dict(flow_ids_equal=bool(same), flows=len(got), tail_packets=st.get('merged_tail_packets', 0), tail_expected=tail_expected,
                           closed_ack=st.get('closed_session_ack_tail', 0), closed_ack_base=manifest[name].get('closed_session_ack_tail', 0))
        if not same or st.get('merged_tail_packets', 0) != tail_expected or not ack_ok:
            raise RuntimeError(f'merge replay diverged from EXP94: {name} {check[name]}')
        with np.load(target / f'{name}.raw.npz') as z:
            lookup = pd.Series(np.arange(len(z['index'])), index=z['index']); seq, ack = z['seq'], z['ack']
        rows = np.flatnonzero(caps == c)
        pos = lookup.reindex(first_idx[rows]).values
        if np.isnan(pos).any():
            raise RuntimeError(f'missing raw fields for {int(np.isnan(pos).sum())} flows in {name}')
        pos = pos.astype(np.int64); first_seq[rows] = seq[pos]; first_ack[rows] = ack[pos]
    np.savez(target / 'raw_first.npz', first_seq=first_seq, first_ack=first_ack)
    write(target / 'verification.json', check)
    index_of = {str(v): i for i, v in enumerate(ids)}
    base_L2 = a['L2']
    rows_idx: list[int] = []; stats_rows, en, pk, fpay, bpay, frst, brst, keys = [], [], [], [], [], [], [], []
    with (target / 'changed_rows.jsonl').open('w') as changed:
        for name in meta['captures']:
            with (target / f'{name}.parents.jsonl').open() as f:
                for line in f:
                    m = json.loads(line); i = index_of[m['flow_id']]
                    k = keys_of(m)
                    rows_idx.append(i); stats_rows.append(m['stats']); en.append(m['end_ns']); pk.append(m['packets'])
                    fpay.append(m['fwd_payload_bytes']); bpay.append(m['bwd_payload_bytes']); frst.append(m['fwd_rst_packets']); brst.append(m['bwd_rst_packets']); keys.append(k)
                    changed.write(json.dumps(dict(row=i, changed=bool(k[1] != int(base_L2[i])), **{t: m[t] for t in ('flow_duration', 'burst_tokens', 'directions', 'bytes', 'iats', 'counts', 'protocol')}), separators=(',', ':')) + '\n')
    K = np.asarray(keys, dtype=np.uint64)
    np.savez(target / 'parents.npz', row=np.asarray(rows_idx, np.int64), stats=np.asarray(stats_rows, np.float32), en=np.asarray(en, np.int64), packets=np.asarray(pk, np.int32),
             fpay=np.asarray(fpay, np.int64), bpay=np.asarray(bpay, np.int64), frst=np.asarray(frst, np.int16), brst=np.asarray(brst, np.int16), L1=K[:, 0], L2=K[:, 1], L3s=K[:, 2])
    roots = np.unique(root[tail])
    if set(rows_idx) != set(roots.tolist()):
        raise RuntimeError('merged parents differ from residual roots')
    print('merge: parents', len(rows_idx), 'with changed input', int((K[:, 1] != base_L2[np.asarray(rows_idx)]).sum()), flush=True)


# ---------------------------------------------------------------------------
# Stage encode: frozen netFound on merged parents whose input changed
# ---------------------------------------------------------------------------
def stage_encode(cfg: dict[str, Any], out: Path, device: str, batch_size: int) -> None:
    target = out / 'merge'
    if (target / 'embeddings.npy').exists():
        return
    snap = snapshot(cfg)
    enc_cfg = snap['encoders']['netfound-v2-small']
    encoder = NetFoundEncoder(ROOT, resolve_path(enc_cfg['checkpoint'], ROOT), device, enc_cfg.get('dtype', 'float32'))
    changed, same = [], []
    with (target / 'changed_rows.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            (changed if r['changed'] else same).append(r)
    emb = np.zeros((len(changed), encoder.metadata['hidden_size']), np.float32)
    t0 = time.time()
    for s in range(0, len(changed), batch_size):
        emb[s:s + batch_size] = encoder.encode(changed[s:s + batch_size])
        if s % (batch_size * 2000) == 0:
            print(f'encode: {s:,}/{len(changed):,} {time.time() - t0:.0f}s', flush=True)
    np.save(target / 'embeddings.npy', emb)
    np.save(target / 'embedded_rows.npy', np.asarray([r['row'] for r in changed], np.int64))
    base = np.load(base_run(cfg) / 'encoders/netfound-v2-small/embeddings.npy', mmap_mode='r')
    probe = same[:4000]
    worst = 0.0
    for s in range(0, len(probe), batch_size):
        part = probe[s:s + batch_size]
        v = encoder.encode(part)
        worst = max(worst, float(np.abs(v - base[[r['row'] for r in part]]).max()))
    write(target / 'encoding.json', dict(rows=len(changed), unchanged_parents=len(same), batch_size=batch_size, probe_rows=len(probe), probe_max_abs_difference=worst, **encoder.metadata))
    if worst > 2e-3:
        raise RuntimeError(f're-encoding unchanged parents differs from EXP94 by {worst}')
    print('encode: done', len(changed), 'probe max diff', worst, flush=True)


# ---------------------------------------------------------------------------
# Stage views: per-handling vectors and side columns
# ---------------------------------------------------------------------------
class Window:
    __slots__ = ('q', 'c', 'f', 't')

    def __init__(self, t: int):
        self.q = deque(); self.c: dict = {}; self.f = 0; self.t = t

    def push(self, now: int, value: Any, fail: int) -> tuple[int, int, int]:
        q, c = self.q, self.c
        while q and q[0][0] < now - self.t:
            _, old, bad = q.popleft(); self.f -= bad
            if c[old] == 1:
                del c[old]
            else:
                c[old] -= 1
        q.append((now, value, fail)); c[value] = c.get(value, 0) + 1; self.f += fail
        return len(q), len(c), self.f


def context_features(a: dict[str, np.ndarray], idx: np.ndarray, en: np.ndarray, fail: np.ndarray, L2: np.ndarray, seconds: float) -> np.ndarray:
    """Causal between-flow context ordered by flow end; IP addresses are grouping keys only."""
    T = int(seconds * 1e9)
    F = np.zeros((len(idx), len(CTX_BASE)), np.float32)
    cap = a['cap'][idx]
    order = np.lexsort((np.arange(len(idx)), en, cap))
    si, di, sp, dp, pr = (a[k][idx].tolist() for k in ('sip', 'dip', 'sport', 'dport', 'proto'))
    ee, fl, kk, cc = en.tolist(), fail.astype(int).tolist(), L2.tolist(), cap.tolist()
    last = -1
    cli: dict = {}; pair: dict = {}; srv: dict = {}; sig: dict = {}
    for i in order.tolist():
        if cc[i] != last:
            cli, pair, srv, sig = {}, {}, {}, {}; last = cc[i]
        s, d, sa, sb, t, bad = si[i], di[i], sp[i], dp[i], ee[i], fl[i]
        if pr[i] == 1:
            service, server, client = 0, d, s
        elif sa < sb:
            service, server, client = sa, s, d
        else:
            service, server, client = sb, d, s
        w = cli.get(client)
        if w is None:
            w = cli[client] = Window(T)
        cf, ct, cb = w.push(t, (server, service), bad)
        key = (server, client); w = pair.get(key)
        if w is None:
            w = pair[key] = Window(T)
        pf, pp, pb = w.push(t, service, bad)
        key = (server, service); w = srv.get(key)
        if w is None:
            w = srv[key] = Window(T)
        sf, sc, sbad = w.push(t, client, bad)
        key = (server, client, kk[i]); w = sig.get(key)
        if w is None:
            w = sig[key] = Window(T)
        rep, _, _ = w.push(t, 0, 0)
        F[i] = (cf, ct, cb / cf, pf, pp, pb / pf, sf, sc, sbad / sf, rep)
    for j, name in enumerate(CTX_BASE):
        if not name.endswith('fail'):
            F[:, j] = np.log1p(F[:, j])
    return F


def stage_views(cfg: dict[str, Any], out: Path) -> None:
    from sklearn.decomposition import PCA
    done = out / 'handling/views.json'
    if done.exists():
        return
    src = base_run(cfg)
    a = dict(np.load(out / 'base/arrays.npz'))
    r = np.load(out / 'base/residual.npz'); tail, root = r['tail'], r['root']
    N = len(tail); split = a['split']; meta = {}
    raw = np.load(out / 'merge/raw_first.npz'); p = np.load(out / 'merge/parents.npz')
    with np.load(src / 'encoders/flowstats-v1/dataset.npz') as z:
        Xs = z['X']
    with np.load(src / 'encoders/netfound-v2-small/dataset.npz') as z:
        Xn = z['X']
    emb = np.load(src / 'encoders/netfound-v2-small/embeddings.npy', mmap_mode='r')
    new_rows = np.load(out / 'merge/embedded_rows.npy'); new_emb = np.load(out / 'merge/embeddings.npy')
    seq0 = ((a['proto'] == 6) & (raw['first_seq'] == 0)).astype(np.float32)
    ack0 = ((a['proto'] == 6) & (raw['first_ack'] == 0)).astype(np.float32)
    tail_cols = np.zeros((N, 3), np.float32)
    roots = root[tail]
    np.add.at(tail_cols[:, 0], roots, a['packets'][tail].astype(np.float32))
    np.add.at(tail_cols[:, 1], roots, (a['frst'][tail] + a['brst'][tail]).astype(np.float32))
    np.add.at(tail_cols[:, 2], roots, a['fin'][tail].astype(np.float32))
    tail_cols = np.log1p(tail_cols)
    keep_rows = np.arange(N); kept_rows = np.flatnonzero(~tail)
    window = float(cfg['context_window_seconds'])
    fail = ((a['frst'] + a['brst']) > 0) | ((a['fpay'] + a['bpay']) == 0)
    names = SEQ + [f'{c}_60' for c in CTX_BASE] + TAIL
    for handling in ('keep', 'drop', 'merge'):
        h = out / 'handling' / handling; h.mkdir(parents=True, exist_ok=True)
        idx = keep_rows if handling == 'keep' else kept_rows
        pos = np.full(N, -1, np.int64); pos[idx] = np.arange(len(idx))
        en = a['en'][idx].copy(); bad = fail[idx].copy(); L1, L2, L3s = a['L1'][idx].copy(), a['L2'][idx].copy(), a['L3s'][idx].copy()
        S = Xs[idx].copy()
        if handling == 'keep':
            Z = Xn.copy(); info = dict(pca='EXP94', pca_fit_rows=None)
        else:
            E = np.asarray(emb[idx], dtype=np.float32)
            if handling == 'merge':
                q = pos[p['row']]
                en[q] = p['en']; S[q] = p['stats']; L1[q] = p['L1']; L2[q] = p['L2']; L3s[q] = p['L3s']
                bad[q] = ((p['frst'] + p['brst']) > 0) | ((p['fpay'] + p['bpay']) == 0)
                E[pos[new_rows]] = new_emb
            tr = split[idx] == 'train'
            t0 = time.time()
            pca = PCA(n_components=64, svd_solver='full', random_state=int(cfg['seed'])).fit(E[tr])
            Z = pca.transform(E).astype(np.float32)
            info = dict(pca='refit on train rows of this handling', pca_fit_rows=int(tr.sum()), explained_variance=float(pca.explained_variance_ratio_.sum()), seconds=round(time.time() - t0))
            del E, pca; gc.collect()
        if not (np.isfinite(Z).all() and np.isfinite(S).all()):
            raise FloatingPointError(f'non-finite vectors in {handling}')
        np.save(h / 'index.npy', idx); np.save(h / 'flowstats-v1.npy', S.astype(np.float32)); np.save(h / 'netfound-v2-small.npy', Z)
        t0 = time.time()
        C = context_features(a, idx, en, bad, L2, window)
        E2 = np.column_stack([seq0[idx], ack0[idx], C, tail_cols[idx]]).astype(np.float32)
        np.savez(h / 'extras.npz', X=E2, names=np.asarray(names))
        np.savez(h / 'keys.npz', L1=L1, L2=L2, L3s=L3s)
        info.update(rows=len(idx), context_seconds=round(time.time() - t0))
        meta[handling] = info
        print('views:', handling, info, flush=True)
    write(done, meta)


# ---------------------------------------------------------------------------
# Stage audit: identical inputs under each handling
# ---------------------------------------------------------------------------
def hash_rows(X: np.ndarray) -> np.ndarray:
    X = np.ascontiguousarray(X)
    return np.fromiter((int.from_bytes(hashlib.blake2b(v.tobytes(), digest_size=8).digest(), 'little') for v in X), np.uint64, len(X))


def identity(key: np.ndarray, y: np.ndarray, split: np.ndarray, packets: np.ndarray | None = None) -> dict[str, Any]:
    m = y >= 0
    d = pd.DataFrame(dict(k=key[m], y=y[m]))
    g = d.groupby(['k', 'y']).size().rename('n').reset_index()
    s = g.groupby('k').n.agg(['sum', 'max', 'size'])
    c = s[s['size'] > 1]
    size = d.groupby('k')['y'].transform('size').values
    rows = len(d)
    conflict, minority = int(c['sum'].sum()), int((c['sum'] - c['max']).sum())
    same = int((size > 1).sum())
    tr = pd.unique(d.k[split[m] == 'train'])
    te = d.k[split[m] == 'test']
    out = dict(rows=rows, distinct=int(len(s)), same_input_rows=same, same_input_pct=round(100 * same / rows, 2), conflict_rows=conflict, conflict_pct=round(100 * conflict / rows, 3),
               same_label_pct=round(100 * (same - conflict) / rows, 2), groups=len(c), minority_rows=minority, minority_pct=round(100 * minority / rows, 3),
               accuracy_ceiling_pct=round(100 * (1 - minority / rows), 3), test_rows_in_train_pct=round(100 * float(te.isin(tr).mean()), 1))
    if packets is not None:
        bins = [(1, 1, '1'), (2, 2, '2'), (3, 5, '3-5'), (6, 20, '6-20'), (21, 10**9, '21+')]
        pk = packets[m]; flag = d.groupby('k')['y'].transform('nunique').values > 1
        out['by_packets'] = {n: dict(rows=int(((pk >= lo) & (pk <= hi)).sum()), same_input_pct=round(100 * float((size[(pk >= lo) & (pk <= hi)] > 1).mean()), 1) if ((pk >= lo) & (pk <= hi)).any() else None,
                                     conflict_rows=int((flag & (pk >= lo) & (pk <= hi)).sum())) for lo, hi, n in bins}
    return out


def stage_audit(cfg: dict[str, Any], out: Path) -> None:
    done = out / 'audit/identity.json'
    if done.exists():
        return
    done.parent.mkdir(exist_ok=True)
    a = np.load(out / 'base/arrays.npz'); y, split, packets = a['y'], a['split'], a['packets']
    p = np.load(out / 'merge/parents.npz'); result: dict[str, Any] = {}
    sets = cfg['feature_sets']
    for handling in ('keep', 'drop', 'merge'):
        h = out / 'handling' / handling
        idx = np.load(h / 'index.npy'); k = np.load(h / 'keys.npz'); ex = np.load(h / 'extras.npz'); names = [str(v) for v in ex['names']]; E = ex['X']
        yy, ss = y[idx], split[idx]; pk = packets[idx].copy()
        if handling == 'merge':
            pos = np.full(len(y), -1, np.int64); pos[idx] = np.arange(len(idx)); pk[pos[p['row']]] = p['packets']
        Z = np.round(np.load(h / 'netfound-v2-small.npy'), 3) + 0.0
        S = np.load(h / 'flowstats-v1.npy')
        res = dict(L1_flow_information=identity(k['L1'], yy, ss), L2_netfound_input=identity(k['L2'], yy, ss, pk), L3_stats45=identity(k['L3s'], yy, ss), L3_netfound_pca64=identity(hash_rows(Z), yy, ss))
        for fs, cols in sets.items():
            if not cols:
                continue
            sel = [names.index(c) for c in cols]
            side = np.round(E[:, sel], 3) + 0.0
            res[f'L3_netfound_pca64+{fs}'] = identity(hash_rows(np.column_stack([Z, side])), yy, ss)
            res[f'L3_stats45+{fs}'] = identity(hash_rows(np.column_stack([S, side])), yy, ss)
        result[handling] = res
        print('audit:', handling, {n: (v['same_input_pct'], v['conflict_pct'], v['minority_rows']) for n, v in res.items()}, flush=True)
        if handling == 'keep':
            m = yy >= 0
            d = pd.DataFrame(dict(k=k['L2'][m], y=yy[m]))
            conflict = np.zeros(len(idx), bool); conflict[np.flatnonzero(m)] = d.groupby('k')['y'].transform('nunique').values > 1
            np.save(out / 'audit/keep_L2_conflict.npy', conflict)
    write(done, result)


# ---------------------------------------------------------------------------
# Stage evaluate: paired few-shot jobs on every arm, with validation and common scoring
# ---------------------------------------------------------------------------
def load_arm(cfg: dict[str, Any], out: Path, handling: str, feature_set: str, view: str) -> tuple[np.ndarray, np.ndarray]:
    source = 'drop' if handling == 'tailcount' else handling
    h = out / 'handling' / source
    idx = np.load(h / 'index.npy'); X = np.load(h / f'{view}.npy')
    with np.load(h / 'extras.npz') as z:
        E, names = z['X'], [str(v) for v in z['names']]
    cols = list(cfg['feature_sets'][feature_set]) + (TAIL if handling == 'tailcount' else [])
    if cols:
        X = np.column_stack([X, E[:, [names.index(c) for c in cols]]])
    return X.astype(np.float32), idx


def score(yt: np.ndarray, pr: np.ndarray, classes: list[int], names: list[str]) -> dict[str, Any]:
    r = _f1(yt, pr, classes, names)
    return dict(macro_f1=r['macro_f1'], rows=r['rows'], classes={c['name']: {k: c[k] for k in ('f1', 'precision', 'recall', 'support')} for c in r['classes']})


def fit_predict(method: str, seed: int, classes: list[int], few: dict[str, Any], X: np.ndarray, chosen: np.ndarray, y_train: np.ndarray, parts: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, Any]]:
    model, construct = _timed(lambda: _make_model(method, seed, classes, few))
    _, fit = _timed(lambda: model.fit(X[chosen], y_train))
    frozen = _weights_digest(model) if method == 'tabpfn' else None
    batch = int(few.get('inference_batch_rows', 8192))
    out, infer = [], 0.0
    for rows in parts:
        proba, t = _timed(lambda: _predict_proba_batched(model, X[rows], batch))
        if proba.shape != (len(rows), len(classes)) or not np.isfinite(proba).all():
            raise RuntimeError('invalid probability matrix')
        out.append(np.asarray(classes, dtype=np.int16)[proba.argmax(1)]); infer += t
    if frozen and frozen != _weights_digest(model):
        raise RuntimeError('TabPFN backbone weights changed during inference')
    del model; gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out, dict(construct_seconds=construct, fit_seconds=fit, inference_seconds=infer, frozen_weights_sha256=frozen)


def evaluate_arm(cfg: dict[str, Any], out: Path, handling: str, feature_set: str, view: str) -> None:
    few = snapshot(cfg)['fewshot']
    a = np.load(out / 'base/arrays.npz'); r = np.load(out / 'base/residual.npz')
    y_all, split_all, ids_all = a['y'], a['split'], a['flow_id']; tail, root = r['tail'], r['root']
    meta = read(out / 'base/meta.json'); names = meta['class_names']
    X, idx = load_arm(cfg, out, handling, feature_set, view)
    y, split = y_all[idx], split_all[idx]; N = len(y_all)
    pos = np.full(N, -1, np.int64); pos[idx] = np.arange(len(idx))
    conflict_keep = np.load(out / 'audit/keep_L2_conflict.npy')
    arm_dir = out / 'evaluations' / f'{handling}__{feature_set}' / view
    train = split == 'train'
    budgets, seeds = few['budgets'], few['seeds']; max_budget = max(budgets)
    for regime in few['regimes']:
        classes = list(range(len(names))) if regime == 'with_benign' else list(range(1, len(names)))
        test_all = (split_all == 'test') & np.isin(y_all, classes)
        own_base = np.flatnonzero(test_all & (pos >= 0))
        if handling == 'keep':
            pred_base = own_base
        else:
            pred_base = np.union1d(own_base, root[test_all & tail])
        pred_rows = pos[pred_base]
        if (pred_rows < 0).any():
            raise RuntimeError('a prediction row is absent from this handling')
        at = np.full(N, -1, np.int64); at[pred_base] = np.arange(len(pred_base))
        full_base = np.flatnonzero(test_all)
        val_base = np.flatnonzero((split_all == 'validation') & np.isin(y_all, classes) & (pos >= 0) & ~tail)
        arm_dir.mkdir(parents=True, exist_ok=True)
        np.save(arm_dir / f'prediction_rows_{regime}.npy', pred_base)
        for seed in seeds:
            selected = {}
            for c in classes:
                pool = np.flatnonzero(train & (y == c))
                selected[c] = np.random.default_rng(seed + 1009 * c).choice(pool, max_budget, replace=False)
            for budget in budgets:
                chosen = np.concatenate([selected[c][:budget] for c in classes])
                local = {c: i for i, c in enumerate(classes)}
                y_train = np.asarray([local[int(v)] for v in y[chosen]], dtype=np.int64)
                for method in few['methods']:
                    job = arm_dir / 'jobs' / f'{regime}_s{seed}_n{budget}_{method}'
                    if (job / 'result.json').exists():
                        continue
                    job.mkdir(parents=True, exist_ok=True)
                    parts = [pred_rows]
                    use_val = regime == 'with_benign' and budget in cfg['validation_budgets']
                    if use_val:
                        parts.append(pos[val_base])
                    preds, timing = fit_predict(method, seed, classes, few, X, chosen, y_train, parts)
                    p_test = preds[0]
                    own_pred = p_test[at[own_base]]
                    common = ~tail[own_base]
                    full_pred = p_test[at[np.where(tail[full_base] & (handling != 'keep'), root[full_base], full_base)]]
                    metrics = dict(own=score(y_all[own_base], own_pred, classes, names),
                                   common=score(y_all[own_base][common], own_pred[common], classes, names),
                                   full=score(y_all[full_base], full_pred, classes, names))
                    conflict_rows = conflict_keep[full_base]
                    extra = dict(conflict_rows=int(conflict_rows.sum()), conflict_accuracy=float((full_pred[conflict_rows] == y_all[full_base][conflict_rows]).mean()) if conflict_rows.any() else None)
                    if 0 in classes:
                        ben = y_all[own_base] == 0
                        extra['benign_alarm_rate_common'] = float((own_pred[ben & common] != 0).mean())
                    row = dict(handling=handling, feature_set=feature_set, view=view, regime=regime, seed=seed, budget=budget, method=method, dimensions=int(X.shape[1]),
                               train_rows=len(chosen), train_flow_ids=[str(v) for v in ids_all[idx[chosen]]], metrics=metrics, extra=extra, **timing)
                    if use_val:
                        row['validation'] = score(y_all[val_base], preds[1], classes, names)
                        np.save(job / 'validation_predictions.npy', preds[1])
                    np.save(job / 'predictions.npy', p_test)
                    write(job / 'result.json', row)
    print('evaluate: done', handling, feature_set, view, flush=True)


def majority(d: pd.DataFrame, cols: list[str]) -> pd.Series:
    t = d.groupby(cols + ['y']).size().rename('n').reset_index().sort_values('n', ascending=False, kind='stable').drop_duplicates(cols)
    return t.set_index(cols)['y']


def lookup(tr: pd.DataFrame, ev: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, float]:
    """Slide 16 rule: train majority on the key, then L2 majority, then the largest class."""
    p = ev[cols].join(majority(tr, cols).rename('p'), on=cols)['p']
    b = ev[['L2']].join(majority(tr, ['L2']).rename('b'), on='L2')['b']
    pred = p.fillna(b).fillna(tr.y.value_counts().idxmax()).astype(np.int64).values
    return pred, float(p.notna().mean())


def lookup_metrics(ev: pd.DataFrame, pred: np.ndarray, seen: float, names: list[str]) -> dict[str, Any]:
    y = ev.y.values
    classes = sorted(int(c) for c in np.unique(y))
    r = score(y, pred, classes, names)
    return dict(rows=len(ev), accuracy=float((pred == y).mean()), macro_f1=r['macro_f1'], seen=seen,
                benign_recall=float((pred[y == 0] == 0).mean()) if (y == 0).any() else None, scan_recall=float((pred[y == 2] == 2).mean()) if (y == 2).any() else None)


def stage_lookup(cfg: dict[str, Any], out: Path) -> None:
    done = out / 'part1/lookup.json'
    if done.exists():
        return
    done.parent.mkdir(exist_ok=True)
    spec = cfg['part1']['lookup']
    a = np.load(out / 'base/arrays.npz'); r = np.load(out / 'base/residual.npz'); tail = r['tail']
    meta = read(out / 'base/meta.json'); names = meta['class_names']; caps = meta['captures']
    with np.load(out / 'handling/keep/extras.npz') as z:
        E, cols = z['X'], [str(v) for v in z['names']]
    ports = np.rint(np.expm1(E[:, cols.index('pair_ports_60')].astype(np.float64)))
    d = pd.DataFrame(dict(L2=a['L2'], y=a['y'], split=a['split'], cap=a['cap'], tail=tail, dport=a['dport'], svc=np.minimum(a['sport'], a['dport']),
                          first_seq_zero=E[:, cols.index('first_seq_zero')].astype(np.int8), first_ack_zero=E[:, cols.index('first_ack_zero')].astype(np.int8),
                          pair_ports_bin=np.floor(np.log2(np.maximum(ports, 1))).astype(np.int16)))
    d = d[d.y >= 0]
    result: dict[str, Any] = {}
    for handling, frame in (('keep', d), ('drop', d[~d.tail])):
        conflict = frame.groupby('L2')['y'].transform('nunique').values > 1
        for scope, sub in (('conflict', frame[conflict]), ('all', frame)):
            tr = sub[sub.split == 'train']
            for name, extra in spec['candidates'].items():
                keys = ['L2'] + extra
                row = dict(rows=len(sub))
                for part in ('validation', 'test'):
                    ev = sub[sub.split == part]
                    pred, seen = lookup(tr, ev, keys)
                    row[part] = lookup_metrics(ev, pred, seen, names)
                result.setdefault(handling, {}).setdefault(scope, {})[name] = row
                print('lookup:', handling, scope, name, round(row['validation']['accuracy'], 4), round(row['test']['accuracy'], 4), flush=True)
    scan = names.index(spec['cross_session_class'])
    cross: dict[str, Any] = {}
    for src, dst in spec['cross_session']:
        a_cap, b_cap = caps[src], caps[dst]
        tag = f"{src.split('_')[1]}_to_{dst.split('_')[1]}"
        for handling, frame in (('keep', d), ('drop', d[~d.tail])):
            tr = frame[(frame.split == 'train') & ((frame.y != scan) | (frame.cap == a_cap))]
            ev = frame[((frame.split == 'test') & (frame.y != scan)) | ((frame.y == scan) & (frame.cap == b_cap))]
            for name, extra in spec['candidates'].items():
                pred, seen = lookup(tr, ev, ['L2'] + extra)
                y = ev.y.values
                cross.setdefault(tag, {}).setdefault(handling, {})[name] = dict(rows=len(ev), scan_rows=int((y == scan).sum()), seen=seen, scan_recall=float((pred[y == scan] == scan).mean()),
                                                                                benign_called_scan=float((pred[y == 0] == scan).mean()), benign_rows=int((y == 0).sum()))
                print('cross:', tag, handling, name, round(cross[tag][handling][name]['scan_recall'], 4), flush=True)
    write(done, dict(lookup=result, cross_session=cross))


def full_label_xgb(cfg: dict[str, Any], out: Path) -> None:
    from xgboost import XGBClassifier  # noqa: F401
    spec = cfg['part1']['full_label_xgb']; few = snapshot(cfg)['fewshot']
    a = np.load(out / 'base/arrays.npz'); r = np.load(out / 'base/residual.npz')
    y, split = a['y'], a['split']; tail = r['tail']
    names = read(out / 'base/meta.json')['class_names']; classes = list(range(len(names)))
    conflict = np.load(out / 'audit/keep_L2_conflict.npy')
    lab = y >= 0
    tr = np.flatnonzero((split == 'train') & lab)
    count = np.bincount(y[tr], minlength=len(names)).astype(np.float64)
    weight = (len(tr) / (len(names) * count))[y[tr]]
    target = out / 'part1/full_label_xgb'
    for view in cfg['views']:
        for feature_set in spec['feature_sets']:
            X, idx = load_arm(cfg, out, 'keep', feature_set, view)
            if not np.array_equal(idx, np.arange(len(y))):
                raise RuntimeError('full-label XGB expects every EXP94 row')
            for seed in spec['seeds']:
                job = target / view / f'{feature_set}_s{seed}'
                if (job / 'result.json').exists():
                    continue
                job.mkdir(parents=True, exist_ok=True)
                model, _ = _timed(lambda: _make_model('xgb', seed, classes, few))
                _, fit = _timed(lambda: model.fit(X[tr], y[tr].astype(np.int64), sample_weight=weight))
                row = dict(view=view, feature_set=feature_set, seed=seed, dimensions=int(X.shape[1]), train_rows=len(tr), fit_seconds=fit)
                for part in ('validation', 'test'):
                    rows = np.flatnonzero((split == part) & lab)
                    pred = np.asarray(model.predict(X[rows]), dtype=np.int16)
                    common = ~tail[rows]; ben = y[rows] == 0; c = conflict[rows]
                    row[part] = dict(full=score(y[rows], pred, classes, names), common=score(y[rows][common], pred[common], classes, names),
                                     benign_alarm_rate=float((pred[ben] != 0).mean()), benign_alarm_rate_common=float((pred[ben & common] != 0).mean()),
                                     conflict_rows=int(c.sum()), conflict_accuracy=float((pred[c] == y[rows][c]).mean()))
                    np.save(job / f'{part}_predictions.npy', pred)
                write(job / 'result.json', row)
                print('xgb:', view, feature_set, seed, f"val {row['validation']['full']['macro_f1']:.4f} test {row['test']['full']['macro_f1']:.4f} {fit:.0f}s", flush=True)
                del model; gc.collect()


def stage_evaluate(cfg: dict[str, Any], out: Path, config_path: str, workers: int) -> None:
    procs = []
    for shard in range(workers):
        log = (out / f'worker_{shard}.log').open('a')
        procs.append(subprocess.Popen([sys.executable, __file__, '--config', config_path, '--output', str(out), '--stage', 'worker', '--shard', str(shard), '--shards', str(workers)], stdout=log, stderr=subprocess.STDOUT))
    log = (out / 'part1_xgb.log').open('a')
    procs.append(subprocess.Popen([sys.executable, __file__, '--config', config_path, '--output', str(out), '--stage', 'full_xgb'], stdout=log, stderr=subprocess.STDOUT))
    failed = [i for i, p in enumerate(procs) if p.wait() != 0]
    if failed:
        raise RuntimeError(f'evaluation processes failed: {failed}')


def worker(cfg: dict[str, Any], out: Path, shard: int, shards: int) -> None:
    tasks = [(h, f, v) for h, f in cfg['arms'] for v in cfg['views']]
    for task in tasks[shard::shards]:
        t0 = time.time()
        evaluate_arm(cfg, out, *task)
        print('task', task, f'{time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------------------
# Stage report: flat tables
# ---------------------------------------------------------------------------
def stage_report(cfg: dict[str, Any], out: Path) -> None:
    few = snapshot(cfg)['fewshot']
    rows = []
    for path in (out / 'evaluations').glob('*/*/jobs/*/result.json'):
        d = read(path)
        base = {k: d[k] for k in ('handling', 'feature_set', 'view', 'regime', 'seed', 'budget', 'method')}
        base.update(macro_own=d['metrics']['own']['macro_f1'], macro_common=d['metrics']['common']['macro_f1'], macro_full=d['metrics']['full']['macro_f1'],
                    macro_validation=d.get('validation', {}).get('macro_f1'), conflict_accuracy=d['extra']['conflict_accuracy'], benign_alarm_rate_common=d['extra'].get('benign_alarm_rate_common'),
                    own_rows=d['metrics']['own']['rows'], common_rows=d['metrics']['common']['rows'], full_rows=d['metrics']['full']['rows'], dimensions=d['dimensions'], seconds=d['fit_seconds'] + d['inference_seconds'])
        for scope in ('common', 'full'):
            for c, v in d['metrics'][scope]['classes'].items():
                base[f'{scope}_f1_{c}'] = v['f1']
        rows.append(base)
    df = pd.DataFrame(rows).sort_values(['view', 'method', 'regime', 'handling', 'feature_set', 'budget', 'seed'])
    df.to_csv(out / 'summary_jobs.csv', index=False)
    expected = len(cfg['arms']) * len(cfg['views']) * len(few['seeds']) * len(few['budgets']) * len(few['methods']) * len(few['regimes'])
    xgb = [read(p) for p in (out / 'part1/full_label_xgb').glob('*/*/result.json')]
    spec = cfg['part1']['full_label_xgb']
    expected_xgb = len(cfg['views']) * len(spec['feature_sets']) * len(spec['seeds'])
    if len(df) != expected or len(xgb) != expected_xgb:
        raise RuntimeError(f'incomplete jobs: {len(df)}/{expected}, {len(xgb)}/{expected_xgb}')
    base = pd.read_csv(base_run(cfg) / 'evaluations/netfound-v2-small/fewshot/summary.csv')
    mine = df[(df.handling == 'keep') & (df.feature_set == 'none') & (df.view == 'netfound-v2-small')]
    joined = mine.merge(base, on=['regime', 'seed', 'budget', 'method'])
    gap = (joined.macro_own - joined.macro_f1).abs()
    write(out / 'reproduction_check.json', dict(jobs=len(joined), max_abs_macro_difference=float(gap.max()), mean_abs_macro_difference=float(gap.mean()), by_method={m: float(gap[joined.method == m].max()) for m in few['methods']}))
    print('report:', len(df), len(xgb), 'reproduction max diff', float(gap.max()), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--stage', default='all', choices=['all', 'prepare', 'base', 'merge', 'merge_one', 'encode', 'views', 'audit', 'lookup', 'full_xgb', 'evaluate', 'worker', 'report'])
    ap.add_argument('--capture'); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()
    cfg = read(resolve_path(args.config, ROOT))
    out = resolve_path(args.output, ROOT); out.mkdir(parents=True, exist_ok=True)
    stage = args.stage
    if stage == 'merge_one':
        return merge_one(cfg, out, args.capture)
    if stage == 'worker':
        return worker(cfg, out, args.shard, args.shards)
    if stage == 'full_xgb':
        return full_label_xgb(cfg, out)
    write(out / 'protocol_snapshot.json', cfg)
    if stage in ('all', 'prepare', 'base'):
        stage_base(cfg, out)
    if stage in ('all', 'prepare', 'merge'):
        stage_merge(cfg, out, args.config)
    if stage in ('all', 'prepare', 'encode'):
        stage_encode(cfg, out, args.device, args.batch_size)
    if stage in ('all', 'prepare', 'views'):
        stage_views(cfg, out)
    if stage in ('all', 'prepare', 'audit'):
        stage_audit(cfg, out)
    if stage in ('all', 'lookup'):
        stage_lookup(cfg, out)
    if stage in ('all', 'evaluate'):
        stage_evaluate(cfg, out, args.config, int(cfg['evaluate_workers']))
    if stage in ('all', 'report'):
        stage_report(cfg, out)
    write(out / 'status.json', dict(status='complete', stage=stage))


if __name__ == '__main__':
    main()
