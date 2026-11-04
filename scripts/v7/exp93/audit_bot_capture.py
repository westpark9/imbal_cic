"""Audit which hosts a CIC2018 host PCAP actually talks to inside a labelled window.

Written for the EXP93 Bot-label review (2026-11-03).  The PCAP has no labels;
this scan reports, per capture, whether the rule's attacker/C2 endpoint and
port appear at all, and what the host's traffic inside the window looks like.
It uses only packet headers and writes a small JSON for the lablog.
"""
from __future__ import annotations
import argparse, collections, datetime as dt, hashlib, json, socket
from pathlib import Path
import dpkt


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def scan(path: Path, host: str, endpoint: str, port: int, start: int, end: int) -> dict:
    peers_all, peers_win, ports_win, hosts = (collections.Counter() for _ in range(4))
    n = endpoint_pkts = port_pkts = port_pkts_win = 0
    first = last = None
    with path.open('rb') as f:
        for ts, buf in dpkt.pcap.Reader(f):
            n += 1
            first = ts if first is None else first
            last = ts
            try:
                ip = dpkt.ethernet.Ethernet(buf).data
            except Exception:
                continue
            if not isinstance(ip, dpkt.ip.IP):
                continue
            src, dst = socket.inet_ntoa(ip.src), socket.inet_ntoa(ip.dst)
            hosts[src] += 1; hosts[dst] += 1
            in_window = start <= ts <= end
            if endpoint in (src, dst):
                endpoint_pkts += 1
            peer = dst if src == host else src
            peers_all[peer] += 1
            if in_window:
                peers_win[peer] += 1
            tr = ip.data
            if hasattr(tr, 'sport'):
                if port in (tr.sport, tr.dport):
                    port_pkts += 1
                    port_pkts_win += in_window
                if in_window:
                    ports_win[min(tr.sport, tr.dport)] += 1
    iso = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat()
    return dict(capture=str(path), sha256=sha256(path), host=host, packets=n,
                first_packet_utc=iso(first), last_packet_utc=iso(last),
                window_utc=[iso(start), iso(end)], rule_endpoint=endpoint, rule_port=port,
                packets_with_rule_endpoint=endpoint_pkts,
                packets_on_rule_port=dict(total=port_pkts, in_window=port_pkts_win),
                top_hosts=hosts.most_common(8), top_peers_all=peers_all.most_common(12),
                top_peers_in_window=peers_win.most_common(12),
                top_low_ports_in_window=ports_win.most_common(12))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', action='append', required=True, help='path=host, repeatable')
    p.add_argument('--endpoint', default='18.219.211.138')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--start', type=int, default=1520000008)
    p.add_argument('--end', type=int, default=1520020492)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = dict(schema=1, purpose='EXP93 Bot capture audit: does the rule endpoint appear in the chosen host PCAP?',
               official_bot_victims=['172.31.69.23', '172.31.69.17', '172.31.69.14', '172.31.69.12', '172.31.69.10',
                                     '172.31.69.8', '172.31.69.6', '172.31.69.26', '172.31.69.29', '172.31.69.30'],
               official_schedule='https://www.unb.ca/cic/datasets/ids-2018.html',
               captures=[])
    for item in a.capture:
        path, host = item.split('=')
        print('scanning', path, flush=True)
        out['captures'].append(scan(Path(path), host, a.endpoint, a.port, a.start, a.end))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    for c in out['captures']:
        print(c['host'], 'endpoint pkts', c['packets_with_rule_endpoint'], 'port pkts', c['packets_on_rule_port'])


if __name__ == '__main__':
    main()
