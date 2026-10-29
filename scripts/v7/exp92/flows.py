"""Offline IPv4 TCP/UDP/ICMP flows, bounded netFound bursts, auditable packet IDs.

The netFound byte order, relative TCP numbers, per-direction 10 ms bursts and
6-packet/12-burst prefix follow the pinned upstream extractor/tokenizer. IPs,
ports, absolute timestamps and labels are metadata, never model input tokens.
"""
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import heapq
import socket
import struct

import dpkt

def digest(value): return hashlib.sha256(repr(value).encode()).hexdigest()[:24]

@dataclass
class Packet:
    index: int
    ns: int
    src: str
    dst: str
    sport: int
    dport: int
    proto: int
    ip_header: list
    flags: int = 0
    window: int = 0
    seq: int = 0
    ack: int = 0
    urgent: int = 0
    udp_len: int = 0
    icmp_type: int = 0
    icmp_code: int = 0
    payload_len: int = 0

    @property
    def key(self): return (self.proto, *sorted(((self.src, self.sport), (self.dst, self.dport))))

def parse_packet(index, ns, buf, linktype, stats):
    if linktype == dpkt.pcap.DLT_EN10MB: ip = dpkt.ethernet.Ethernet(buf).data
    elif linktype == dpkt.pcap.DLT_LINUX_SLL: ip = dpkt.sll.SLL(buf).data
    elif linktype in (dpkt.pcap.DLT_RAW, 101, 228): ip = dpkt.ip.IP(buf)
    else: raise ValueError(f'Unsupported PCAP link type {linktype}')
    if not isinstance(ip, dpkt.ip.IP): stats['non_ipv4'] += 1; return None
    if ip.mf or ip.offset: stats['ipv4_fragments_excluded'] += 1; return None
    if ip.p not in (1, 6, 17): stats['other_transport'] += 1; return None
    transport = ip.data
    expected = {6: dpkt.tcp.TCP, 17: dpkt.udp.UDP, 1: dpkt.icmp.ICMP}[ip.p]
    if not isinstance(transport, expected): stats['undecoded_transport'] += 1; return None
    p = Packet(index, ns, socket.inet_ntoa(ip.src), socket.inet_ntoa(ip.dst),
               getattr(transport, 'sport', 0), getattr(transport, 'dport', 0), ip.p,
               [ip.hl*4, ip.tos, ip.len, (int(ip.rf)<<2) | (int(ip.df)<<1) | int(ip.mf), ip.ttl])
    p.payload_len = len(bytes(transport.data))
    if ip.p == 6:
        p.flags, p.window, p.seq, p.ack, p.urgent = transport.flags & 255, transport.win, transport.seq, transport.ack, transport.urp
    elif ip.p == 17: p.udp_len = transport.ulen
    else:
        p.icmp_type, p.icmp_code = transport.type, transport.code
        # Separate echo conversations by identifier; pair request/reply types.
        # Other ICMP types stay separate rather than collapsing every ICMP message.
        ident = getattr(transport.data, 'id', 0)
        p.sport = p.dport = ident if transport.type in (0, 8) else 65536+transport.type*256+transport.code
    return p

def classic_records(f, stats, allow_truncated):
    """Unlike dpkt.Reader, do not yield a short final captured packet as complete."""
    header=f.read(24)
    formats={b'\xd4\xc3\xb2\xa1':('<',1000),b'\xa1\xb2\xc3\xd4':('>',1000),
             b'\x4d\x3c\xb2\xa1':('<',1),b'\xa1\xb2\x3c\x4d':('>',1)}
    if len(header)!=24 or header[:4] not in formats:raise ValueError('Invalid classic PCAP header')
    order,multiplier=formats[header[:4]]
    _,_,_,_,snaplen,link=struct.unpack(order+'HHIIII',header[4:])
    link &= 0xffff
    def records():
        while True:
            offset=f.tell();h=f.read(16)
            if not h:return
            if len(h)!=16:
                if not allow_truncated:raise IOError('Truncated PCAP packet header')
                stats['truncated_capture_tail']+=1;stats['truncated_tail_bytes']=len(h);return
            sec,fraction,included,wire=struct.unpack(order+'IIII',h)
            if included>wire or included>(256<<20):raise ValueError('Invalid PCAP packet length')
            raw=f.read(included)
            if len(raw)!=included:
                if not allow_truncated:raise IOError('Truncated PCAP packet body')
                stats['truncated_capture_tail']+=1;stats['truncated_tail_bytes']=f.tell()-offset;return
            if included<wire:stats['snaplen_shortened_packets']+=1
            yield sec*1_000_000_000+fraction*multiplier,raw
    return link,records()

def packets(path, stats, reorder_seconds=1.0, allow_truncated=False):
    """Bounded timestamp ordering; reject disorder beyond the stated window."""
    heap, latest, previous = [], -1, -1
    with open(path, 'rb') as f:
        magic = f.read(4); f.seek(0)
        if magic == b'\x0a\x0d\x0d\x0a':
            reader=dpkt.pcapng.Reader(f);link=reader.datalink()
            iterator=((int(Decimal(str(ts))*1_000_000_000),raw) for ts,raw in reader)
        else:
            link,iterator=classic_records(f,stats,allow_truncated)
        if link not in (dpkt.pcap.DLT_EN10MB,dpkt.pcap.DLT_LINUX_SLL,dpkt.pcap.DLT_RAW,101,228):
            raise ValueError(f'Unsupported PCAP link type {link}')
        index = 0
        while True:
            try: ns, raw = next(iterator)
            except StopIteration: break
            except (ValueError, dpkt.NeedData, struct.error) as e:
                if not allow_truncated: raise IOError(f'Invalid/truncated capture: {path}') from e
                stats['truncated_capture_tail'] += 1; break
            index += 1; stats['raw_packets'] += 1
            if ns < latest: stats['out_of_order_packets'] += 1
            latest = max(latest, ns)
            try: p = parse_packet(index, ns, raw, link, stats)
            except (dpkt.UnpackError, ValueError, struct.error):
                stats['malformed_or_truncated_packet'] += 1; continue
            if p is None: continue
            heapq.heappush(heap, (ns, index, p))
            while heap and heap[0][0] <= latest-int(reorder_seconds*1e9):
                time, _, packet = heapq.heappop(heap)
                if time < previous: raise ValueError('Capture disorder exceeds reorder window; run reordercap or increase configured window')
                previous = time; yield packet
        while heap:
            time, _, packet = heapq.heappop(heap)
            if time < previous: raise ValueError('Capture disorder exceeds reorder window')
            previous = time; yield packet

@dataclass
class Flow:
    first: Packet
    capture: str
    group: str
    start: int = field(init=False)
    end: int = field(init=False)
    count: int = 0
    payload: list = field(default_factory=lambda: [0, 0])
    direction_count: list = field(default_factory=lambda: [0, 0])
    fin: set = field(default_factory=set)
    last_index: int = 0
    seq0: int = 0
    ack0: int = 0
    last_dir_time: list = field(default_factory=lambda: [None, None])
    last_burst_start: list = field(default_factory=lambda: [None, None])
    burst_count: list = field(default_factory=lambda: [0, 0])
    bursts: list = field(default_factory=lambda: [[], []])

    def __post_init__(self):
        self.start = self.end = self.first.ns
        self.seq0, self.ack0 = self.first.seq, self.first.ack

    def add(self, p):
        d = int((p.src, p.sport) != (self.first.src, self.first.sport))
        if self.count == 1 and self.ack0 == 0: self.ack0 = p.seq
        # Replicate upstream uint32 subtraction, including wrap-around.
        seq = (p.seq-(self.seq0 if d == 0 else self.ack0)) & 0xffffffff
        ack = (p.ack-(self.ack0 if d == 0 else self.seq0)) & 0xffffffff
        if d == 0 and self.ack0 == 0: ack = 0
        hdr = p.ip_header.copy()
        if p.proto == 6: hdr += [p.flags, p.window, seq>>16, seq&65535, ack>>16, ack&65535, p.urgent]
        elif p.proto == 17: hdr += [p.udp_len]
        else: hdr += [p.icmp_type, p.icmp_code]
        # Official tokenizer accepts six payload tokens and strips them for v2.
        # Zero placeholders keep its field layout; no payload enters this model.
        hdr += [0]*6
        fresh = self.last_dir_time[d] is None or p.ns-self.last_dir_time[d] > 10_000_000
        if fresh:
            iat = 0 if self.last_burst_start[d] is None else p.ns-self.last_burst_start[d]
            self.last_burst_start[d] = p.ns
            self.burst_count[d] += 1
            if self.burst_count[d] <= 12:
                self.bursts[d].append(dict(start=p.ns, direction=d == 0, iat=round(iat/1000),
                                          tokens=[], bytes=0, count=0, packet_indices=[]))
        if self.burst_count[d] <= 12:
            b = self.bursts[d][-1]
            if b['count'] < 6:
                b['tokens'].extend(hdr); b['bytes'] += p.ip_header[2]; b['count'] += 1
                b['packet_indices'].append(p.index)
        self.last_dir_time[d] = p.ns
        self.end, self.last_index = p.ns, p.index
        self.count += 1; self.direction_count[d] += 1; self.payload[d] += p.payload_len
        if p.flags & 1: self.fin.add(d)

    def row(self, reason):
        bs = sorted(self.bursts[0]+self.bursts[1], key=lambda b:b['start'])[:12]
        return dict(flow_id=digest((self.capture, self.first.key, self.start, self.first.index)),
                    group_id=digest((self.group, self.first.key)), capture_id=self.capture,
                    src_ip=self.first.src, dst_ip=self.first.dst, src_port=self.first.sport, dst_port=self.first.dport,
                    start_ns=self.start, end_ns=self.end, protocol=self.first.proto,
                    packets=self.count, fwd_packets=self.direction_count[0], bwd_packets=self.direction_count[1],
                    fwd_payload_bytes=self.payload[0], bwd_payload_bytes=self.payload[1],
                    first_packet_index=self.first.index, last_packet_index=self.last_index,
                    retained_packet_indices=[i for b in bs for i in b['packet_indices']], end_reason=reason,
                    flow_duration=(self.end-self.start)//1000,
                    burst_tokens=[b['tokens'] for b in bs], directions=[b['direction'] for b in bs],
                    bytes=[b['bytes'] for b in bs], iats=[b['iat'] for b in bs], counts=[b['count'] for b in bs],
                    observed_bursts=sum(self.burst_count))

def iter_flows(path, capture, group, stats, idle_seconds=120, active_seconds=120,
               reorder_seconds=1, allow_truncated=False):
    active, expiry, tombstone = {}, [], {}
    idle_ns, active_ns = int(idle_seconds*1e9), int(active_seconds*1e9)
    for p in packets(path, stats, reorder_seconds, allow_truncated):
        while expiry and expiry[0][0] < p.ns:
            _, key, start, version = heapq.heappop(expiry)
            f = active.get(key)
            if f is not None and f.start == start and f.count == version:
                yield f.row('timeout'); del active[key]
        key, f = p.key, active.get(p.key)
        syn = p.proto == 6 and p.flags & 2 and not p.flags & 16
        if f and ((syn and (f.first.seq != p.seq or f.fin)) or p.ns-f.start > active_ns):
            yield f.row('new_syn' if syn else 'active_timeout'); del active[key]; f = None
        if f is None:
            # Only pure ACK tails of a just-closed TCP session are suppressed.
            # One-packet SYN/RST scans and data packets are retained.
            if p.proto == 6 and p.flags == 16 and p.payload_len == 0 and p.ns <= tombstone.get(key, -1):
                stats['closed_session_ack_tail'] += 1; continue
            f = active[key] = Flow(p, capture, group)
        f.add(p)
        closed = p.proto == 6 and (p.flags & 4 or (len(f.fin) == 2 and p.flags & 16 and not p.flags & 1 and not p.payload_len))
        if closed:
            yield f.row('rst' if p.flags & 4 else 'fin_handshake')
            del active[key]; tombstone[key] = p.ns+2_000_000_000
        else:
            heapq.heappush(expiry, (min(f.start+active_ns, f.end+idle_ns), key, f.start, f.count))
        if len(expiry) > max(4096, 4*len(active)):
            expiry = [(min(v.start+active_ns, v.end+idle_ns), k, v.start, v.count) for k,v in active.items()]
            heapq.heapify(expiry)
        if stats['raw_packets'] % 10000 == 0:
            tombstone = {k:v for k,v in tombstone.items() if v >= p.ns}
        stats['peak_active_flows'] = max(stats['peak_active_flows'], len(active))
    for f in active.values():
        row=f.row('capture_end')
        row['capture_truncated']=bool(stats['truncated_capture_tail'])
        yield row
