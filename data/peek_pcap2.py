#!/usr/bin/env python3
"""
Sample packets spread across the full pcapng by reading every SAMPLE_EVERY-th packet.
Analyze time span, IP structure, port distribution across the whole day.
"""

import os, sys, datetime, collections
from scapy.all import PcapNgReader
from scapy.layers.inet import IP, TCP, UDP

PCAP = os.path.join(os.path.dirname(__file__), "Friday-WorkingHours.pcap")
SAMPLE_EVERY = 200   # keep 1 in every N packets
MAX_RECORDS  = 3000  # stop after this many records

records = []  # (ts_unix, src_ip, dst_ip, proto, sport, dport)
pkt_count = 0

print(f"File: {os.path.getsize(PCAP)/1e9:.2f} GB  |  sampling 1/{SAMPLE_EVERY} packets (max {MAX_RECORDS} records) ...")

with PcapNgReader(PCAP) as reader:
    for pkt in reader:
        pkt_count += 1
        if pkt_count % SAMPLE_EVERY != 0:
            continue
        if not pkt.haslayer(IP):
            continue
        ip = pkt[IP]
        ts = float(pkt.time)
        proto = ip.proto
        sport = dport = None
        if pkt.haslayer(TCP):
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
        elif pkt.haslayer(UDP):
            sport, dport = pkt[UDP].sport, pkt[UDP].dport
        records.append((ts, ip.src, ip.dst, proto, sport, dport))
        if len(records) >= MAX_RECORDS:
            break

print(f"Scanned {pkt_count:,} packets, captured {len(records)} samples.\n")
if not records:
    print("No IP records found."); sys.exit(1)

timestamps = [r[0] for r in records]
t_min, t_max = min(timestamps), max(timestamps)
dt_min = datetime.datetime.fromtimestamp(t_min)
dt_max = datetime.datetime.fromtimestamp(t_max)

print("=== FULL CAPTURE TIME SPAN ===")
print(f"  Start : {dt_min}")
print(f"  End   : {dt_max}")
print(f"  Span  : {(t_max-t_min)/3600:.2f} hours")

# 5-minute traffic volume buckets
print("\n=== TRAFFIC VOLUME per 5-min bucket ===")
bucket_counts = collections.Counter()
for ts, *_ in records:
    bucket_counts[int((ts - t_min) // 300)] += 1
for b in sorted(bucket_counts):
    t_label = datetime.datetime.fromtimestamp(t_min + b * 300).strftime("%H:%M")
    bar = "#" * bucket_counts[b]
    print(f"  {t_label}  {bar:30s} ({bucket_counts[b]})")

# IP analysis
def is_private(ip):
    p = list(map(int, ip.split(".")))
    return (p[0] == 10 or
            (p[0] == 172 and 16 <= p[1] <= 31) or
            (p[0] == 192 and p[1] == 168))

internal_src = collections.Counter(r[1] for r in records if is_private(r[1]))
external_src = collections.Counter(r[1] for r in records if not is_private(r[1]))

print("\n=== INTERNAL SOURCE IPs ===")
for ip, cnt in internal_src.most_common(15):
    print(f"  {ip:20s}  {cnt}")
print("\n=== EXTERNAL SOURCE IPs (top 10) ===")
for ip, cnt in external_src.most_common(10):
    print(f"  {ip:20s}  {cnt}")

# Direction breakdown
dirs = collections.Counter()
for _, src, dst, *_ in records:
    sp, dp = is_private(src), is_private(dst)
    key = ("int" if sp else "ext") + "→" + ("int" if dp else "ext")
    dirs[key] += 1
print("\n=== FLOW DIRECTION ===")
for d, cnt in dirs.most_common():
    print(f"  {d:12s}  {cnt}")

# Dest port distribution by time third
thirds = len(records) // 3
chunks = [records[:thirds], records[thirds:2*thirds], records[2*thirds:]]
labels = ["early", "middle", "late"]
print("\n=== TOP DST PORTS BY TIME THIRD ===")
for label, chunk in zip(labels, chunks):
    t0 = datetime.datetime.fromtimestamp(chunk[0][0]).strftime("%H:%M")
    t1 = datetime.datetime.fromtimestamp(chunk[-1][0]).strftime("%H:%M")
    ports = collections.Counter(r[5] for r in chunk if r[5] is not None)
    print(f"  [{label} {t0}–{t1}]  {ports.most_common(10)}")

# Unique IPs per time bucket (entropy proxy)
print("\n=== UNIQUE SRC IPs per 5-min bucket ===")
bucket_ips = collections.defaultdict(set)
for ts, src, *_ in records:
    bucket_ips[int((ts - t_min) // 300)].add(src)
for b in sorted(bucket_ips):
    t_label = datetime.datetime.fromtimestamp(t_min + b * 300).strftime("%H:%M")
    print(f"  {t_label}  {len(bucket_ips[b])} unique src IPs")

print("\nDone.")
