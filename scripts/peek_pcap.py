#!/usr/bin/env python3
"""
Read the first N packets from a pcap and extract:
  timestamp, src_ip, dst_ip, src_port, dst_port, protocol, pkt_len
Then print summary statistics on time windows and IP/port distributions.
"""

import sys
import datetime
import collections
import csv
import os

from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP, UDP, ICMP

PCAP = os.path.join(os.path.dirname(__file__), "Friday-WorkingHours.pcap")
MAX_PACKETS = 20_000
OUT_CSV = os.path.join(os.path.dirname(__file__), "friday_peek.csv")

rows = []
print(f"Reading up to {MAX_PACKETS} packets from {PCAP} ...")

with PcapReader(PCAP) as reader:
    for i, pkt in enumerate(reader):
        if i >= MAX_PACKETS:
            break
        if not pkt.haslayer(IP):
            continue
        ip = pkt[IP]
        ts = float(pkt.time)
        proto = ip.proto  # 6=TCP, 17=UDP, 1=ICMP
        src_port = dst_port = None
        if pkt.haslayer(TCP):
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif pkt.haslayer(UDP):
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
        rows.append({
            "ts": ts,
            "src_ip": ip.src,
            "dst_ip": ip.dst,
            "src_port": src_port,
            "dst_port": dst_port,
            "proto": proto,
            "pkt_len": len(pkt),
        })

print(f"Captured {len(rows)} IP packets.\n")

if not rows:
    print("No IP packets found.")
    sys.exit(1)

# ── Save CSV ──────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved raw peek to {OUT_CSV}\n")

# ── Time window analysis ──────────────────────────────────────────────────────
timestamps = [r["ts"] for r in rows]
t_min = min(timestamps)
t_max = max(timestamps)
dt_min = datetime.datetime.fromtimestamp(t_min)
dt_max = datetime.datetime.fromtimestamp(t_max)
span_sec = t_max - t_min

print("=== TIME WINDOW ===")
print(f"  First packet : {dt_min}  (unix {t_min:.3f})")
print(f"  Last  packet : {dt_max}  (unix {t_max:.3f})")
print(f"  Span         : {span_sec:.1f} s  ({span_sec/60:.1f} min)")

# Bucket into 1-minute bins
bucket_counts = collections.Counter()
for r in rows:
    minute = int((r["ts"] - t_min) // 60)
    bucket_counts[minute] += 1
print(f"  1-min buckets: {len(bucket_counts)} non-empty bins")
top_bins = bucket_counts.most_common(5)
print(f"  Busiest bins (minute offset, count): {top_bins}")

# ── IP analysis ───────────────────────────────────────────────────────────────
src_ips = collections.Counter(r["src_ip"] for r in rows)
dst_ips = collections.Counter(r["dst_ip"] for r in rows)
print("\n=== SOURCE IPs (top 15) ===")
for ip, cnt in src_ips.most_common(15):
    print(f"  {ip:20s}  {cnt:6d}")
print("\n=== DEST IPs (top 15) ===")
for ip, cnt in dst_ips.most_common(15):
    print(f"  {ip:20s}  {cnt:6d}")

# Subnets (/24)
src_subnets = collections.Counter(".".join(r["src_ip"].split(".")[:3]) for r in rows)
dst_subnets = collections.Counter(".".join(r["dst_ip"].split(".")[:3]) for r in rows)
print("\n=== SOURCE /24 subnets (top 10) ===")
for sn, cnt in src_subnets.most_common(10):
    print(f"  {sn+'.0/24':22s}  {cnt:6d}")
print("\n=== DEST /24 subnets (top 10) ===")
for sn, cnt in dst_subnets.most_common(10):
    print(f"  {sn+'.0/24':22s}  {cnt:6d}")

# ── Port analysis ─────────────────────────────────────────────────────────────
dst_ports = collections.Counter(r["dst_port"] for r in rows if r["dst_port"] is not None)
src_ports = collections.Counter(r["src_port"] for r in rows if r["src_port"] is not None)
print("\n=== DEST PORTS (top 20) ===")
for p, cnt in dst_ports.most_common(20):
    print(f"  {str(p):8s}  {cnt:6d}")
print("\n=== SRC PORTS (top 10) ===")
for p, cnt in src_ports.most_common(10):
    print(f"  {str(p):8s}  {cnt:6d}")

# ── Protocol breakdown ────────────────────────────────────────────────────────
protos = collections.Counter(r["proto"] for r in rows)
proto_names = {6: "TCP", 17: "UDP", 1: "ICMP"}
print("\n=== PROTOCOLS ===")
for proto, cnt in protos.most_common():
    name = proto_names.get(proto, str(proto))
    print(f"  {name:6s}  {cnt:6d}")

print("\nDone.")
