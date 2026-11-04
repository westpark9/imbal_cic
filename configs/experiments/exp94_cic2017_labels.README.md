# EXP94 — CIC-IDS2017 corrected labelling rules (transcription notes)

Companion to `configs/experiments/exp94_cic2017_labels.json`. Transcribed 2026-11-03 from:

- N = notebook `CICIDS2017_labelling_fixed_CICFlowMeter.ipynb` (GintsEngelen/CNS2022_Code, master) — the executable source; every `start_ns`/`end_ns`, IP, port and feature condition below is copied from its `label_flows(...)` calls.
- D = https://intrusion-detection.distrinet-research.be/CNS2022/CICIDS2017.html (per-attack prose + "Labelling logic" blocks, attempted-category names).
- DL = https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html (how to treat "Attempted").
- T = https://intrusion-detection.distrinet-research.be/CNS2022/Tools_Documentation.html (CICFlowMeter changes: UTC timestamps, RST features, "TCP Segmentation Offset" fix).
- W = https://intrusion-detection.distrinet-research.be/WTMC2021/extended_doc.html and .../tools_datasets.html (2021 predecessor: flow-construction bugs, manually tuned windows, class counts).
- R = https://github.com/GintsEngelen/CNS2022_Code/blob/master/README.md (pcapfix + reordercap, CICFlowMeter timeouts 120 s / 5 s).

## Time handling

N states: "Timestamps are in NANOSECONDS (!) Unix time. Note that the CSV files are in the UTC timezone." D prints every bound as "Time Start UTC (First Packet) ... (unix: ...)". So the windows are already UTC epoch nanoseconds and were copied verbatim; **no local-to-UTC conversion was performed and the notebook never gives local-time strings**. Each rule's `window_note` prints the UTC wall clock and the derived ADT (UTC-3, Fredericton in July) wall clock so it can be checked against the CIC schedule (e.g. FTP-Patator 12:17 UTC = 09:17 ADT; CIC lists 9:20–10:20). The notebook matches the CICFlowMeter *flow-start* timestamp against the window; `CorrectedRuleSet` requires the whole session inside the window (otherwise `boundary`).

## Matcher extensions required BEFORE use

The matcher ignores unknown fields, so these five rules (`status: requires_matcher_extension`) apply only their expressible part until `CorrectedRuleSet._condition` is extended:

| rule | missing condition (N) | effect until fixed |
|---|---|---|
| `ssh_patator_attempted_nomalicious` | `Total Length of Bwd Packet == 0` (proposed `max_peer_payload_bytes: 0`) | over-matches ≤32 B-attacker-payload flows with victim payload into Attempted (→ benign) instead of SSH-Patator |
| `dos_slowhttptest_attempted_unresponsive` | `Total Bwd packets == 0` (proposed `max_peer_packets: 0`) | category 6 vs 0 swap only; both Attempted |
| `dos_slowloris_attempted_unresponsive` | N writes it in the victim frame (Src IP 192.168.10.50, Src Port 80, Dst Port ∉ {33358,33360,33362,54114}, `Total Length of Bwd Packet == 0`, `Flow Duration >= 199800`); restated in the attacker frame because our sessions have no CICFlowMeter direction flip | category 6 vs 0 swap only; both Attempted |
| `web_bruteforce_attempted_artefact` | `Total Bwd packets == 5` (proposed `min_peer_packets`/`max_peer_packets: 5`) | annotation only; end label benign either way |
| `infiltration_portscan_round3_host50` | `~((Fwd Packet Length Max == 408) & Dst IP == .50) & ~((Total Length of Fwd Packet in [176, 20514]) & Dst IP == .50)` | **CONFIRMED rule over-labels benign 192.168.10.8→192.168.10.50 background as portscan — disable or extend first** |

Other translation choices: the notebook OR `(Total Fwd Packet > 20) | (Src Port == 44464)` is split into two confirmed rules (`web_bruteforce`, `web_bruteforce_port44464`) because `any_of` cannot test ports; DoS Hulk's two confirmed intervals are expressed as a union rule with the port exclusion plus `dos_hulk_late` (so a session straddling 13:45:00 UTC is not left `boundary`); no `protocol` field is set because the notebook has none; precedence of later notebook calls is encoded as higher `priority`.

## Per day

Attacker host is 172.16.0.1 unless stated. All windows: [UTC first packet, UTC last packet] → (ADT).

### Monday 2017-07-03 — `cic2017_monday`
No attacks; N labels everything BENIGN. Empty rule list.

### Tuesday 2017-07-04 — `cic2017_tuesday`

| attack | window UTC (ADT) | attacker → victim | evidence condition (confirmed) | attempted categories |
|---|---|---|---|---|
| FTP-Patator (bf) | 12:17:52.838272 – 13:20:16.931403 (09:17:52 – 10:20:16) | 172.16.0.1 → 192.168.10.50:21 | time + endpoints + dst port 21 only (N) | cat 0 NoPayload (`Total Length of Fwd Packet == 0`); cat 2 StartupTeardown (src port 52108 = successful-login pre-test, D) |
| SSH-Patator (bf) | 17:09:01.049616 – 19:04:19.018486 (14:09:01 – 16:04:19) | 172.16.0.1 → 192.168.10.50:22 | time + endpoints + dst port 22 only (N) | cat 0 NoPayload; cat 3 NoMaliciousPayload (`Fwd ≤ 32 & Bwd == 0`, TLS handshake only, D) — needs `max_peer_payload_bytes` |

### Wednesday 2017-07-05 — `cic2017_wednesday`

| attack | window UTC (ADT) | attacker → victim | evidence condition (confirmed) | attempted categories |
|---|---|---|---|---|
| DoS Slowloris (dos) | 12:48:54.539220 – 13:11:18.500956 (09:48:54 – 10:11:18) | 172.16.0.1 → 192.168.10.50:80 | src port ∉ {33358, 33360, 33362, 54114} (N) | cat 5 ImplementedIncorrectly: separate early-launch window 12:48:46.211817 – 12:48:47.000000 (N; D quotes 12:48:46.264300 as last packet); cat 2 StartupTeardown (ports 33358/33360/33362); cat 0 NoPayload; cat 6 TargetUnresponsive (0 payload & `Flow Duration ≥ 199800`); cat 4 AttackArtefact (port 54114) |
| DoS Slowhttptest (dos) | 13:15:37.936810 – 13:37:49.331517 (10:15:37 – 10:37:49) | 172.16.0.1 → 192.168.10.50:80 | src port ≠ 33372 (N) | cat 2 (port 33372); cat 0 NoPayload (excl. 33372, 37670); cat 6 (0 payload & `Flow Duration ≥ 199984` & bwd packets == 0); cat 4 (port 37670) |
| DoS Hulk (dos) | 13:43:23.194704 – 13:44:59.999999999 with src port ∉ {48678, 43664}; 13:45:00.000000 – 14:07:21.326171 unrestricted (10:43:23 – 11:07:21) | 172.16.0.1 → 192.168.10.50:80 | time/endpoints/port exclusion only (N); tool misimplemented but kept as Hulk (D) | cat 4 AttackArtefact (ports 48678/43664, first interval); cat 0 NoPayload; cat 3 NoMaliciousPayload (`0 < Fwd payload < 282`) |
| DoS GoldenEye (dos) | 14:10:03.231753 – 14:20:08.915718 (11:10:03 – 11:20:08) | 172.16.0.1 → 192.168.10.50:80 | time + endpoints + port 80 (N) | cat 0 NoPayload |
| Heartbleed (heartbleed) | 18:12:15.650811 – 18:32:43.294455 (15:12:15 – 15:32:43) | 172.16.0.1:45022 → 192.168.10.51:444 | fixed src port 45022 (N, D) | cat 0 NoPayload |

Documentation inconsistency: D prints Slowloris' end as "14:20:08.91578" beside unix 1499260278.500956 (= 13:11:18.500956 UTC); the unix value/N is used.

### Thursday 2017-07-06 — `cic2017_thursday`

| attack | window UTC (ADT) | attacker → victim | evidence condition (confirmed) | attempted categories |
|---|---|---|---|---|
| Web Attack – Brute Force (web) | 12:19:27.660566 – 13:00:11.622209 (09:19:27 – 10:00:11) | 172.16.0.1 → 192.168.10.50:80 | `Total Fwd Packet > 20` OR src port 44464 (N) | cat 2 StartupTeardown: separate set-up window 12:15:54.880049 – 12:18:51.179279; cat 0 NoPayload (& not the confirmed condition); cat 4 AttackArtefact (payload > 0 & fwd == 5 & bwd == 5 & port ≠ 44464) |
| Web Attack – XSS (web) | 13:15:35.283859 – 13:35:21.341704 (10:15:35 – 10:35:21) | 172.16.0.1 → 192.168.10.50:80 | `Total Fwd Packet ≥ 150` & src port ∉ {36180, 36182, 36184, 36186, 36188, 36190} (N) | cat 0 NoPayload; cat 2 StartupTeardown (payload > 0 & fwd packets < 150) |
| Web Attack – SQL Injection (web) | 13:35:45.732950 – 13:42:55.320284 (10:35:45 – 10:42:55) | 172.16.0.1 → 192.168.10.50:80 | src port ∉ {36180, 36182, 36184, 36186, 36188} (N) | cat 2 StartupTeardown: set-up window 13:35:27.852814 – 13:35:45.720612, src ports ∈ that set; cat 0 NoPayload — **N applies it to the set-up window without port filter; D lists it for the main window with the port exclusion** (N followed) |
| Infiltration – communication (inf) | 17:19:02.547210 – 18:46:09.364731 (14:19:02 – 15:46:09) | victim 192.168.10.8 → attacker 205.174.165.73 (no ports) | time + endpoints (N); Dropbox download left Benign (D) | cat 0 NoPayload; cat 2 StartupTeardown = 192.168.10.9 → 205.174.165.73, 17:13:48.830533 – 17:15:01.251276 (dry run) |
| Infiltration – Cool disk MAC (inf) | 17:53:36.453990 – 20:02:19.347892 (14:53:36 – 17:02:19) | 192.168.10.25 → 205.174.165.73 (no ports) | time + endpoints (N); Meterpreter push (D); missed in original | cat 0 NoPayload |
| Infiltration – Portscan round 1 (portscan) | 17:00:31.706755 – 17:00:45.728887 (14:00:31 – 14:00:45) | 172.16.0.1 → 192.168.10.51, src ports 50122/50133 | fixed src ports (N) | none |
| Infiltration – Portscan round 2 | 17:33:30.884008 – 17:34:04.285175 (14:33:30 – 14:34:04) | 192.168.10.8 → 192.168.10.5 | time + endpoints (N); benign in original | none |
| Infiltration – Portscan round 3 | 18:05:14.425162 – 18:46:04.331875 (15:05:14 – 15:46:04) | 192.168.10.8 → {.5, .9, .12, .14, .15, .16, .17, .19, .25, .50, .51} | time + endpoints; for dst .50 exclude `Fwd Packet Length Max == 408` and `Total Length of Fwd Packet ∈ {176, 20514}` (N) — **not expressible, see table above** | none |

### Friday 2017-07-07 — `cic2017_friday`

| attack | window UTC (ADT) | attacker → victim | evidence condition (confirmed) | attempted categories |
|---|---|---|---|---|
| Botnet Ares (bot) | 13:04:13.990571 – 14:02:02.903736 (10:04:13 – 11:02:02) | {192.168.10.15, .9, .14, .5, .8} → 205.174.165.73 (no ports) | time + endpoints (N) | cat 0 NoPayload; cat 1 PortClosed: 14:03:00.000000 – 20:01:24.606663 (N; D prints start 14:03:14.356593) |
| Portscan (portscan) | round 1 16:55:32.117090 – 17:19:08.582083 (13:55:32 – 14:19:08); round 2 17:51:45.450532 – 18:24:01.699238 (14:51:45 – 15:24:01) | 172.16.0.1 → 192.168.10.50 (no ports) | time + endpoints (N); round 1 behind firewall, only 21/22/80/443/444 pass (D) | none |
| DDoS LOIC HTTP (ddos) | 18:56:31.796937 – 19:16:12.216560 (15:56:31 – 16:16:12) | 172.16.0.1 → 192.168.10.50 (no ports) | time + endpoints (N); mostly empty GETs, identical flows exist in background (D) | cat 0 NoPayload |

## Attempted categories (D)

0 No payload sent by attacker · 1 Port/System closed · 2 Attack Startup/Teardown Artefact · 3 No malicious payload · 4 Attack Artefact · 5 Attack Implemented Incorrectly · 6 Target System Unresponsive. DL: never a separate ML class; merge into the parent attack or (default) relabel Benign. The JSON maps every `*-Attempted-*` annotation to `benign`, which is also what `CorrectedRuleSet` does via `unmatched_policy`.

## Not available / not translated

- "Friday timezone shift": not documented in N, D, DL, T, R or W as fetched on 2026-11-03. Nothing asserted.
- TCP segmentation offload: T only says "Fixed TCP Segmentation Offset issue (more details in our paper under section IV-B)"; the CNS2022 paper has no open full text (IEEE DOI 10.1109/CNS56114.2022.9947235; imec/cps-vo records carry no PDF), so no 2017-specific detail is transcribed.
- Original PCAP download URLs/zip members are not given (UNB distributes plain per-day PCAPs; the EXP93 fetch step expects zip members). Pre-processing per R: pcapfix → reordercap.
- Family for the notebook label "Infiltration - Portscan" is proposed as `portscan` (behaviour) rather than `inf` (original CIC label); confirm before use.
- D's prose contains typos ("205.174.165.74", "102.168.10.25") in the Cool disk section; N uses 205.174.165.73 and 192.168.10.25, which were used.
