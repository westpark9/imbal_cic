"""Build the English version of the 2026-11-06 report deck.

The Korean generator stays the single source of layout and numbers.  Its scenes
are built, then every Korean string is replaced from the tables below.  A string
without a translation stops the build, so the two decks cannot drift silently.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = next(path for path in Path(__file__).resolve().parents if (path / 'scripts/common/experiment_paths.py').exists())
sys.path.insert(0, str(ROOT / 'scripts/common'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_20261023_references as r
import build_20261104_exp94 as b04
import build_20261106_pcap_merged as ko

HERE = ROOT / 'docs/slides'
QA = HERE / '_support/qa/20261106_results_eng'
OUT = HERE / 'eng/20261106_PCAP_Results'
ASSETS = HERE / '_support/assets/20261106_results_eng'
HANGUL = re.compile('[가-힣]')

# Text, or (text, size) to change the font size for that string.
TR: dict[str, str | tuple[str, float]] = {
    # 1
    '11/06  CSV에서 PCAP으로': '11/06  From CSV to PCAP',
    'CSV 입력의 한계를 확인하고, 패킷 원본에서 feature를 직접 만드는 실험으로 전환': 'CSV inputs showed clear limits, so features are now built directly from raw packets',
    '이전 실험': 'Prior experiments',
    'CICFlowMeter CSV 벡터 입력\n\nXGB · TabPFN\nSOTA · 제안 모델\n\n평가 시나리오 표현 시도': ('CICFlowMeter CSV vectors\n\nXGB · TabPFN\nSOTA · proposed models\n\nEvaluation scenarios attempted', 16),
    '확인한 문제': 'Problems found',
    '분류 성능 부진\n\nfeature가 같은데 라벨이 다른 행\n\nfew-shot에서도 일부 공격의\n성능이 높아 개선 여지 부족': ('Weak classification\n\nSame features, different labels\n\nSome attacks score high even\nin few-shot, leaving little to gain', 15.5),
    '이번 보고': 'This report',
    'CSV 대신 PCAP 사용\n\nnetFound로 feature\n네트워크 데이터용\ntransformer 인코더': ('PCAP instead of CSV\n\nFeatures from netFound,\na transformer encoder\nfor network data', 16),
    '검증 질문  패킷 기반 feature로 앞의 문제가 개선되는가': 'Question  Do packet-based features fix these problems?',
    'CSV 대신 PCAP과 netFound 인코더로 feature를 만들어 앞의 문제가 개선되는지 실험': 'Build features from PCAP with the netFound encoder and test whether the problems improve',
    '01 배경': '01 Background',
    # 2
    '문제1  CIC-IDS 데이터의 동일 feature·상이 라벨': 'Problem 1  Same Features, Different Labels',
    '실험망에서 PCAP을 수집하고, CICFlowMeter가 패킷을 flow로 묶어 미리 정한 통계를 계산': 'PCAP is captured in a testbed, then CICFlowMeter groups packets into flows and computes fixed statistics',
    '실험망': 'Testbed',
    '정상·공격 통신': 'Benign and attack traffic',
    '시간순 원시 패킷': 'Raw packets in time order',
    'flow 묶기 · 고정 통계 계산': 'Group flows · fixed statistics',
    '통계 벡터 + 라벨': 'Statistics vector + label',
    '고정 통계  flow마다 같은 계산식 78개  ·  지속 시간 · 보낸·받은 패킷 수 · 패킷 길이 평균 · 패킷 간격 평균 · FIN 수': ('Fixed statistics  the same 78 formulas per flow  ·  duration · packets sent, received · mean packet length · mean gap · FIN count', 13.5),
    '같은 feature에 다른 라벨이 생기는 경로': 'How the same features get different labels',
    '① flow 경계 결함 (주원인)': '① Main cause: flow boundary',
    '첫 FIN에서 끊고 RST 무시\n남은 ACK·FIN이 별도 flow\n이런 flow가 CIC2017의 25.9%': 'Cuts at the first FIN, ignores RST\nLeftover ACK and FIN become flows\nThese make up 25.9% of CIC2017',
    '② 패킷이 매우 적은 flow': '② Very short flows',
    '남은 패킷 1~2개로 계산한 통계\n값이 쉽게 일치': 'Statistics from 1 or 2 packets\neasily match',
    '③ 라벨은 원래 flow를 따름': '③ Labels follow the parent',
    '같은 모양의 꼬리 flow에\n정상과 공격 라벨이 혼재': 'Identical tail flows carry\nbenign and attack labels',
    '근거  Engelen 2021 · Liu 2022  ·  PCAP 재구성에서 꼬리 flow를 빼면 통계 충돌 6,367 → 69': 'Evidence  Engelen 2021 · Liu 2022  ·  In our PCAP rebuild, removing tail flows cuts statistics conflicts from 6,367 to 69',
    '데이터': 'Data',
    '통계가 모두 같은 flow의 예': 'Flows with identical statistics',
    '붙어 있는 라벨': 'Labels attached',
    '목적 포트 80 · 1 µs · 보낸 2 · 받은 0': 'Dst port 80 · 1 µs · sent 2 · received 0',
    'BENIGN 2행  ·  DoS Hulk 3,360행': 'BENIGN 2 rows  ·  DoS Hulk 3,360 rows',
    '목적 포트 21 · 2 µs · 보낸 1 · 받은 1': 'Dst port 21 · 2 µs · sent 1 · received 1',
    'SlowHTTPTest 62,024행  ·  FTP 87,297행  ·  SSH 10행': 'SlowHTTPTest 62,024 rows  ·  FTP 87,297 rows  ·  SSH 10 rows',
    'flow 경계 결함이 패킷이 매우 적은 flow를 만들고, 같은 통계에 서로 다른 라벨이 붙음': 'The flow boundary defect creates very short flows, and identical statistics get different labels',
    '02 문제': '02 Problems',
    # 3
    '문제1  CIC2017·2018 CSV의 class별 충돌': 'Problem 1  Conflicts by Class in the CSVs',
    '원본 CSV 전체 행 기준  ·  통계가 완전히 같은데 class가 다른 묶음에 속한 행  ·  행 순서는 CIC2017 행 수 내림차순': 'All rows of the original CSVs  ·  Rows in groups with identical statistics but different classes  ·  Ordered by CIC2017 rows',
    'CIC2017 행': 'CIC2017 rows',
    '충돌 행': ('Conflicts', 13.5),
    '비율 %': 'Share %',
    'CIC2018 행': 'CIC2018 rows',
    '합계': 'Total',
    'CIC2018은 bf 42.65% · inf 25.49% · dos 11.81%로 공격 class 충돌이 큼  ·  CIC2017은 dos 2.07%가 가장 큼': 'CIC2018 attack classes conflict heavily: bf 42.65%, inf 25.49%, dos 11.81%  ·  CIC2017 peaks at dos 2.07%',
    'PCAP 크기 때문에 실험은 CIC2017로 진행 (CIC2018 450 GB, CIC2017 51 GB)': 'Experiments use CIC2017 because of PCAP size (CIC2018 450 GB, CIC2017 51 GB)',
    '충돌 행은 CIC2018 2.51%, CIC2017 0.25%이고, CIC2018은 공격 class에 크게 집중': 'Conflict rows are 2.51% in CIC2018 and 0.25% in CIC2017, and CIC2018 concentrates them in attacks',
    # 4
    '문제2  few-shot에서도 일부 공격은 이미 높은 성능': 'Problem 2  Some Attacks Score High Even in Few-Shot',
    'CIC2017 CSV  ·  공격만 학습·평가  ·  클래스당 라벨 n개  ·  seed 5개 평균': 'CIC2017 CSV  ·  Attacks only in training and evaluation  ·  n labels per class  ·  Mean of 5 seeds',
    'n=20에서 portscan 98% · ddos 94% · dos 92%  ·  n=50에서 98% · 95% · 95%': 'At n=20: portscan 98%, ddos 94%, dos 92%  ·  At n=50: 98%, 95%, 95%',
    '반면 web은 n=50에서도 55%, bot 0%는 CSV 라벨 결함  ·  평균 성능만으로는 난이도 판단 불가': 'web stays at 55% at n=50, and bot 0% is a CSV label defect  ·  Mean scores alone cannot show difficulty',
    '대량·반복 공격은 라벨 20개에서 이미 높아 라벨을 늘려도 개선 여지가 적음': 'High-volume attacks are already high at 20 labels, so more labels add little',
    # 5
    'PCAP에서 feature까지': 'From PCAP to Features',
    '라벨은 feature 계산에 쓰지 않음  ·  PCA를 학습 데이터로만 맞추기 위해 시간순 분할이 PCA보다 먼저': 'Labels are not used to compute features  ·  The time split comes first so PCA is fit on training data only',
    'CIC2017 월~금\n하루 1개, 파일 5개\n51 GB': ('CIC2017 Mon to Fri\nOne file per day\n5 files, 51 GB', 15),
    '② flow로 묶기': '② Group into flows',
    '프로토콜·IP·포트 쌍이 같은 패킷\nRST · 양쪽 FIN 뒤 ACK\n새 SYN · 120초에서 종료': ('Same protocol, IPs and ports\nEnds at RST · ACK after both FINs\nnew SYN · 120 s', 13.5),
    '③ 라벨 붙이기': '③ Label',
    '공격자·피해자 IP인가\n공격 시간창 안인가\n공격 행위 증거가 있는가': ('Attacker and victim IPs?\nInside the attack window?\nEvidence of attack behavior?', 13.5),
    '④ 시간순 분할': '④ Time split',
    '날짜·공격별로\n앞 70% 학습\n다음 15% · 뒤 15% 평가': ('By day and attack\nFirst 70% train\nNext 15% val · last 15% test', 13),
    '⑧ 평가': '⑧ Evaluate',
    '⑤ 토큰화': '⑤ Tokenize',
    'IP  헤더 길이·TOS·전체 길이·flag·TTL\nTCP  flag·window·seq·ack·urgent\nIP·포트·시각·payload 제외': ('IP  hdr len, TOS, length, flags, TTL\nTCP  flags, window, seq, ack, urgent\nNo IPs, ports, times or payload', 13.5),
    '토큰열 → 512차원\n가중치 고정': 'Tokens → 512 dims\nFrozen weights',
    '512차원 → 64차원\n학습 데이터로 맞춤': '512 → 64 dims\nFit on training data',
    '패킷을 flow로 묶고 라벨을 붙인 뒤, 헤더 토큰을 인코더에 넣어 flow당 64차원 feature를 생성': 'Packets become labelled flows, then header tokens pass through the encoder to give 64 features per flow',
    '03 파이프라인': '03 Pipeline',
    # 6
    '패킷의 생김새': 'What a Packet Looks Like',
    '패킷은 네트워크로 오가는 데이터 조각 하나  ·  아래는 월요일 사내 PC가 웹서버로 보낸 실제 패킷 151바이트': 'A packet is one piece of data on the network  ·  Below: a real 151-byte packet from an office PC to a web server',
    '이더넷 14 B': 'Ethernet 14 B',
    'IP 헤더 20 B': 'IP header 20 B',
    'TCP 헤더 20 B': 'TCP header 20 B',
    '바로 옆 장비까지\n전달하기 위한 표\n\n보낸 장비 번호\n받는 장비 번호': ('Delivery to the\nnext device\n\nSource device\nDestination device', 14.5),
    '어느 컴퓨터로 갈지 적는 칸\n\n보낸 주소  192.168.10.9\n받는 주소  69.31.33.224\n전체 길이  137 B\nTTL  128\n프로토콜  6번 (TCP)': 'Which computer\n\nSource  192.168.10.9\nDestination  69.31.33.224\nTotal length  137 B\nTTL  128\nProtocol  6 (TCP)',
    '어느 프로그램으로 갈지 적는 칸\n\n보낸 포트  1057\n받는 포트  80\nseq  2039531092\nack  2775650109\n신호  ACK+PSH\nwindow  256': 'Which program\n\nSource port  1057\nDestination port  80\nseq  2039531092\nack  2775650109\nFlags  ACK+PSH\nwindow  256',
    '실제 내용\n\nGET /ncsi.txt HTTP/1.1\nConnection: Close\nUser-Agent: Microsoft NCSI\nHost: www.msftncsi.com': 'The actual content\n\nGET /ncsi.txt HTTP/1.1\nConnection: Close\nUser-Agent: Microsoft NCSI\nHost: www.msftncsi.com',
    '패킷은 겉 정보인 헤더와 속 내용인 payload로 구성되고, 본 연구는 payload 내용을 읽지 않음': 'A packet has headers outside and payload inside, and this work does not read payload content',
    # 7
    '같은 대화의 패킷을 한 flow로 묶기': 'Grouping One Conversation into a Flow',
    'flow = 두 컴퓨터의 두 프로그램이 같은 약속으로 주고받은 패킷 묶음, 즉 대화 한 건': 'A flow is the packets two programs on two computers exchange under one protocol: one conversation',
    '프로토콜': 'Protocol',
    '포트 A': 'Port A',
    '포트 B': 'Port B',
    '다섯 값이\n모두 같으면': 'If all five\nvalues match',
    '같은 flow  ·  보낸 쪽과 받는 쪽이 바뀌어도 같은 flow': 'Same flow  ·  even when sender and receiver swap',
    '패킷': 'Packet',
    '보낸 쪽': 'From',
    '받는 쪽': 'To',
    '묶인 결과': 'Result',
    '요청': 'Request',
    '응답': 'Response',
    'flow A  방향만 반대': 'flow A  direction reversed',
    '인증 접속 1': 'Auth 1',
    '인증 접속 2': 'Auth 2',
    'flow C  임시 포트가 다름': 'flow C  different temporary port',
    '도메인 조회': 'DNS lookup',
    'flow D  프로토콜과 상대가 다름': 'flow D  other protocol and peer',
    '묶는 이유  패킷 하나로는 행동을 알 수 없고, 대화 전체를 봐야 요청·응답·종료를 판단 가능': 'Why group  One packet says little; the whole conversation shows the request, response and end',
    '다섯 값이 같아도 대화가 끝나면 다음 패킷은 새 flow  ·  끝나는 조건은 다음 장': 'Once a conversation ends, the next packet starts a new flow  ·  End conditions follow',
    '프로토콜과 양쪽 IP·포트 다섯 값이 같은 패킷이 한 flow': 'Packets that share the protocol and both IPs and ports form one flow',
    # 8
    'flow의 끝: FIN과 RST': 'Where a Flow Ends: FIN and RST',
    '→ PC가 보냄   ← 서버가 보냄   ·   소속 칸이 각 패킷의 처리   ·   두 대화 모두 월요일 실제 패킷': '→ PC sends   ← server sends   ·   The last column shows how each packet is handled   ·   Real Monday packets',
    '케이스 1  FIN으로 끝남  ·  192.168.10.9:1057 ↔ 69.31.33.224:80': 'Case 1  Ends with FIN  ·  192.168.10.9:1057 ↔ 69.31.33.224:80',
    '시각 (µs)': 'Time (µs)',
    '방향': 'Dir',
    '길이': 'Len',
    '소속': 'Flow',
    'flow A 마감': 'flow A closes',
    '케이스 2  RST로 끝남  ·  192.168.10.51:38488 ↔ 209.15.36.31:80': 'Case 2  Ends with RST  ·  192.168.10.51:38488 ↔ 209.15.36.31:80',
    'flow B 마감': 'flow B closes',
    '버림': 'dropped',
    'PC가 4번 FIN으로 먼저 닫아 도착하는 5·7번마다 RST로 응답': 'The PC closed first with FIN at 4, so it answers 5 and 7 with RST',
    '첫 RST까지가 한 flow  ·  뒤따르는 패킷은 앞 대화의 꼬리라 버림': 'The flow ends at the first RST  ·  Later tail packets are dropped',
    'TCP 신호': 'TCP flags',
    'SYN 연결 요청  ·  ACK 받았다는 확인\nPSH 내용 바로 전달  ·  FIN 보낼 것이 끝남\nRST 연결을 버림, 확인 없이 즉시 종료': 'SYN open  ·  ACK acknowledge\nPSH deliver now  ·  FIN done sending\nRST abort at once, no reply needed',
    '버리는 꼬리 패킷': 'Dropped tail packets',
    '닫힌 연결에서 2초 안에 도착\nSYN·payload 없는 패킷 2개 이하': 'Arrive within 2 s of the close\nAt most 2 packets, no SYN or payload',
    'flow가 끝나는 때': 'When a flow ends',
    'RST  ·  양쪽 FIN 뒤 ACK  ·  같은 끝점의 새 SYN\n120초 무통신  ·  시작 후 120초\nUDP·ICMP는 시간 조건만 사용': 'RST  ·  ACK after both FINs  ·  New SYN\n120 s idle  ·  120 s since start\nUDP and ICMP use time only',
    'FIN 종료는 마지막 ACK까지, RST 종료는 첫 RST까지가 한 flow이고 뒤따르는 패킷은 버림': 'A FIN flow runs to the last ACK, an RST flow to the first RST, and later packets are dropped',
    # 9
    '헤더 토큰에서 netFound feature로': 'From Header Tokens to netFound Features',
    'flow의 패킷마다 헤더 값을 숫자 토큰으로 옮겨 netFound에 입력  ·  아래는 케이스 1 flow A의 패킷': 'Each packet header becomes numeric tokens fed to netFound  ·  Below: packets of flow A in case 1',
    'IP 헤더\n길이': 'IP hdr\nlen',
    '전체\n길이': 'Total\nlen',
    'seq\n앞': 'seq\nhigh',
    'seq\n뒤': 'seq\nlow',
    'ack\n앞': 'ack\nhigh',
    'ack\n뒤': 'ack\nlow',
    '1번 SYN': 'Packet 1 SYN',
    '2번 SYN+ACK': ('Pkt 2 SYN+ACK', 12),
    '4번 요청': 'Packet 4 request',
    'flow 한 건의 헤더 토큰  ·  패킷 1개 = 숫자 12개  ·  IP·포트·시각·payload 내용 제외': 'Header tokens of one flow  ·  1 packet = 12 numbers  ·  No IPs, ports, times or payload content',
    '헤더 토큰': 'Header tokens',
    'flow의 패킷 순서대로\n간격·방향 함께 입력': 'In packet order, with\ngaps and direction',
    '사전학습 transformer\n가중치 고정': 'Pretrained transformer\nFrozen weights',
    'flow 한 건 = 64개\n분류기의 한 행': 'One flow = 64 values\nOne classifier row',
    'netFound  네트워크 트래픽용 사전학습 transformer': 'netFound  a transformer pretrained on network traffic',
    '라벨 없는 트래픽에서 가려진 헤더 토큰을 맞히며 학습, 본 실험은 인코더로만 사용': 'Learns by predicting masked header tokens in unlabelled traffic, used here as an encoder only',
    'flow의 헤더 값을 토큰으로 netFound에 넣어 flow 한 건당 feature 64개를 얻음': 'Flow headers enter netFound as tokens and each flow yields 64 features',
    # 10
    'flow에 라벨 붙이기': 'Labelling Flows',
    '각 flow를 왼쪽부터 판정  ·  CNS 2022의 교정 규칙 53개를 그대로 적용': 'Each flow is judged from left to right  ·  The 53 corrected rules of CNS 2022 are applied as published',
    '공격자 IP ↔\n피해자 IP인가': 'Attacker IP ↔\nvictim IP?',
    '공격 시간창\n안인가': 'Inside the\nattack window?',
    '공격 행위의\n증거가 있는가': 'Evidence of\nattack behavior?',
    '예': 'Yes',
    '공격': 'Attack',
    '아니오': 'No',
    '없음': 'No',
    '정상': 'Benign',
    '보류': 'Unresolved',
    '라벨': 'Label',
    '조건': 'Condition',
    '처리': 'Handling',
    'IP·시간창·증거를 모두 만족': 'IPs, window and evidence all met',
    '학습·평가에 공격으로 사용': 'Used as attack',
    'IP·시간창은 맞지만 증거 없음': 'IPs and window match, no evidence',
    '학습·평가에서 제외': 'Excluded from train and test',
    'IP나 시간창이 규칙 밖': 'IPs or window outside the rules',
    '학습·평가에 정상으로 사용': 'Used as benign',
    'CNS 2022  Liu 외': 'CNS 2022  Liu et al.',
    'Error Prevalence in NIDS Datasets\n\nCIC-IDS 2017·2018의 flow 생성과\n라벨 오류를 패킷 단위로 조사\n교정한 라벨 규칙과 도구를 공개': 'Error Prevalence in NIDS Datasets\n\nPacket-level study of flow and label\nerrors in CIC-IDS 2017 and 2018\nPublished corrected rules and tools',
    '증거 예  FTP·SSH brute force는 공격자 payload ≥ 1 B': 'Evidence  FTP and SSH brute force: attacker payload ≥ 1 B',
    '증거 예  Portscan은 SYN 1개와 무응답 또는 RST': 'Evidence  Portscan: one SYN with no reply or RST',
    'IP·시간창·공격 행위 증거 세 조건으로 공격, 보류, 정상을 나누고 보류는 학습·평가에서 제외': 'IPs, window and evidence split flows into attack, unresolved and benign; unresolved flows are excluded',
    '04 라벨': '04 Labels',
    # 11
    '최종 데이터 구성': 'Final Data',
    '잔여 flow 삭제 후 분할 전 기준  ·  CSV는 공식 라벨 행 수, PCAP은 라벨 있는 flow 수  ·  행 순서는 PCAP flow 수 내림차순': 'After deleting residual flows, before the split  ·  CSV: official label rows; PCAP: labelled flows  ·  Ordered by PCAP flows',
    '세부 공격 (PCAP flow 수)': 'Attacks (PCAP flows)',
    'CSV 행': 'CSV rows',
    '월요일 전일 + 공격일 배경': 'All of Monday + background of attack days',
    '금 Portscan 161,202 · 목 NMAP 76,947': 'Fri Portscan 161,202 · Thu NMAP 76,947',
    'CSV와 수가 다른 이유  flow 경계와 라벨 규칙 교정  ·  목요일 NMAP은 CSV에서 BENIGN, 교정 규칙에서 portscan': 'Counts differ from CSV because flow boundaries and labels are corrected  ·  Thursday NMAP is BENIGN in CSV, portscan here',
    'inf는 학습 25개·test 6개, heartbleed는 학습 7개·test 2개로 표본이 매우 적음': 'inf has 25 train and 6 test flows, and heartbleed has 7 train and 2 test flows',
    '라벨 있는 flow 2,123,394개, 9종을 학습·평가': '2,123,394 labelled flows across nine classes',
    # 12
    '문제1 재점검: 동일 feature·상이 라벨': 'Problem 1 Revisited: Label Conflicts',
    '잔여 flow 삭제 후  ·  feature가 같은데 class가 다른 flow  ·  패킷이 매우 적은 flow에 집중': 'After deleting residual flows  ·  Same features, different classes  ·  Concentrated in very short flows',
    '충돌 flow': 'Conflicts',
    '비율': 'Share',
    'CSV 통계 78개': 'CSV 78 statistics',
    '남은 충돌  Portscan 42,609 · 정상 770 · DoS 1': 'Remaining  Portscan 42,609 · benign 770 · DoS 1',
    '패킷 수': 'Packets',
    'flow 수': 'Flows',
    '입력 같은 %': 'Same input %',
    '21 이상': '21 or more',
    '원인': 'Causes',
    '패킷이 매우 적은 flow는 숫자가 적고 대부분 고정값\n예  스캔 SYN과 거절 RST, 패킷 2개\n\nnetFound가 입력 전에 버리는 정보\nIP · 포트 · 시각 · payload 내용': 'Very short flows have few numbers, mostly fixed\nExample  scan SYN and refusing RST, 2 packets\n\nInformation netFound drops before input\nIPs · ports · times · payload content',
    '추가 정보로 극복 가능': 'Side information can help',
    'netFound 벡터 뒤에 열을 붙여 분류기에 입력, IP·포트 값은 넣지 않음': 'Columns behind the netFound vector, without IP or port values',
    '감소': 'Reduced',
    '+ 둘 다': '+ both',
    '최근 60초 포트 수 하나로 충돌 flow 97.8% 감소': 'The 60 s port count alone cuts conflict flows by 97.8%',
    '남은 충돌 2.04%는 패킷이 매우 적은 flow와 netFound가 버리는 정보에서 발생, 추가 정보로 0.05%까지 감소': 'The remaining 2.04% comes from very short flows and dropped fields; side information cuts it to 0.05%',
    '05 결과': '05 Results',
    # 13
    '문제2 재점검: few-shot 성능': 'Problem 2 Revisited: Few-Shot Performance',
    '전체 9종  ·  시간순 분할  ·  seed 5개  ·  두 모델에 같은 학습 행  ·  netFound는 잔여 flow 삭제 데이터': 'All nine classes  ·  Time split  ·  5 seeds  ·  Same training rows  ·  netFound after deleting residual flows',
    'CSV와 다른 점': 'Difference from CSV',
    'CSV는 두 모델이 비슷\nnetFound는 TabPFN이 크게 우위\n잔여 삭제 전후 7종 n=50 −0.5 pp': ('Both models are close on CSV\nTabPFN leads clearly on netFound\nDeletion: −0.5 pp (7 classes, n=50)', 13.5),
    'TabPFN이 우위인 이유  n=50, 삭제 전 측정': 'Why TabPFN leads  n=50, before deletion',
    '학습과 같은 입력의 정확도는 두 모델 모두 97.0%\n나머지 행은 XGB 91.1% · TabPFN 97.5%\n처음 보는 입력에서 격차 발생\nPCA 벡터는 열 단위 분할에 불리 (추정)': 'Rows equal to training inputs: both models 97.0%\nOther rows: XGB 91.1% · TabPFN 97.5%\nThe gap comes from unseen inputs\nPCA vectors suit axis splits poorly (inferred)',
    'netFound + TabPFN n=50 class별 F1': 'netFound + TabPFN n=50 F1 by class',
    'benign 97.9 · portscan 97.3 · dos 89.9 · ddos 99.1 · bf 94.6\nbot 35.2 · web 10.0 · inf 0.2 · heartbleed 5.1\ninf·heartbleed는 test 6개·2개, 재현율은 높고 정밀도 낮음': 'benign 97.9 · portscan 97.3 · dos 89.9 · ddos 99.1 · bf 94.6\nbot 35.2 · web 10.0 · inf 0.2 · heartbleed 5.1\ninf and heartbleed: 6 and 2 test flows, high recall, low precision',
    'few-shot에서 netFound + TabPFN이 가장 높고, 격차는 학습과 겹치지 않는 입력에서 발생': 'netFound + TabPFN scores highest in few-shot, and the gap comes from inputs unseen in training',
    # 14
    '추가 정보를 붙인 few-shot': 'Few-Shot with Side Information',
    '전체 9종 macro-F1  ·  netFound 벡터 뒤에 열 추가  ·  seed 5개  ·  같은 학습 행  ·  잔여 flow 삭제 데이터': 'Nine-class macro-F1  ·  Columns behind the netFound vector  ·  5 seeds  ·  Same training rows  ·  Residual flows deleted',
    '입력': 'Input',
    'val 차이 pp\nTabPFN n=50': 'val gain pp\nTabPFN n=50',
    '정상 오탐 %\nTabPFN n=50': 'False alarm %\nTabPFN n=50',
    '+ 원본 seq 표시': '+ raw seq flag',
    '+ 최근 60초 포트 수': '+ ports in the last 60 s',
    '최근 60초 포트 수': 'Ports in the last 60 s',
    '원본 seq 표시': 'Raw seq flag',
    'TabPFN n=50 +1.4 pp, seed 5개 중 4개 상승\nvalidation에서도 +2.3 pp\n\n정상 오탐 2.97 → 2.48%\nXGB Portscan F1 94.2 → 98.5': 'TabPFN n=50 +1.4 pp, up in 4 of 5 seeds\nValidation also +2.3 pp\n\nBenign false alarms 2.97 → 2.48%\nXGB Portscan F1 94.2 → 98.5',
    'TabPFN n=50 +0.4 pp, seed 5개 중 2개 상승\nvalidation은 −0.4 pp\n\n변화가 seed 편차 안에 있어 효과 불분명': 'TabPFN n=50 +0.4 pp, up in 2 of 5 seeds\nValidation −0.4 pp\n\nChanges stay within seed spread\nso the effect is unclear',
    '최근 60초 포트 수는 validation과 test에서 소폭 개선, 원본 seq 표시는 효과가 불분명': 'Ports in the last 60 s help slightly on validation and test; the raw seq flag shows no clear effect',
    # 14
    '정리와 다음 단계': 'Summary and Next Steps',
    '문제1  동일 feature·상이 라벨': 'Problem 1  Label conflicts',
    '잔여 flow 삭제로 4.86 → 2.04%\n\n남은 충돌은 패킷이 적은 스캔\n60초 포트 수로 0.05%까지 감소': 'Deleting residual flows\n4.86 → 2.04%\n\nRemaining: short scan flows\n60 s port count: down to 0.05%',
    '문제2  few-shot 성능': 'Problem 2  Few-shot',
    'few-shot은 여전히 잘 동작\n9종 n=50 macro 58.8%, CSV 53.6%\n\n대량 공격은 소수 라벨로 포화\nbot·web·inf는 여전히 낮음': ('Few-shot still works well\nNine-class n=50: 58.8%, CSV 53.6%\n\nHigh-volume attacks saturate early\nbot, web and inf remain low', 15),
    'PCAP 직접 가공의 장점': 'Why PCAP directly',
    'flow 경계와 라벨을 직접 교정\n\nCSV에 없는 패킷 순서·맥락 사용\n\n입력 구성을 실험 가능': ('Correct flow boundaries and labels\n\nUse packet order and context\n\nExperiment with input design', 15),
    '앞으로': 'Next',
    'PCAP으로 가공할 수 있는 데이터 추가 탐색': 'Find more datasets available as PCAP',
    '모델': 'Models',
    'netFound feature로 SOTA와 제안 모델 실험': 'Test SOTA and proposed models on netFound features',
    '최근 60초 포트 수를 입력에 포함해 모델 비교': 'Compare models with the 60 s port count in the input',
    'PCAP을 직접 가공하면 데이터 문제를 고칠 수 있고, 다음은 데이터 확장과 SOTA·제안 모델 실험': 'Direct PCAP processing fixes data problems; next come more data and SOTA and proposed models',
    '06 정리': '06 Summary',
}
# Same Korean text in a narrow place.
TR_SIZED = {('이더넷 14 B', 12.5): 'Eth 14 B', ('보류', 13.5): ('Unresolved', 11.5)}

NOTES = [
    'Prior experiments fed CICFlowMeter CSV vectors to XGB, TabPFN, SOTA models and the proposed models, and tried to express evaluation scenarios. Classification performance was weak. The data also held rows with identical features and different labels. And even in few-shot some attacks scored so high that little room was left to improve. So this report replaces CSV with PCAP and builds features with netFound, a transformer encoder for network data, to test whether those problems improve.',
    'CIC-IDS data is captured in a testbed as PCAP, then CICFlowMeter groups packets into flows and computes fixed statistics. Fixed means the same 78 formulas for every flow, such as duration, packets sent and mean packet length. The main cause of identical features with different labels is a flow boundary defect. Engelen et al. report that CICFlowMeter ends a flow at the first FIN and does not treat RST as an end, so leftover ACK and FIN packets become their own flows. These TCP appendices make up 25.9% of CIC2017 and inherit the label of the original flow. With only one or two packets their statistics easily match, while labels follow the parent, so identical statistics carry different labels. When we rebuilt flows from PCAP, removing tail flows cut statistics conflicts from 6,367 to 69. The table shows real groups with identical statistics.',
    'These are rows in groups with identical statistics but different classes, counted by class on both original CSVs. CIC2017 has 7,020 such rows, 0.25%, led by dos at 2.07%. CIC2018 has 406,811 rows, 2.51%, concentrated in attacks: bf 42.65%, inf 25.49%, dos 11.81%. The full CIC2018 PCAP is 450 GB, so experiments use CIC2017, whose whole-network capture is 51 GB.',
    'This is the few-shot curve on CIC2017 CSV with attacks only. High-volume attacks such as portscan, ddos and dos reach high F1 at 20 labels per class, so more labels add little. web stays low, and bot at 0 comes from a CSV label defect: every bot row in the evaluation set is a reconnect after the master stopped. The earlier version of this figure used CIC2018 and is redrawn here for CIC2017.',
    'CIC2017 was captured at a switch from Monday to Friday, one file per day, so there are five PCAP files totalling 51 GB. Packets are grouped into flows and each flow is labelled. Flows are then split in time by day and attack. The split comes before PCA because PCA is fit on training data only. Labels are not used in feature computation, so labelling first does not affect features. Headers then become tokens, netFound produces a vector, PCA reduces it to 64 dimensions, and the result is evaluated.',
    'Data on a network travels in small pieces called packets. This is a real packet an office PC sent to a web server on Monday, a Windows connectivity check. The first 54 bytes are headers and the last 97 are content. The Ethernet header handles the next device, the IP header the destination computer, and the TCP header the program and connection state. Like a parcel, headers are the label and payload is the contents. The netFound view uses header values only.',
    'One packet says little, so packets are grouped into conversations called flows. The key is five values: protocol, both IPs and both ports. A request and its response are the same flow even though sender and receiver swap. A different temporary port means another flow, and so does a different protocol or peer.',
    'In case 1, packets 7 and 9 are the two FINs and flow A closes at the ACK in packet 10. In case 2 the PC closed first with FIN at packet 4. When server packets 5 and 7 arrive late, the PC answers each with RST because the connection is already closed. The flow closes at the first RST, so packets 1 to 6 are flow B. There are two reasons. In the TCP specification RST resets the connection, and Engelen et al. identified ignoring RST as a defect and made RST end the flow. Packets 7 and 8 are tail packets of a finished conversation, so they are dropped. These tails formed 284,600 residual flows, 11.7% of all flows, and were a main source of conflicts. Deleting them removes 6.5% of labelled flows and 0.57% of packets, with no loss in rare classes.',
    'Each flow is a sequence of packets as in the table. Header fields of each packet become numeric tokens, 12 numbers per packet, with seq and ack as relative values. The tokens enter netFound in packet order. netFound is a transformer pretrained on network traffic by predicting masked header tokens in unlabelled traffic, and here it is a frozen encoder. It outputs a 512-dimensional vector per flow, and PCA fit on training data reduces it to 64 dimensions. So one flow becomes one row with 64 features.',
    'Each flow is judged from left to right. CNS 2022 is the paper by Liu et al. that studied flow and label errors in CIC-IDS 2017 and 2018 at packet level and published corrected labels and tools. We apply its 53 rules as published. A flow outside the attacker and victim IPs or outside the time window is benign. If both match but there is no evidence of attack behavior, it is unresolved. If all three hold, it is an attack. Unresolved flows are attempts without effect, so following the authors they are not used as a class and are excluded from training and evaluation.',
    'This is the resulting data after deleting residual flows, with 2,123,394 labelled flows. Counts differ from CSV because flow boundaries and label rules are corrected. For example, Thursday NMAP scans are BENIGN in CSV and portscan under the corrected rules. inf and heartbleed have fewer training flows than the budget of 50, so they use what exists, and with 6 and 2 test flows their scores vary widely.',
    'Since residual flows are deleted, this is the state after deletion. netFound features still conflict for 2.04% of flows, above the 0.25% of CSV. Two causes combine. Very short flows have few numbers, mostly fixed values, and netFound drops IPs, ports, times and payload content before input. Of 43,380 conflict flows, 42,443 have two packets, and the 12-burst limit plays no part. Side information can overcome this. The published weights fix the input format, so the encoder stays as is and columns are added behind the vector. The count of service ports between two IPs in the last 60 seconds cuts conflict flows from 43,380 to 964, a 97.8% reduction. The raw seq flag leaves 42,571, a 1.9% reduction. The conflict count includes every flow of a mixed input, however, and the flows that stay wrong with the best label per input fall from 212 to 64. Both together leave 302 flows, or 0.01%.',
    'These are few-shot results over all nine classes. On CSV, XGB and TabPFN are close. On netFound features TabPFN leads clearly, and its n=50 macro-F1 of 58.8 exceeds the 53.6 on CSV. On test rows equal to training inputs the two models match, so the gap comes from unseen inputs. One reading is that CSV columns each carry meaning and suit axis-aligned splits, while single PCA columns carry little meaning, which favors TabPFN comparing whole vectors. This reading is inferred. inf and heartbleed have only 25 and 7 training flows and 6 and 2 test flows. On average 7,089 rows are predicted as these two classes and only 8 are correct. Of the wrong rows, 72% are benign and 27% are DoS. Their F1 is therefore low and the nine-class macro-F1 sits below the earlier seven-class value. Treat these two scores as indicative only. netFound scores come from data with residual flows deleted, keeping the EXP94 training rows and PCA and replacing only residual rows. On seven classes and the same test rows, deletion changes n=50 by -0.5 pp for TabPFN and -0.1 pp for XGB, which is effectively no change.',
    'The two side columns were added behind the netFound vector and all nine classes were evaluated again, as paired runs on the same training rows. The count of ports in the last 60 seconds adds 1.4 pp for TabPFN at n=50 and 2.3 pp on validation, and benign false alarms fall. XGB Portscan F1 rises from 94.2 to 98.5, so the scan separation this column targets did improve. The raw seq flag stays within seed spread and its validation sign is mixed, so its effect is unclear. Neither column targets rare attacks such as bf or bot.',
    'Label conflicts fell by deleting residual flows; what remains is short scan traffic. The 60 second port count cuts conflicts to 0.05% and improves few-shot slightly. Few-shot still works well and netFound with TabPFN beats CSV, but high-volume attacks still saturate and bot, web and inf stay low. Processing PCAP directly lets us correct flow boundaries and labels, use packet order and context absent from CSV, and experiment with input design. Next we will look for more datasets and test SOTA and proposed models.',
]


def english_saturation() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    font, bold = ko.fonts()
    classes = ['dos', 'portscan', 'ddos', 'bf', 'bot', 'web']
    budgets = [2, 3, 5, 10, 20, 50]
    fig = plt.figure(figsize=(13.4, 4.2), dpi=190)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=.2)
    ax = fig.add_subplot(grid[0])
    for method, label, color in (('xgb', 'XGB', '#B44948'), ('tabpfn', 'TabPFN', '#007BC6')):
        m = ko.csv_macro('attack_only', method)
        ys = [m[n][0] for n in budgets]
        ax.plot(budgets, ys, marker='o', ms=5.4, lw=2.4, color=color, label=label)
        ax.annotate(f'{ys[-1]:.1f}', (50, ys[-1]), xytext=(6, 3 if method == 'tabpfn' else -12), textcoords='offset points', color=color, fontsize=10, fontproperties=bold)
    ax.set_xscale('log', base=2)
    ax.set_xticks(budgets, [str(n) for n in budgets], fontproperties=font, fontsize=10)
    ax.set_ylim(35, 90); ax.set_ylabel('macro-F1 (%)', fontproperties=font, fontsize=11)
    ax.set_xlabel('Training labels per class, n', fontproperties=font, fontsize=11)
    ax.set_title('Macro-F1 over six attacks', fontproperties=bold, fontsize=14, pad=9)
    ax.grid(axis='y', color='#D9E1E8', lw=.75); ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='lower right', prop=font)
    ax = fig.add_subplot(grid[1])
    M = np.array([[ko.csv_class('attack_only', 'tabpfn', n)[c] for n in budgets] for c in classes])
    ax.imshow(M, aspect='auto', cmap='Blues', vmin=0, vmax=100)
    for yy in range(len(classes)):
        for xx in range(len(budgets)):
            ax.text(xx, yy, f'{M[yy, xx]:.0f}', ha='center', va='center', fontsize=11, color='white' if M[yy, xx] >= 62 else '#172033', fontproperties=bold)
    ax.set_xticks(range(len(budgets)), [f'n={n}' for n in budgets], fontproperties=font, fontsize=10.5)
    ax.set_yticks(range(len(classes)), classes, fontproperties=bold, fontsize=11)
    ax.set_title('F1 by class (%) · TabPFN', fontproperties=bold, fontsize=14, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(ASSETS / 'saturation_cic2017.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def english_fewshot(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    font, bold = ko.fonts()
    budgets = [2, 3, 5, 10, 20, 50]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.45), dpi=200, sharey=True)
    for ax, title, get in ((axes[0], 'CSV 78 statistics', lambda m: ko.ac_curve('csv', m)), (axes[1], 'PCAP + netFound 64', lambda m: ko.ac_curve('netfound', m))):
        for method, label, color in (('xgb', 'XGB', '#B44948'), ('tabpfn', 'TabPFN', '#007BC6')):
            v = get(method)
            ys = [v[n][0] for n in budgets]
            ax.errorbar(budgets, ys, yerr=[v[n][1] for n in budgets], color=color, marker='o', ms=4.6, lw=2.1, capsize=2.4, label=label)
            ax.annotate(f'{ys[-1]:.1f}', (50, ys[-1]), xytext=(5, 4 if method == 'tabpfn' else -12), textcoords='offset points', color=color, fontsize=9.5, fontproperties=bold)
        ax.set_xscale('log', base=2)
        ax.set_xticks(budgets, [str(n) for n in budgets], fontproperties=font, fontsize=9)
        ax.set_ylim(10, 70); ax.set_title(title, fontproperties=bold, fontsize=12.5, pad=7)
        ax.set_xlabel('Training labels per class, n', fontproperties=font, fontsize=9.6)
        ax.grid(axis='y', color='#D9E1E8', lw=.75); ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='y', labelsize=8.8)
    axes[0].set_ylabel('macro-F1 (%) · nine classes', fontproperties=font, fontsize=9.6)
    axes[1].legend(frameon=False, loc='lower right', prop=font)
    fig.subplots_adjust(left=.075, right=.975, top=.9, bottom=.165, wspace=.09)
    fig.savefig(ASSETS / 'fewshot_csv_netfound.png', facecolor='white')
    plt.close(fig)


def translate() -> None:
    missing: list[str] = []
    for index, scene in enumerate(r.SCENES):
        for item in scene['items']:
            if item.get('kind') != 'text' or not HANGUL.search(item['text']):
                continue
            value = TR_SIZED.get((item['text'], item['size']), TR.get(item['text']))
            if value is None:
                missing.append(f"{index + 1}: {item['text']!r}")
                continue
            text, size = value if isinstance(value, tuple) else (value, item['size'])
            lines_old, lines_new = item['text'].count('\n') + 1, text.count('\n') + 1
            if lines_old != lines_new:
                item['h'] = item['h'] * lines_new / lines_old
            item['text'], item['size'] = text, size
        scene['title'] = next(it['text'] for it in scene['items'] if it.get('kind') == 'text')
        sources = scene['notes'].split('\n\n원문 근거\n', 1)[1]
        scene['notes'] = 'Speaker notes\n' + NOTES[index] + ('\n\nSources\n' + sources if sources.strip() else '')
    if missing:
        raise KeyError('missing translations:\n' + '\n'.join(missing))


def main() -> None:
    QA.mkdir(parents=True, exist_ok=True); ASSETS.mkdir(parents=True, exist_ok=True)
    for old in list(QA.glob('page-*.png')) + list(ASSETS.glob('*.png')):
        old.unlink()
    r.QA, r.ASSETS, r.OUT, r.N_SLIDES = QA, ASSETS, OUT, 15
    r.SOURCES = {key: (value['title'], f"{value['venue']}, {value['year']}", value['url'], value['evidence']) for key, value in b04.SOURCE_DATA.items()}
    ko.ASSETS = ASSETS
    ko.saturation_chart, ko.fewshot_chart = english_saturation, english_fewshot
    r.SCENES = []
    recorded: list[dict] = []
    b04.page = lambda title, items, summary, section, notes, sources: recorded.append(dict(title=title, items=items, summary=summary, section=section, notes=notes, sources=sources))
    b04.scenes()
    for extra in ('layer_conflicts', 'fewshot_three_views'):
        (ASSETS / f'{extra}.png').unlink(missing_ok=True)
    r.SCENES = []
    ko.scenes(recorded[3])
    translate()
    r.validate_scenes()
    r.pptx()
    from pptx import Presentation
    presentation = Presentation(OUT.with_suffix('.pptx'))
    presentation.core_properties.title = '11/06 From CSV to PCAP'
    presentation.core_properties.subject = 'Label conflicts and few-shot performance with PCAP and netFound features'
    presentation.save(OUT.with_suffix('.pptx'))
    checks = r.pdf()
    for index in range(1, len(checks) + 1):
        source = QA / f'page-{index}.png'
        target = QA / f'page-{index:02d}.png'
        if source.exists():
            source.replace(target)
    from PIL import Image, ImageDraw
    for old in QA.glob('contact-*.png'):
        old.unlink()
    paths = sorted(QA.glob('page-*.png'))
    canvas = Image.new('RGB', (1500, 8 * 445), '#DCE3EB')
    for index, image_path in enumerate(paths):
        im = Image.open(image_path)
        im.thumbnail((730, 411))
        x, y = (index % 2) * 750 + 10, (index // 2) * 445 + 8
        canvas.paste(im, (x, y))
        ImageDraw.Draw(canvas).text((x, y + 414), str(index + 1), fill='black')
    canvas.save(QA / 'contact-1.png')
    (QA / 'scenes.json').write_text(json.dumps(r.SCENES, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    visible = '\n'.join(it.get('text', '') for scene in r.SCENES for it in scene['items'])
    if HANGUL.search(visible) or any(HANGUL.search(scene['notes']) for scene in r.SCENES):
        raise ValueError('Korean text left in the English deck')
    manifest = {'date': '2026-11-06', 'slides': len(r.SCENES), 'korean_source': 'kor/20261106_PCAP_통합.pptx',
                'validation': 'shared editable PPTX/HTML-PDF scenes; geometry/font/text checks; visual contact sheet reviewed', 'pages': checks,
                'pptx_sha256': hashlib.sha256(OUT.with_suffix('.pptx').read_bytes()).hexdigest(), 'pdf_sha256': hashlib.sha256(OUT.with_suffix('.pdf').read_bytes()).hexdigest()}
    (QA / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pptx': str(OUT.with_suffix('.pptx')), 'pdf': str(OUT.with_suffix('.pdf')), 'slides': len(r.SCENES)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
