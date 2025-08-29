# 데이터 업로드 가이드

## CICIDS-2017 데이터셋 업로드 방법

### 1. 로컬에서 데이터 준비
```bash
# 데이터 파일들이 있는 디렉토리 확인
ls MachineLearningCVE/
# Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
# Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
# Friday-WorkingHours-Morning.pcap_ISCX.csv
# Monday-WorkingHours.pcap_ISCX.csv
# Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
# Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
# Tuesday-WorkingHours.pcap_ISCX.csv
# Wednesday-workingHours.pcap_ISCX.csv
```

### 2. 서버로 파일 전송

#### 방법 A: SCP 사용 (권장)
```bash
# 단일 파일 전송
scp MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv username@server:/path/to/destination/

# 전체 디렉토리 전송
scp -r MachineLearningCVE/ username@server:/path/to/destination/

# 압축 후 전송 (네트워크가 느린 경우)
tar -czf MachineLearningCVE.tar.gz MachineLearningCVE/
scp MachineLearningCVE.tar.gz username@server:/path/to/destination/
```

#### 방법 B: rsync 사용
```bash
# 동기화 전송 (중단된 경우 재개 가능)
rsync -avz --progress MachineLearningCVE/ username@server:/path/to/destination/
```

#### 방법 C: SFTP 사용
```bash
# SFTP 클라이언트로 연결
sftp username@server
cd /path/to/destination/
put -r MachineLearningCVE/
```

### 3. 서버에서 압축 해제
```bash
# 서버에 접속
ssh username@server

# 압축 해제 (압축해서 전송한 경우)
tar -xzf MachineLearningCVE.tar.gz

# 권한 설정
chmod 644 MachineLearningCVE/*.csv
```

### 4. 데이터 경로 설정
```bash
# 코드 실행 시 올바른 경로 지정
python final3.py --data_dir /path/to/MachineLearningCVE --epochs 50
```

## 대안 데이터셋 사용

### UNSW-NB15 데이터셋
```bash
# 자동 다운로드
python final3_alternative.py --dataset unsw --epochs 100

# 수동 다운로드
wget https://raw.githubusercontent.com/UNSW-CSE-CS-CC-17-2/UNSW-NB15/master/data/UNSW_NB15_training-set.csv
wget https://raw.githubusercontent.com/UNSW-CSE-CS-CC-17-2/UNSW-NB15/master/data/UNSW_NB15_testing-set.csv
```

### NSL-KDD 데이터셋
```bash
# GitHub에서 다운로드
git clone https://github.com/defcom17/NSL_KDD.git
cd NSL_KDD
# 데이터 파일들을 적절한 위치로 이동
```

## 네트워크 문제 해결

### 느린 네트워크 환경
```bash
# 압축 전송
tar -czf MachineLearningCVE.tar.gz MachineLearningCVE/
scp -C MachineLearningCVE.tar.gz username@server:/path/to/destination/

# 중단된 전송 재개
rsync -avz --partial --progress MachineLearningCVE/ username@server:/path/to/destination/
```

### 방화벽 문제
```bash
# 다른 포트 사용
scp -P 2222 MachineLearningCVE.tar.gz username@server:/path/to/destination/

# 프록시 사용
scp -o ProxyCommand="ssh -W %h:%p proxy_server" MachineLearningCVE.tar.gz username@server:/path/to/destination/
```

## 파일 크기 확인
```bash
# 로컬에서 파일 크기 확인
du -sh MachineLearningCVE/*.csv

# 서버에서 파일 크기 확인
ssh username@server "du -sh /path/to/MachineLearningCVE/*.csv"
```

## 권장사항

1. **압축 전송**: 대용량 파일은 압축 후 전송
2. **rsync 사용**: 중단된 전송 재개 가능
3. **백그라운드 실행**: `nohup` 또는 `screen` 사용
4. **체크섬 확인**: 전송 후 파일 무결성 확인
5. **대안 데이터셋**: CICIDS-2017이 어려운 경우 UNSW-NB15 사용
