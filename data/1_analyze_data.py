import pandas as pd
from pathlib import Path
from collections import defaultdict

def analyze_dataset(root_path, dataset_name, label_column='Label'):
    """데이터셋의 클래스별 통계를 분석 (메모리 효율적)"""
    print(f"\n{'='*80}")
    print(f"📊 {dataset_name} 데이터셋 분석")
    print(f"{'='*80}")
    
    # CSV 파일 모두 찾기
    csv_files = list(Path(root_path).glob("*.csv"))
    print(f"CSV 파일 개수: {len(csv_files)}")
    print(f"Label 컬럼: {label_column}")
    
    # 클래스별 카운트를 누적할 딕셔너리
    class_counts = defaultdict(int)
    total_samples = 0
    
    # 각 파일을 청크 단위로 읽어서 통계만 누적
    for f in csv_files:
        try:
            file_samples = 0
            # 청크 단위로 읽기 (메모리 절약)
            for chunk in pd.read_csv(f, chunksize=50000):
                # Label 컬럼 확인
                if label_column not in chunk.columns:
                    print(f"  ✗ {f.name}: '{label_column}' 컬럼이 없습니다")
                    break
                
                # 잘못된 레이블 제거 (CIC-IDS의 경우 'Label' 값이 들어있는 행)
                if label_column == 'Label':
                    chunk = chunk[chunk[label_column] != 'Label']
                
                # 클래스별 카운트
                counts = chunk[label_column].value_counts()
                for label, count in counts.items():
                    class_counts[label] += count
                
                file_samples += len(chunk)
            
            total_samples += file_samples
            print(f"  ✓ {f.name}: {file_samples:,} rows")
        except Exception as e:
            print(f"  ✗ {f.name}: 오류 - {e}")
    
    print(f"\n총 샘플 수: {total_samples:,}개")
    
    # 클래스별 통계 출력 (내림차순 정렬)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'클래스명':<40} {'샘플 수':>15} {'비율(%)':>12}")
    print("-" * 70)
    
    for label, count in sorted_classes:
        percentage = (count / total_samples) * 100
        print(f"{label:<40} {count:>15,} {percentage:>11.2f}%")
    
    print("-" * 70)
    print(f"{'총합':<40} {total_samples:>15,} {100.0:>11.2f}%")
    print(f"\n클래스 개수: {len(class_counts)}개")
    
    # 불균형 비율 계산
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count
        print(f"불균형 비율 (IR): {imbalance_ratio:.2f} (최대/최소)")
    
    return dict(class_counts)

# CIC-IDS 2017 분석
cic2017_stats = analyze_dataset(
    './cic2017',
    'CIC-IDS 2017',
    label_column='Label'
)

# CIC-IDS 2018 분석
cic2018_stats = analyze_dataset(
    './cic2018',
    'CIC-IDS 2018',
    label_column='Label'
)

# UNSW-NB15 분석
unswnb15_stats = analyze_dataset(
    './unswnb15',
    'UNSW-NB15',
    label_column='attack_cat'
)

print(f"\n{'='*80}")
print("📈 데이터셋 비교 요약")
print(f"{'='*80}")
print(f"CIC-IDS 2017: {sum(cic2017_stats.values()):,} samples, {len(cic2017_stats)} classes")
print(f"CIC-IDS 2018: {sum(cic2018_stats.values()):,} samples, {len(cic2018_stats)} classes")
print(f"UNSW-NB15:    {sum(unswnb15_stats.values()):,} samples, {len(unswnb15_stats)} classes")

print(f"\n{'='*80}")
print("✅ 분석 완료")
print(f"{'='*80}")
