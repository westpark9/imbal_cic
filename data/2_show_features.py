import pandas as pd
from pathlib import Path

def show_features(root_path, dataset_name):
    """데이터셋의 모든 CSV 파일에서 feature 이름 출력"""
    print(f"\n{'='*80}")
    print(f"📋 {dataset_name} - Feature 목록")
    print(f"{'='*80}")

    csv_files = sorted(Path(root_path).glob("*.csv"))

    if not csv_files:
        print(f"❌ CSV 파일을 찾을 수 없습니다: {root_path}")
        return None

    all_columns = []
    column_map = {}

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, nrows=0)
            columns = df.columns.tolist()
            column_map[csv_path.name] = columns
            all_columns.append(set(columns))

            print(f"\n🗂️  파일: {csv_path.name} (Feature {len(columns)}개)")
            print(f"{'No.':<6} {'Feature 이름':<60}")
            print("-" * 70)
            # for i, col in enumerate(columns, 1):
                # print(f"{i:<6} {col:<60}")
            print("-" * 70)
        except Exception as e:
            print(f"❌ {csv_path.name} 읽는 중 오류 발생: {e}")

    if not all_columns:
        return None

    union_columns = sorted(set().union(*all_columns))
    intersection_columns = sorted(set.intersection(*all_columns)) if len(all_columns) > 1 else union_columns

    print(f"\n{'='*80}")
    print(f"📊 요약 - {dataset_name}")
    print(f"{'='*80}")
    print(f"총 CSV 파일 수: {len(column_map)}")
    print(f"전체 Feature (합집합) 수: {len(union_columns)}")
    print(f"모든 파일에 공통으로 존재하는 Feature 수: {len(intersection_columns)}")

    if len(column_map) > 1:
        print("\n파일별 Feature 수 요약:")
        for file_name, columns in column_map.items():
            extra = set(columns) - set(intersection_columns)
            missing = set(intersection_columns) - set(columns)
            print(f"  - {file_name}: {len(columns)}개 (추가 {len(extra)}개, 누락 {len(missing)}개)")

    return union_columns

# CIC-IDS 2017 features
cic2017_features = show_features(
    './cic2017',
    'CIC-IDS 2017'
)

# CIC-IDS 2018 features
cic2018_features = show_features(
    './cic2018',
    'CIC-IDS 2018'
)

# UNSW-NB15 features
unswnb15_features = show_features(
    './unswnb15',
    'UNSW-NB15'
)

# 공통 및 차이점 분석
if cic2017_features and cic2018_features and unswnb15_features:
    print(f"\n{'='*80}")
    print("🔍 데이터셋 비교")
    print(f"{'='*80}")
    
    set_2017 = set(cic2017_features)
    set_2018 = set(cic2018_features)
    set_unsw = set(unswnb15_features)
    
    # 전체 공통 Feature
    common_all = set_2017 & set_2018 & set_unsw
    print(f"\n모든 데이터셋 공통 Feature: {len(common_all)}개")
    if common_all:
        print("  공통 Features:", sorted(common_all)[:10], "..." if len(common_all) > 10 else "")
    
    # CIC-IDS 간 비교
    print(f"\nCIC-IDS 2017 vs 2018:")
    common_cic = set_2017 & set_2018
    only_2017 = set_2017 - set_2018
    only_2018 = set_2018 - set_2017
    
    print(f"  공통 Feature: {len(common_cic)}개")
    print(f"  2017에만 있는 Feature: {len(only_2017)}개")
    print(f"  2018에만 있는 Feature: {len(only_2018)}개")
    
    if only_2017:
        print(f"  [2017 전용]: {sorted(only_2017)}")
    
    if only_2018:
        print(f"  [2018 전용]: {sorted(only_2018)}")
    
    # UNSW-NB15 고유 특징
    print(f"\nUNSW-NB15:")
    only_unsw = set_unsw - set_2017 - set_2018
    print(f"  UNSW에만 있는 Feature: {len(only_unsw)}개")
    if only_unsw:
        print(f"  [UNSW 전용]: {sorted(only_unsw)}")

print(f"\n{'='*80}")
print("✅ Feature 분석 완료!")
print(f"{'='*80}")

