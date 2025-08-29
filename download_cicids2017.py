#!/usr/bin/env python3
"""
CICIDS-2017 데이터셋 자동 다운로드 스크립트
"""

import os
import requests
import zipfile
from tqdm import tqdm
import hashlib

def download_file(url, filename, expected_md5=None):
    """파일을 다운로드하고 MD5 체크섬을 확인합니다."""
    print(f"다운로드 중: {filename}")
    
    # 파일이 이미 존재하는지 확인
    if os.path.exists(filename):
        print(f"파일이 이미 존재합니다: {filename}")
        return True
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # MD5 체크섬 확인
        if expected_md5:
            with open(filename, 'rb') as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
                if file_md5 != expected_md5:
                    print(f"MD5 체크섬 불일치: {filename}")
                    os.remove(filename)
                    return False
                else:
                    print(f"MD5 체크섬 확인 완료: {filename}")
        
        return True
        
    except Exception as e:
        print(f"다운로드 실패: {filename} - {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

def extract_zip(zip_path, extract_to):
    """ZIP 파일을 압축 해제합니다."""
    print(f"압축 해제 중: {zip_path}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"압축 해제 완료: {extract_to}")
        return True
    except Exception as e:
        print(f"압축 해제 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("CICIDS-2017 데이터셋 다운로드 시작")
    print("=" * 50)
    
    # 데이터 디렉토리 생성
    data_dir = "MachineLearningCVE"
    os.makedirs(data_dir, exist_ok=True)
    
    # 다운로드할 파일 목록 (실제 URL은 변경될 수 있음)
    files_to_download = [
        {
            "url": "https://www.unb.ca/cic/datasets/ids-2017.html",
            "filename": "cicids2017_info.html",
            "description": "데이터셋 정보 페이지"
        }
    ]
    
    print("⚠️  주의: CICIDS-2017 데이터셋은 공식적으로 공개되지 않았습니다.")
    print("다음 방법 중 하나를 선택하세요:")
    print("1. https://www.unb.ca/cic/datasets/ids-2017.html 에서 직접 다운로드")
    print("2. Kaggle에서 CICIDS-2017 데이터셋 검색")
    print("3. 연구 목적으로 사용 가능한 대안 데이터셋 사용")
    
    print("\n" + "=" * 50)
    print("대안 데이터셋 다운로드:")
    
    # 대안 데이터셋들
    alternative_datasets = [
        {
            "name": "UNSW-NB15",
            "url": "https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/",
            "description": "UNSW-NB15 네트워크 침입 탐지 데이터셋"
        },
        {
            "name": "NSL-KDD",
            "url": "https://github.com/defcom17/NSL_KDD",
            "description": "NSL-KDD 데이터셋 (KDD Cup 1999 개선 버전)"
        }
    ]
    
    for dataset in alternative_datasets:
        print(f"📊 {dataset['name']}: {dataset['description']}")
        print(f"   URL: {dataset['url']}")
        print()
    
    print("=" * 50)
    print("권장사항:")
    print("1. 연구 목적이라면 UNSW-NB15나 NSL-KDD 사용")
    print("2. CICIDS-2017이 꼭 필요하다면 공식 웹사이트에서 수동 다운로드")
    print("3. 데이터 파일을 MachineLearningCVE/ 폴더에 넣고 실행")

if __name__ == "__main__":
    main()
