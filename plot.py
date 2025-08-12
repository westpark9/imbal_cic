import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import argparse

def plot_per_class_metrics(csv_path, save_dir):
    """
    CSV 파일에서 클래스별 성능 지표를 읽어와 시각화합니다.
    
    Args:
        csv_path: 성능 지표가 저장된 CSV 파일 경로
        save_dir: 결과를 저장할 디렉토리
    """
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    
    # 클래스 이름과 메트릭스 추출
    class_names = df['Class'].values
    x = np.arange(len(class_names))
    
    # Accuracy 시각화
    plt.figure(figsize=(15, 6))
    plt.plot(x, df['Baseline_Accuracy'], 'b-o', label='Baseline', linewidth=2)
    plt.plot(x, df['Ensemble_Accuracy'], 'r-o', label='Ensemble', linewidth=2)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Class-wise Accuracy')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Accuracy 그래프 저장
    acc_save_path = os.path.join(save_dir, 'class_wise_accuracy.png')
    plt.savefig(acc_save_path, bbox_inches='tight')
    plt.close()
    
    # F1 Score 시각화
    plt.figure(figsize=(15, 6))
    plt.plot(x, df['Baseline_F1'], 'b-o', label='Baseline', linewidth=2)
    plt.plot(x, df['Ensemble_F1'], 'r-o', label='Ensemble', linewidth=2)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('Class-wise F1 Score')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # F1 Score 그래프 저장
    f1_save_path = os.path.join(save_dir, 'class_wise_f1.png')
    plt.savefig(f1_save_path, bbox_inches='tight')
    plt.close()
    
    print(f"그래프가 저장되었습니다:")
    print(f"- Accuracy: {acc_save_path}")
    print(f"- F1 Score: {f1_save_path}")

def main():
    parser = argparse.ArgumentParser(description='클래스별 성능 지표 시각화')
    parser.add_argument('--csv_path', type=str, required=True, help='성능 지표 CSV 파일 경로')
    parser.add_argument('--save_dir', type=str, required=True, help='결과를 저장할 디렉토리')
    
    args = parser.parse_args()
    
    # 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 시각화 실행
    plot_per_class_metrics(args.csv_path, args.save_dir)

if __name__ == '__main__':
    main()
