"""
Prority 2: 픽셀 기반 피처 추출 및 병합 (`extract_pixel_features.py`)
- front, top 이미지 픽셀 정보(밝기/대비, 엣지 밀도, HSV 분포) 추출
- 추출한 데이터를 물리 피처(`physics_features.csv`, `video_motion_features.csv`)와 결합
- 최종 `combined_features_v2.csv` 파일 출력
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import TRAIN_CSV, DEV_CSV, PROJECT_ROOT, DATASET_DIR

def extract_image_features(img_path: Path):
    """
    단일 이미지(png)에 대한 픽셀 레벨 피처 추출
    """
    if not img_path.exists():
        return None
        
    img = cv2.imread(str(img_path))
    if img is None:
        return None
        
    # BGR -> RGB & HSV 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 밝기 (Brightness) & 대비 (Contrast - RMS)
    mean_brightness = np.mean(img_gray)
    rms_contrast = np.std(img_gray)
    
    # 2. 엣지 밀도 (Edge density via Canny)
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.sum(edges > 0) / (img_gray.shape[0] * img_gray.shape[1])
    
    # 3. HSV 분포 (Hue, Saturation, Value)
    h_mean = np.mean(img_hsv[:, :, 0])
    s_mean = np.mean(img_hsv[:, :, 1])
    v_mean = np.mean(img_hsv[:, :, 2])
    
    return {
        "brightness": mean_brightness,
        "contrast": rms_contrast,
        "edge_density": edge_density,
        "hsv_h_mean": h_mean,
        "hsv_s_mean": s_mean,
        "hsv_v_mean": v_mean
    }

def main():
    # 1. 모든 CSV 목록 로드 (train, dev, test)
    # TEST_CSV가 dataset에 없으므로 (sample_submission.csv 기준)
    test_csv_path = PROJECT_ROOT.parent / "건축물진단" / "codebase" / "guideline" / "sample_submission.csv"
    
    dfs = {
        "train": pd.read_csv(TRAIN_CSV),
        "dev": pd.read_csv(DEV_CSV)
    }
    if test_csv_path.exists():
        dfs["test"] = pd.read_csv(test_csv_path)

    all_pixel_features = []
    
    # 2. 모든 데이터 순회하며 피처 추출
    print("Extracting pixel-level features...")
    for split_name, df in dfs.items():
        base_dir = DATASET_DIR / split_name
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            sample_id = str(row['id'])
            sample_dir = base_dir / sample_id
            
            front_feats = extract_image_features(sample_dir / "front.png")
            top_feats = extract_image_features(sample_dir / "top.png")
            
            # 피처 병합
            sample_dict = {"id": sample_id}
            
            if front_feats:
                for k, v in front_feats.items():
                    sample_dict[f"front_{k}"] = v
            else:
                for k in ["brightness", "contrast", "edge_density", "hsv_h_mean", "hsv_s_mean", "hsv_v_mean"]:
                    sample_dict[f"front_{k}"] = 0.0
                    
            if top_feats:
                for k, v in top_feats.items():
                    sample_dict[f"top_{k}"] = v
            else:
                for k in ["brightness", "contrast", "edge_density", "hsv_h_mean", "hsv_s_mean", "hsv_v_mean"]:
                    sample_dict[f"top_{k}"] = 0.0
                    
            all_pixel_features.append(sample_dict)
            
    pixel_df = pd.DataFrame(all_pixel_features)
    print(f"Extraction complete! Row count: {len(pixel_df)}")
    
    # 3. 기존 분석본과 피처 조인 (combined_features_v2)
    # 찬호님 물리 피처 경로 지정
    old_base = PROJECT_ROOT.parent / "건축물진단" / "codebase" / "0324_제공데이터분석"
    phys_csv = old_base / "03_physical_feature" / "outputs" / "physics_features.csv"
    video_csv = old_base / "04_mp4_ana" / "outputs" / "video_motion_features.csv"
    
    combined_df = pixel_df.copy()
    
    if phys_csv.exists():
        # 원본 라벨이 포함된 경우 드롭
        p_df = pd.read_csv(phys_csv).drop(columns=['label', 'label_bin'], errors='ignore')
        combined_df = combined_df.merge(p_df, on='id', how='left')
        print(f"Merged physical features. Columns: {len(combined_df.columns)}")
        
    if video_csv.exists():
        v_df = pd.read_csv(video_csv).drop(columns=['label', 'label_bin'], errors='ignore')
        combined_df = combined_df.merge(v_df, on='id', how='left')
        print(f"Merged video motion features. Columns: {len(combined_df.columns)}")
        
    # 결과 저장
    output_dir = PROJECT_ROOT / "features"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "combined_features_v2.csv"
    
    combined_df.fillna(0, inplace=True) # 누락 피처(Test셋에 모션 결측 등) 처리
    combined_df.to_csv(out_path, index=False)
    
    print(f"✅ Final combined feature shape: {combined_df.shape}")
    print(f"✅ Saved to: {out_path}")

if __name__ == "__main__":
    main()
