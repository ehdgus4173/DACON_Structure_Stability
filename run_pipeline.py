import argparse
import sys
import pandas as pd
from pathlib import Path

# Add project root to sys.path if needed
sys.path.append(str(Path(__file__).resolve().parent))
from config import PROJECT_ROOT

from src.features.extract_base import main as extract_pixel_main
from src.features.extract_advanced import main as extract_physics_main
from src.pipeline.train import main as train_main
from src.pipeline.inference import main as inference_main

def validate_features(feature_csv_path):
    """피처 CSV의 test 샘플 정상 여부 확인"""
    df = pd.read_csv(feature_csv_path)
    test_df = df[df['id'].str.startswith('TEST_')]

    issues = []
    if len(test_df) > 0:
        if (test_df['f_cx_offset'] == 0).all():
            issues.append("f_cx_offset이 전부 0 → 물리 피처 미추출 상태")
        if (test_df['FS_overturning'] > 100).mean() > 0.5:
            issues.append("FS_overturning 이상값 다수 → 입력 피처 오류")

    if issues:
        for msg in issues:
            print(f"[❌ 피처 검증 실패] {msg}")
        raise ValueError("피처 검증 실패. extract_pixel_features.py 수정 필요.")
    else:
        print("[✅ 피처 검증 통과] test 샘플 물리 피처 정상 확인")

def main():
    parser = argparse.ArgumentParser(description="DACON Structure Stability Pipeline")
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'all'], default='all')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--skip_extract', action='store_true', help="피처 CSV가 이미 존재하면 추출 단계 스킵")
    parser.add_argument('--output', type=str, default='submission.csv')
    
    args = parser.parse_args()
    
    feature_v3_path = PROJECT_ROOT / "features" / "combined_features_v3.csv"
    
    # 1. & 2. Feature Extraction
    if args.skip_extract and feature_v3_path.exists():
        print("Skipping feature extraction as requested (--skip_extract).")
    else:
        if args.mode in ['train', 'all']:
            print("=== Step 1: Pixel & Base Physics Feature Extraction ===")
            extract_pixel_main([])
            
            print("\n=== Step 2: Advanced Physics Feature Generation ===")
            extract_physics_main([])
    
    # 3. Feature Validation
    if feature_v3_path.exists():
        print("\n=== Step 3: Feature Validation ===")
        validate_features(feature_v3_path)
    elif not args.skip_extract and args.mode in ['train', 'all']:
        raise FileNotFoundError(f"Feature file not found: {feature_v3_path}")
        
    # 4. Train
    if args.mode in ['train', 'all']:
        print("\n=== Step 4: Training ===")
        train_args = ['--epochs', str(args.epochs), '--batch_size', str(args.batch_size), '--lr', str(args.lr)]
        train_main(train_args)
        
    # 5. Inference
    if args.mode in ['inference', 'all']:
        print("\n=== Step 5: Inference ===")
        infer_args = ['--batch_size', str(args.batch_size), '--output', args.output]
        inference_main(infer_args)

if __name__ == "__main__":
    main()
