import os
from pathlib import Path

# ================================================================
# 경로 상수
# ================================================================

# 현재 파일(config.py)이 위치한 디렉토리 (DACON_Structure_Stability/)
PROJECT_ROOT = Path(__file__).resolve().parent

# 원본 데이터 (DACON/dataset/)
DATASET_DIR = PROJECT_ROOT.parent / "dataset"
TRAIN_DIR   = DATASET_DIR / "train"
DEV_DIR     = DATASET_DIR / "dev"
TEST_DIR    = DATASET_DIR / "test"

# CSV 경로
TRAIN_CSV = DATASET_DIR / "train.csv"
DEV_CSV   = DATASET_DIR / "dev.csv"

# 제출 샘플 파일 경로
GUIDELINE_DIR         = PROJECT_ROOT.parent / "EDA" / "codebase" / "example"
SAMPLE_SUBMISSION_CSV = GUIDELINE_DIR / "sample_submission.csv"

# 모델 체크포인트 저장 경로
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

# ================================================================
# 모델 입력 피처 상수
# ================================================================

# CNN Fusion 모델 입력 물리 피처 10종
# ⚠️ 이 목록을 변경하면 combined_features_v3.csv 재생성 + 재학습 필요
# 참조 파일: src/features/extract_advanced.py, src/pipeline/train.py, src/pipeline/inference.py
PHYS_COLS_V2 = [
    # 1세대 Safe 피처 (extract_base.py 이미지 직접 추출)
    't_compactness',
    'f_cx_offset',
    't_left_mass_ratio',
    't_cx_offset',
    'f_mass_upper_ratio',
    # 2세대 Safe + 고중요도 (extract_advanced.py 수식 계산)
    'FS_overturning',
    'kern_ratio',
    'effective_eccentricity',
    'eccentric_combined',
    'p_delta_eccentricity',
]
