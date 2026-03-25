import os
from pathlib import Path

# 현재 파일(config.py)이 위치한 디렉토리 (DACON_Structure_Stability/)
PROJECT_ROOT = Path(__file__).resolve().parent

# 원본 데이터가 위치한 디렉토리 연결
# 루트(DACON/) 바로 아래 dataset/ 폴더 참조
DATASET_DIR = PROJECT_ROOT.parent / "dataset"
TRAIN_DIR   = DATASET_DIR / "train"
DEV_DIR     = DATASET_DIR / "dev"
TEST_DIR    = DATASET_DIR / "test"

# CSV 경로
TRAIN_CSV = DATASET_DIR / "train.csv"
DEV_CSV   = DATASET_DIR / "dev.csv"

# 제공된 guideline의 submission 샘플 파일 경로
GUIDELINE_DIR = PROJECT_ROOT.parent / "EDA" / "codebase" / "example"
SAMPLE_SUBMISSION_CSV = GUIDELINE_DIR / "sample_submission.csv"

# 모델 등 결과물을 저장할 경로 설정
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
