import os
from pathlib import Path

# ================================================================
# 경로 상수
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# ── 환경 분기 ────────────────────────────────────────────────
_KAGGLE = os.environ.get('KAGGLE_MODE') == '1'

if _KAGGLE:
    _KAGGLE_INPUT  = Path('/kaggle/input/datasets/ehdgus4173')
    DATASET_DIR    = _KAGGLE_INPUT / 'dacon-structure-data' / 'dataset'
    FEATURES_DIR   = Path('/kaggle/working/features')
    CHECKPOINT_DIR = Path('/kaggle/working/checkpoints')
    OUTPUT_DIR     = Path('/kaggle/working')
else:
    DATASET_DIR    = PROJECT_ROOT.parent / 'dataset'
    FEATURES_DIR   = PROJECT_ROOT / 'features'
    CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
    OUTPUT_DIR     = PROJECT_ROOT

FEATURES_DIR.mkdir(exist_ok=True, parents=True)
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

# ── 데이터 경로 ──────────────────────────────────────────────
TRAIN_DIR = DATASET_DIR / 'train'
DEV_DIR   = DATASET_DIR / 'dev'
TEST_DIR  = DATASET_DIR / 'test'
TRAIN_CSV = DATASET_DIR / 'train.csv'
DEV_CSV   = DATASET_DIR / 'dev.csv'

if _KAGGLE:
    SAMPLE_SUBMISSION_CSV = DATASET_DIR / 'sample_submission.csv'
else:
    GUIDELINE_DIR         = PROJECT_ROOT.parent / 'EDA' / 'codebase' / 'example'
    SAMPLE_SUBMISSION_CSV = GUIDELINE_DIR / 'sample_submission.csv'

# ================================================================
# 모델 입력 피처 상수 (PHYS_COLS_V2)
# ================================================================
# ⚠️ 변경 시 combined_features_v3.csv 재생성 + 재학습 필요
# 참조: src/features/extract_base.py, extract_advanced.py, pipeline/train.py, pipeline/inference.py

PHYS_COLS_V2 = [
    # 1세대: front 뷰 (extract_base.py)
    'f_cx_offset',            # 수평 무게중심 편차
    'f_cy_ratio',             # 수직 무게중심 위치 (0=상단, 1=하단) ← 신규
    'f_mass_upper_ratio',     # 상단 질량 비율

    # 1세대: top 뷰 (extract_base.py)
    't_compactness',          # 평면 컴팩트니스
    't_cx_offset',            # 좌우 무게중심 편차 (raw)
    't_left_mass_ratio',      # 좌우 질량 불균형
    't_frontback_mass_ratio', # 앞뒤 질량 불균형 ← 신규
    't_pa_cx_offset',         # Principal Axis 보정 후 좌우 편심 ← 신규
    't_pa_cy_offset',         # Principal Axis 보정 후 앞뒤 편심 ← 신규

    # 2세대: 구조공학 수식 (extract_advanced.py)
    'FS_overturning',
    'kern_ratio',
    'effective_eccentricity',
    'eccentric_combined',
    'p_delta_eccentricity',

    # 격자 기반 카메라 시점 (extract_base.py, Hough 기반)
    'front_grid_detected',
    'front_grid_tilt_angle',
    'front_grid_perspective_ratio',
    'top_grid_detected',
    'top_grid_tilt_angle',
    'top_grid_perspective_ratio',
]
