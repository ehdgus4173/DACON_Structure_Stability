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
    # 로컬: 실제 데이터 위치 (structure-stability/data/)
    # DATASET_DIR 환경변수로 오버라이드 가능 (팀원 경로 다를 때)
    _env_dataset = os.environ.get('DATASET_DIR')
    DATASET_DIR = Path(_env_dataset) if _env_dataset else PROJECT_ROOT.parent / 'structure-stability' / 'data'
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

# ================================================================
# 모델 입력 피처 상수 (PHYS_COLS_V2) — dh 브랜치 기준
# ================================================================
# ⚠️ 변경 시 combined_features_v3.csv 재생성 + 재학습 필요
# dh 브랜치 (이기호 방법 A): top-view 기준 2세대 피처로 전면 교체

PHYS_COLS_V2 = [
    # 1세대: front 뷰 (extract_base.py)
    'f_cx_offset',            # 수평 무게중심 편차
    'f_cy_ratio',             # 수직 무게중심 위치 = f_cy_norm

    # 1세대: top 뷰 (extract_base.py)
    't_compactness',          # 평면 컴팩트니스
    't_cx_offset',            # 좌우 무게중심 편차
    't_cy_offset',            # 앞뒤 무게중심 편차  ← dh 신규
    't_left_mass_ratio',      # 좌우 질량 불균형
    't_frontback_mass_ratio', # 앞뒤 질량 불균형

    # 2세대: 구조공학 수식 (extract_advanced.py) — 찬호 v3 / 이기호 방법 A
    'FS_top',              # (B_top/2) / t_cx_offset  ← top-view 통일
    'kern_top',            # t_cx_offset / (B_top/6)  ← top-view 통일
    't_ecc_2d',            # sqrt(t_cx² + t_cy²) — 2축 편심
    'mass_asymmetry_2d',   # sqrt(t_left² + t_frontback²) — 2축 질량 비대칭
    'support_margin_min',  # (B_top/2) - t_ecc_2d — 양수=안전, 음수=전도 위험
    'height_support_risk', # f_cy_ratio / (|margin| + ε) — 복합 위험도
    'compact_ecc',         # t_compactness / (ecc_front_2d + ε)
    't_compactness_sq',    # t_compactness²

    # 격자 기반 카메라 시점 (extract_base.py, Hough 기반)
    'front_grid_detected',
    'front_grid_tilt_angle',
    'front_grid_perspective_ratio',
    'top_grid_detected',
    'top_grid_tilt_angle',
    'top_grid_perspective_ratio',
]
