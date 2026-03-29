import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

_KAGGLE = os.environ.get('KAGGLE_MODE') == '1'

if _KAGGLE:
    _KAGGLE_INPUT  = Path('/kaggle/input/datasets/ehdgus4173')
    DATASET_DIR    = _KAGGLE_INPUT / 'dacon-structure-data' / 'dataset'
    FEATURES_DIR   = Path('/kaggle/working/features')
    CHECKPOINT_DIR = Path('/kaggle/working/checkpoints')
    OUTPUT_DIR     = Path('/kaggle/working')
else:
    _env_dataset = os.environ.get('DATASET_DIR')
    DATASET_DIR = Path(_env_dataset) if _env_dataset else PROJECT_ROOT.parent / 'structure-stability' / 'data'
    FEATURES_DIR   = PROJECT_ROOT / 'features'
    CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
    OUTPUT_DIR     = PROJECT_ROOT

FEATURES_DIR.mkdir(exist_ok=True, parents=True)
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_DIR = DATASET_DIR / 'train'
DEV_DIR   = DATASET_DIR / 'dev'
TEST_DIR  = DATASET_DIR / 'test'
TRAIN_CSV = DATASET_DIR / 'train.csv'
DEV_CSV   = DATASET_DIR / 'dev.csv'

# ================================================================
# 모델 입력 피처 상수 (PHYS_COLS_V2) — 교수 자문 반영 최종본
# ================================================================
# 변경 이력:
#   v1: 기존 front-view 기반 FS/kern (찬호 원본)
#   v2: dh 브랜치 — top-view 기준 통일 (이기호 방법 A)
#   v3: 교수 자문 (2026-03-29) — height_support_risk 부호 수정,
#       t_compactness_sq→elongation_index, stability_number/moment_ratio/pd_risk_index 추가

PHYS_COLS_V2 = [
    # 1세대: front 뷰 (extract_base.py)
    'f_cx_offset',            # 수평 무게중심 편차 — SHAP 1위
    'f_cy_ratio',             # 수직 무게중심 위치 (0=상단/높음, 1=하단/낮음)

    # 1세대: top 뷰 (extract_base.py)
    't_compactness',          # 평면 컴팩트니스
    't_cx_offset',            # 좌우 무게중심 편차 (raw)
    't_cy_offset',            # 앞뒤 무게중심 편차
    't_left_mass_ratio',      # 좌우 질량 불균형
    't_frontback_mass_ratio', # 앞뒤 질량 불균형

    # 2세대: 구조공학 수식 (extract_advanced.py) — dh 브랜치 + 교수 자문
    'FS_top',              # (B_top/2) / t_cx_offset — 전도 안전율
    'kern_top',            # t_cx_offset / (B_top/6) — kern ratio
    't_ecc_2d',            # sqrt(t_cx² + t_cy²) — 2축 합성 편심
    'mass_asymmetry_2d',   # sqrt(left² + frontback²) — 2축 질량 비대칭
    'support_margin_min',  # (B_top/2) - t_ecc_2d — 지지 여유
    'height_support_risk', # (1-f_cy_ratio) / |margin| — 복합 위험도 [부호 수정]
    'compact_ecc',         # t_compactness / ecc_front_2d — 형상/편심 복합
    'elongation_index',    # 1 - t_compactness — 세장 형상 지수 [t_compactness_sq 대체]

    # 교수 자문 추가 피처 (2026-03-29)
    'stability_number',    # (B_top/2) / (1-f_cy_ratio) — 실무 안정수 SN
    'moment_ratio',        # (B_top/2 - t_ecc_2d) / t_ecc_2d — 전도 모멘트 비율
    'pd_risk_index',       # t_ecc_2d × (1-f_cy_ratio) / B_top — P-delta 근사

    # 격자 기반 카메라 시점 (extract_base.py, Hough 기반)
    'front_grid_detected',
    'front_grid_tilt_angle',
    'front_grid_perspective_ratio',
    'top_grid_detected',
    'top_grid_tilt_angle',
    'top_grid_perspective_ratio',
]
