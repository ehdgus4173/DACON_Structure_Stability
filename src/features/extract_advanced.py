"""
extract_physics_v2.py
찬호의 구조공학 2세대 피처(FS_overturning, kern_ratio 등 9종)를
combined_features_v2.csv에 병합하여 combined_features_v3.csv 생성.

출처: 찬호 검증 스크립트 (add_physics_features 함수 기반)
실행: python extract_physics_v2.py
출력: DACON_Structure_Stability/features/combined_features_v3.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import PROJECT_ROOT, TRAIN_CSV, DEV_CSV, SAMPLE_SUBMISSION_CSV


# ================================================================
# 2세대 구조공학 피처 추가 함수 (찬호 코드 그대로 이식)
# ================================================================
def add_physics_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    기존 피처(physics_features.csv 기반)에 구조공학 이론 피처 9종 추가.

    입력 컬럼 필요:
        t_footprint_area, f_cx_offset, f_cy_offset,
        f_mass_upper_ratio, f_cy_norm, f_tilt_angle,
        f_height_ratio, f_width_ratio, t_left_mass_ratio

    추가되는 컬럼:
        FS_overturning, kern_ratio, top_heavy_index,
        p_delta_eccentricity, effective_eccentricity,
        eccentric_combined, base_stability_index,
        lateral_moment_approx, slenderness_approx
    """
    d = data.copy()

    # 기초 폭 추정 (footprint_area의 제곱근으로 근사)
    B = np.sqrt(d['t_footprint_area'].clip(lower=1e-6))

    # 피처 1. 전도 안전율 (FS < 1.5 = Danger)
    d['FS_overturning'] = (B / 2) / (d['f_cx_offset'].abs() + 1e-6)

    # 피처 2. Kern Ratio (> 1.0 = Unstable)
    d['kern_ratio'] = d['f_cx_offset'].abs() / (B / 6 + 1e-6)

    # 피처 3. Top-Heavy Index (상부질량 × 높이 비율)
    d['top_heavy_index'] = d['f_mass_upper_ratio'] * d['f_cy_norm']

    # 피처 4. P-delta 편심
    d['p_delta_eccentricity'] = (
        np.sin(np.radians(d['f_tilt_angle'])) * d['f_cy_norm']
    )

    # 피처 5. 유효 편심 (P-delta 포함)
    d['effective_eccentricity'] = (
        d['f_cx_offset'].abs() + d['p_delta_eccentricity'].abs()
    )

    # 피처 6. 복합 편심 (x, y 방향 편차 합성)
    d['eccentric_combined'] = np.sqrt(
        d['f_cx_offset']**2 + d['f_cy_offset']**2
    )

    # 피처 7. 기저 안정 지수
    d['base_stability_index'] = (
        d['t_footprint_area'] / (d['eccentric_combined'] + 1e-6)
    )

    # 피처 8. 횡방향 모멘트 근사
    d['lateral_moment_approx'] = d['t_left_mass_ratio'] * d['f_cy_norm']

    # 피처 9. 세장비 근사
    d['slenderness_approx'] = d['f_height_ratio'] / (d['f_width_ratio'] + 1e-6)

    return d


ENGINEERED_COLS = [
    'FS_overturning',
    'kern_ratio',
    'top_heavy_index',
    'p_delta_eccentricity',
    'effective_eccentricity',
    'eccentric_combined',
    'base_stability_index',
    'lateral_moment_approx',
    'slenderness_approx',
]

# 도메인 시프트 Safe 피처 (찬호 검증 기준)
SAFE_ENGINEERED = [
    'FS_overturning',       # Feature Importance 2위, Safe
    'kern_ratio',           # Feature Importance 3위, Safe
    'effective_eccentricity',  # Feature Importance 5위
    'eccentric_combined',   # Safe
    'p_delta_eccentricity', # Safe
]

# Phase 2 최종 사용 피처 세트 (CNN Fusion용)
PHYSICS_FEATURE_COLS_V2 = [
    # 1세대 Safe 피처
    't_compactness',
    'f_cx_offset',
    't_left_mass_ratio',
    't_cx_offset',
    'f_mass_upper_ratio',
    # 2세대 Safe + 고중요도
    'FS_overturning',
    'kern_ratio',
    'effective_eccentricity',
    'eccentric_combined',
    'p_delta_eccentricity',
]


def main(args=None):
    # ----------------------------------------------------------------
    # 1. combined_features_v2.csv 로드
    # ----------------------------------------------------------------
    v2_path = PROJECT_ROOT / 'features' / 'combined_features_v2.csv'
    if not v2_path.exists():
        raise FileNotFoundError(
            f"combined_features_v2.csv 없음: {v2_path}\n"
            "먼저 extract_pixel_features.py 실행 필요"
        )

    df = pd.read_csv(v2_path)
    print(f"[1] combined_features_v2.csv 로드: {df.shape[0]}행, {df.shape[1]}컬럼")

    # ----------------------------------------------------------------
    # 2. 필요 컬럼 존재 확인
    # ----------------------------------------------------------------
    required = [
        't_footprint_area', 'f_cx_offset', 'f_cy_offset',
        'f_mass_upper_ratio', 'f_cy_norm', 'f_tilt_angle',
        'f_height_ratio', 'f_width_ratio', 't_left_mass_ratio'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필요 컬럼 누락: {missing}")
    print(f"[2] 필요 컬럼 확인 완료")

    # ----------------------------------------------------------------
    # 3. 2세대 피처 계산 및 병합
    # ----------------------------------------------------------------
    df = add_physics_features(df)
    print(f"[3] 2세대 피처 {len(ENGINEERED_COLS)}종 추가 → 총 {df.shape[1]}컬럼")
    for col in ENGINEERED_COLS:
        null_cnt = df[col].isnull().sum()
        print(f"    {col:<30} null={null_cnt}")

    # ----------------------------------------------------------------
    # 4. 저장
    # ----------------------------------------------------------------
    output_dir = PROJECT_ROOT / 'features'
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / 'combined_features_v3.csv'
    df.to_csv(out_path, index=False)

    print(f"\n[4] 저장 완료: {out_path}")
    print(f"    최종 shape: {df.shape}")
    print(f"\n[Phase 2 Fusion용 피처 세트 ({len(PHYSICS_FEATURE_COLS_V2)}종)]")
    for col in PHYSICS_FEATURE_COLS_V2:
        status = '✅ Safe' if col in SAFE_ENGINEERED or col in [
            't_compactness', 'f_cx_offset', 't_left_mass_ratio',
            't_cx_offset', 'f_mass_upper_ratio'
        ] else '⚠️  확인필요'
        print(f"    {col:<30} {status}")

    print("\n✅ combined_features_v3.csv 생성 완료. train.py에서 PHYS_COLS_V2로 교체하세요.")


if __name__ == '__main__':
    main()
