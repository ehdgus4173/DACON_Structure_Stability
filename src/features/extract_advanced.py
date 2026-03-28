import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import PROJECT_ROOT, PHYS_COLS_V2


# ================================================================
# 2세대 피처 메타 정보  (찬호 feature_engineering_v3.py 기반)
# ================================================================

ENGINEERED_COLS = [
    # FS / kern — top-view 기준 통일 (이기호 방법 A)
    'FS_top',              # (B_top/2) / t_cx_offset  ← B, e 모두 top
    'kern_top',            # t_cx_offset / (B_top/6)  ← 동일 좌표계

    # 2축 편심 / 질량 비대칭
    't_ecc_2d',            # sqrt(t_cx_offset² + t_cy_offset²)
    'mass_asymmetry_2d',   # sqrt(t_left_mass_ratio² + t_frontback_mass_ratio²)

    # 지지 여유 파생
    'support_margin_min',  # (B_top/2) - t_ecc_2d  — 양수=안전, 음수=전도 위험
    'height_support_risk', # f_cy_ratio / (|support_margin_min| + ε)  — 복합 위험도

    # 형상 복합
    'compact_ecc',         # t_compactness / (ecc_front_2d + ε)
    't_compactness_sq',    # t_compactness²
]


# ================================================================
# 2세대 구조공학 피처 계산  (찬호 build_v3 로직 이식)
# ================================================================
def add_physics_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    extract_base.py에서 추출한 1세대 피처를 입력받아 구조공학 피처 추가.

    입력 컬럼 필요:
        t_footprint_area, t_cx_offset, t_cy_offset,
        t_left_mass_ratio, t_frontback_mass_ratio,
        f_cx_offset, f_cy_offset, f_cy_ratio

    추가되는 컬럼 (ENGINEERED_COLS):
        FS_top, kern_top,
        t_ecc_2d, mass_asymmetry_2d,
        support_margin_min, height_support_risk,
        compact_ecc, t_compactness_sq
    """
    d = data.copy()

    # 기초폭 — top-view 기준
    B_top = np.sqrt(d['t_footprint_area'].clip(lower=1e-6))

    # ── FS / kern  (이기호 방법 A: B, e 모두 top-view) ──────────────
    d['FS_top']   = (B_top / 2) / (d['t_cx_offset'].abs() + 1e-6)
    d['kern_top'] = d['t_cx_offset'].abs() / (B_top / 6 + 1e-6)

    # ── 2축 편심 ────────────────────────────────────────────────────
    d['t_ecc_2d'] = np.sqrt(
        d['t_cx_offset'] ** 2 + d['t_cy_offset'] ** 2
    )

    # ── 2축 질량 비대칭 ─────────────────────────────────────────────
    # t_frontback_mass_ratio 없으면 t_cy_offset abs로 근사 (찬호 fallback 동일)
    if 't_frontback_mass_ratio' in d.columns:
        frontback = d['t_frontback_mass_ratio']
    else:
        frontback = d['t_cy_offset'].abs()
    d['mass_asymmetry_2d'] = np.sqrt(
        d['t_left_mass_ratio'] ** 2 + frontback ** 2
    )

    # ── 지지 여유 파생 ───────────────────────────────────────────────
    d['support_margin_min'] = (B_top / 2) - d['t_ecc_2d']
    d['height_support_risk'] = (
        d['f_cy_ratio'] / (d['support_margin_min'].abs() + 1e-6)
    )

    # ── 형상 복합 ────────────────────────────────────────────────────
    ecc_front_2d = np.sqrt(d['f_cx_offset'] ** 2 + d['f_cy_offset'] ** 2)
    d['compact_ecc']      = d['t_compactness'] / (ecc_front_2d + 1e-6)
    d['t_compactness_sq'] = d['t_compactness'] ** 2

    return d


def main(args=None):
    # ----------------------------------------------------------------
    # 1. combined_features_v2.csv 로드
    # ----------------------------------------------------------------
    v2_path = PROJECT_ROOT / 'features' / 'combined_features_v2.csv'
    if not v2_path.exists():
        raise FileNotFoundError(
            f"combined_features_v2.csv 없음: {v2_path}\n"
            "먼저 extract_base.py (Step 1) 실행 필요"
        )

    df = pd.read_csv(v2_path)
    print(f"[1] combined_features_v2.csv 로드: {df.shape[0]}행, {df.shape[1]}컬럼")

    # ----------------------------------------------------------------
    # 2. 필요 컬럼 존재 확인
    # ----------------------------------------------------------------
    required = [
        't_footprint_area', 't_cx_offset', 't_cy_offset',
        't_left_mass_ratio', 't_compactness',
        'f_cx_offset', 'f_cy_offset', 'f_cy_ratio',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필요 컬럼 누락: {missing}\nextract_base.py 재실행 필요")
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
    print(f"\n[모델 입력 피처 ({len(PHYS_COLS_V2)}종, PHYS_COLS_V2 from config.py)]")
    for col in PHYS_COLS_V2:
        tag = '(2세대)' if col in ENGINEERED_COLS else '(1세대)'
        print(f"    {col:<30} {tag}")

    print("\n✅ combined_features_v3.csv 생성 완료.")


if __name__ == '__main__':
    main()
