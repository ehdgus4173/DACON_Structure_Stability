import sys
import numpy as np
import pandas as pd
from pathlib import Path
import os

_IS_KAGGLE = os.environ.get('KAGGLE_MODE') == '1' or os.path.exists('/kaggle')

if _IS_KAGGLE:
    _FEAT_DIR    = Path(os.environ.get('FEAT_OUT_DIR', '/kaggle/working/features'))
    PROJECT_ROOT = Path('/kaggle/working')
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    try:
        from config import PHYS_COLS_V2
    except ImportError:
        PHYS_COLS_V2 = []
else:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from config import PROJECT_ROOT, PHYS_COLS_V2
    _FEAT_DIR = PROJECT_ROOT / 'features'


# ================================================================
# 2세대 피처 메타 정보 — dh 브랜치 + 교수 자문 반영
# ================================================================

ENGINEERED_COLS = [
    # FS / kern — top-view 기준 통일 (이기호 방법 A)
    'FS_top',              # (B_top/2) / t_cx_offset  ← B, e 모두 top
    'kern_top',            # t_cx_offset / (B_top/6)  ← 동일 좌표계

    # 2축 편심 / 질량 비대칭
    't_ecc_2d',            # sqrt(t_cx_offset² + t_cy_offset²)
    'mass_asymmetry_2d',   # sqrt(t_left_mass_ratio² + t_frontback_mass_ratio²)

    # 지지 여유 파생
    'support_margin_min',  # (B_top/2) - t_ecc_2d
    'height_support_risk', # (1 - f_cy_ratio) / (|support_margin_min| + ε)  ← 부호 수정

    # 형상 복합
    'compact_ecc',         # t_compactness / (ecc_front_2d + ε)
    'elongation_index',    # 1 - t_compactness  ← t_compactness_sq 대체

    # 교수 자문 추가 피처
    'stability_number',    # (B_top/2) / ((1-f_cy_ratio) × H + ε)  — 실무 안정수
    'moment_ratio',        # (B_top/2 - t_ecc_2d) / (t_ecc_2d + ε)  — 전도 모멘트 비율
    'pd_risk_index',       # t_ecc_2d × (1-f_cy_ratio) / (B_top + ε)  — P-delta 근사
]


# ================================================================
# 2세대 구조공학 피처 계산
# ================================================================
def add_physics_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    extract_base.py에서 추출한 1세대 피처를 입력받아 구조공학 피처 추가.
    교수 자문 (2026-03-29) 반영:
      - height_support_risk 부호 수정: f_cy_ratio → (1 - f_cy_ratio)
      - t_compactness_sq → elongation_index (1 - t_compactness)
      - stability_number 추가
      - moment_ratio 추가
      - pd_risk_index 추가
    """
    d = data.copy()

    # 기초폭 — top-view 기준
    B_top = np.sqrt(d['t_footprint_area'].clip(lower=1e-6))

    # ── FS / kern (이기호 방법 A: B, e 모두 top-view) ────────────
    d['FS_top']   = (B_top / 2) / (d['t_cx_offset'].abs() + 1e-6)
    d['kern_top'] = d['t_cx_offset'].abs() / (B_top / 6 + 1e-6)

    # ── 2축 편심 ────────────────────────────────────────────────
    d['t_ecc_2d'] = np.sqrt(
        d['t_cx_offset'] ** 2 + d['t_cy_offset'] ** 2
    )

    # ── 2축 질량 비대칭 ─────────────────────────────────────────
    if 't_frontback_mass_ratio' in d.columns:
        frontback = d['t_frontback_mass_ratio']
    else:
        frontback = d['t_cy_offset'].abs()
    d['mass_asymmetry_2d'] = np.sqrt(
        d['t_left_mass_ratio'] ** 2 + frontback ** 2
    )

    # ── 지지 여유 파생 ───────────────────────────────────────────
    d['support_margin_min'] = (B_top / 2) - d['t_ecc_2d']

    # 부호 수정: (1 - f_cy_ratio) — 무게중심이 높을수록(0에 가까울수록) 위험
    h_ratio = 1.0 - d['f_cy_ratio']  # 0=낮은 위치, 1=높은 위치
    d['height_support_risk'] = h_ratio / (d['support_margin_min'].abs() + 1e-6)

    # ── 형상 복합 ────────────────────────────────────────────────
    ecc_front_2d = np.sqrt(d['f_cx_offset'] ** 2 + d['f_cy_offset'] ** 2)
    d['compact_ecc']       = d['t_compactness'] / (ecc_front_2d + 1e-6)
    d['elongation_index']  = 1.0 - d['t_compactness']  # t_compactness_sq 대체

    # ── 교수 자문 추가 피처 ──────────────────────────────────────

    # Stability Number: SN = (B_top/2) / H_CoM
    # H_CoM = (1-f_cy_ratio) × 정규화 높이 (f_cy_ratio=0이면 CoM이 꼭대기)
    # h_ratio가 이미 (1-f_cy_ratio)이므로 그대로 사용
    d['stability_number'] = (B_top / 2) / (h_ratio + 1e-6)

    # Moment Ratio: (B_top/2 - t_ecc_2d) / (t_ecc_2d + ε)
    # 양수=안정 여유 있음, 0=전도 임계, 음수=전도
    d['moment_ratio'] = (B_top / 2 - d['t_ecc_2d']) / (d['t_ecc_2d'] + 1e-6)

    # P-delta Risk Index: t_ecc_2d × h_ratio / (B_top + ε)
    # 높이×편심/기저폭 — 무차원 P-delta 인자
    d['pd_risk_index'] = d['t_ecc_2d'] * h_ratio / (B_top + 1e-6)

    return d


def main(args=None):
    v2_path = _FEAT_DIR / 'combined_features_v2.csv'
    if not v2_path.exists():
        # v2가 없으면 v3(기존)에서 1세대만 있는 버전으로 시도
        raise FileNotFoundError(
            f"combined_features_v2.csv 없음: {v2_path}\n"
            "먼저 extract_base.py (Step 1) 실행 필요"
        )

    df = pd.read_csv(v2_path)
    print(f"[1] combined_features_v2.csv 로드: {df.shape[0]}행, {df.shape[1]}컬럼")

    required = [
        't_footprint_area', 't_cx_offset', 't_cy_offset',
        't_left_mass_ratio', 't_compactness',
        'f_cx_offset', 'f_cy_offset', 'f_cy_ratio',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필요 컬럼 누락: {missing}")
    print(f"[2] 필요 컬럼 확인 완료")

    df = add_physics_features(df)
    print(f"[3] 2세대 피처 {len(ENGINEERED_COLS)}종 추가 → 총 {df.shape[1]}컬럼")
    for col in ENGINEERED_COLS:
        null_cnt = df[col].isnull().sum()
        inf_cnt  = np.isinf(df[col]).sum() if df[col].dtype in [np.float64, np.float32] else 0
        print(f"    {col:<25} null={null_cnt} inf={inf_cnt}")

    out_path = _FEAT_DIR / 'combined_features_v3.csv'
    df.to_csv(out_path, index=False)
    print(f"\n[4] 저장 완료: {out_path}")
    print(f"    최종 shape: {df.shape}")
    print("\n✅ combined_features_v3.csv 생성 완료.")


if __name__ == '__main__':
    main()
