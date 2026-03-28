"""
피처 중요도 분석 — combined_features_v3.csv 기반 (팀원 피처 20개)
목적: 새 피처들의 실제 기여도 파악 + 그룹별 Ablation 기준 설정
실행: conda stability 환경에서
      cd DACON_Structure_Stability/notebooks
      python shap_analysis.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 한글 깨짐 방지
import matplotlib.pyplot as plt
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

# config.py에서 PHYS_COLS_V2 로드
try:
    from config import PHYS_COLS_V2, DATASET_DIR, PROJECT_ROOT
    DATA_DIR   = DATASET_DIR
    FEAT_PATH  = PROJECT_ROOT / 'features' / 'combined_features_v3.csv'
    print(f"config.py 로드 완료")
    print(f"DATA_DIR:  {DATA_DIR}")
    print(f"FEAT_PATH: {FEAT_PATH}")
except ImportError:
    print("config.py 없음 — 경로 직접 설정")
    DATA_DIR  = Path("C:/Users/userPC/Desktop/Univ/2026/1학기/structure-stability/data")
    FEAT_PATH = REPO_ROOT / 'features' / 'combined_features_v3.csv'
    PHYS_COLS_V2 = None

# ================================================================
# 1. 피처 CSV 로드
# ================================================================
if not FEAT_PATH.exists():
    raise FileNotFoundError(
        f"피처 파일 없음: {FEAT_PATH}\n"
        "먼저 실행: python src/features/extract_base.py && python src/features/extract_advanced.py"
    )

feature_df = pd.read_csv(FEAT_PATH)
print(f"\n피처 CSV 로드: {feature_df.shape}")
print(f"컬럼 수: {len(feature_df.columns)}")

# 사용할 피처 컬럼 결정
if PHYS_COLS_V2:
    # config.py의 PHYS_COLS_V2 사용 (20개)
    phys_cols = [c for c in PHYS_COLS_V2 if c in feature_df.columns]
    missing   = [c for c in PHYS_COLS_V2 if c not in feature_df.columns]
    if missing:
        print(f"WARNING: 누락된 피처: {missing}")
else:
    # 자동 선택 (숫자형 컬럼 전체)
    phys_cols = [c for c in feature_df.columns
                 if c != 'id' and feature_df[c].dtype in ['float64', 'float32', 'int64']]

print(f"\n사용할 피처 {len(phys_cols)}개:")
for i, c in enumerate(phys_cols):
    print(f"  {i+1:2d}. {c}")

# ================================================================
# 2. 라벨 로드 및 피처 병합
# ================================================================
train_df = pd.read_csv(Path(str(DATA_DIR)) / 'train.csv')
dev_df   = pd.read_csv(Path(str(DATA_DIR)) / 'dev.csv')
all_df   = pd.concat([train_df, dev_df], ignore_index=True)
all_df['id'] = all_df['id'].astype(str)
feature_df['id'] = feature_df['id'].astype(str)

# train/dev 샘플만 필터 (test 제외)
train_ids = set(all_df['id'].values)
feat_train = feature_df[feature_df['id'].isin(train_ids)].copy()
merged = all_df.merge(feat_train[['id'] + phys_cols], on='id', how='inner')
print(f"\n병합 후 샘플: {len(merged)}개")

X = merged[phys_cols].fillna(0).values.astype(np.float32)
y = (merged['label'] == 'unstable').astype(int).values

# ================================================================
# 3. XGBoost 학습
# ================================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    base_score=0.5,
    eval_metric='logloss',
    random_state=42
)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

val_probs   = clf.predict_proba(X_val)[:, 1]
val_logloss = log_loss(y_val, val_probs)
val_acc     = accuracy_score(y_val, (val_probs >= 0.5).astype(int))
print(f"\nXGBoost Val LogLoss: {val_logloss:.4f} | Acc: {val_acc:.4f}")

# ================================================================
# 4. 피처 중요도 분석 — gain / weight / cover 3가지
# ================================================================
importance_types = {
    'gain':   '예측 기여도 (gain)',
    'weight': '사용 빈도 (weight)',
    'cover':  '커버 샘플 수 (cover)',
}

FIG_DIR = REPO_ROOT / 'reports' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(20, 8))

importance_results = {}
for ax, (imp_type, desc) in zip(axes, importance_types.items()):
    scores = clf.get_booster().get_score(importance_type=imp_type)
    named  = {}
    for k, v in scores.items():
        idx = int(k[1:])
        if idx < len(phys_cols):
            named[phys_cols[idx]] = v
    for name in phys_cols:
        if name not in named:
            named[name] = 0.0
    importance_results[imp_type] = named

    imp_df = pd.DataFrame({'feature': list(named.keys()),
                            'score':   list(named.values())}).sort_values('score', ascending=True)
    ax.barh(imp_df['feature'], imp_df['score'], color='steelblue')
    ax.set_title(f'{imp_type}\n({desc})', fontsize=11)
    ax.set_xlabel('Score')

plt.suptitle(f'Physical Feature Importance (XGBoost, {len(phys_cols)} features)', fontsize=14, y=1.02)
plt.tight_layout()
save_path = str(FIG_DIR / 'feature_importance_v2.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n저장: {save_path}")

# ================================================================
# 5. gain 기준 순위 출력 + 그룹별 분석
# ================================================================
gain_df = pd.DataFrame({'feature': list(importance_results['gain'].keys()),
                         'gain':   list(importance_results['gain'].values())}
                        ).sort_values('gain', ascending=False).reset_index(drop=True)

print("\n=== gain 기준 피처 중요도 (높을수록 중요) ===")
print(f"{'순위':>4} {'피처':35} {'gain':>8}")
print("-" * 52)

# 그룹 정의
GROUP_A = {'f_cx_offset', 'f_mass_upper_ratio', 't_compactness',
           't_cx_offset', 't_left_mass_ratio'}
GROUP_B = {'f_cy_ratio', 't_frontback_mass_ratio', 't_pa_cx_offset', 't_pa_cy_offset'}
GROUP_C = {'FS_overturning', 'kern_ratio', 'effective_eccentricity',
           'eccentric_combined', 'p_delta_eccentricity'}
GROUP_D = {'front_grid_detected', 'front_grid_tilt_angle', 'front_grid_perspective_ratio',
           'top_grid_detected', 'top_grid_tilt_angle', 'top_grid_perspective_ratio'}

group_map = {}
for f in GROUP_A: group_map[f] = 'A (기존)'
for f in GROUP_B: group_map[f] = 'B (신규 1세대)'
for f in GROUP_C: group_map[f] = 'C (2세대 구조공학)'
for f in GROUP_D: group_map[f] = 'D (격자 시점)'

for i, row in gain_df.iterrows():
    grp = group_map.get(row['feature'], '?')
    print(f"{i+1:>4}. {row['feature']:35} {row['gain']:8.3f}  [{grp}]")

# 그룹별 평균 중요도
print("\n=== 그룹별 평균 gain ===")
for grp_name, grp_set in [('A 기존 피처', GROUP_A), ('B 신규 1세대', GROUP_B),
                            ('C 2세대 구조공학', GROUP_C), ('D 격자 시점', GROUP_D)]:
    grp_gains = [importance_results['gain'].get(f, 0) for f in grp_set]
    print(f"  {grp_name:20}: 평균 {np.mean(grp_gains):.3f}  합 {np.sum(grp_gains):.3f}")

# ================================================================
# 6. 권장 physics_dim 조합 제안
# ================================================================
print("\n=== Ablation 실험 권장 조합 ===")
print("  EXP-017: dim=20 (전체)")
print("  EXP-018: dim=20 + Phys MLP")
print("  EXP-019: dim=14 (격자 D 제외)")
print("  EXP-019b: dim=9 (격자 D + 2세대 C 제외)")
print("  EXP-019c: dim=5 (기존 A만)")
print("\n각 실험 결과로 어떤 그룹이 실제로 기여하는지 확인 후 최적 dim 결정")

plt.show()
print("\n분석 완료.")
