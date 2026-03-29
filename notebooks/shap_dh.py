"""
dh 브랜치 신규 피처 SHAP 분석
기존 20개 + 신규 8개 = 28개 전체 중요도 비교
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from config import PHYS_COLS_V2, DATASET_DIR, PROJECT_ROOT

DATA_DIR  = DATASET_DIR
FEAT_PATH = PROJECT_ROOT / 'features' / 'combined_features_v4.csv'

# 기존 20개 + 신규 8개
NEW_COLS = [
    'FS_top', 'kern_top', 't_ecc_2d', 'mass_asymmetry_2d',
    'support_margin_min', 'height_support_risk', 'compact_ecc', 't_compactness_sq'
]
ALL_COLS = [c for c in PHYS_COLS_V2] + NEW_COLS  # 28개

# 피처 로드
feature_df = pd.read_csv(FEAT_PATH)
print(f"피처 CSV: {feature_df.shape}")

# 라벨 로드
train_df = pd.read_csv(DATA_DIR / 'train.csv')
dev_df   = pd.read_csv(DATA_DIR / 'dev.csv')
all_df   = pd.concat([train_df, dev_df], ignore_index=True)
all_df['id']     = all_df['id'].astype(str)
feature_df['id'] = feature_df['id'].astype(str)

train_ids  = set(all_df['id'].values)
feat_train = feature_df[feature_df['id'].isin(train_ids)].copy()
merged     = all_df.merge(feat_train[['id'] + ALL_COLS], on='id', how='inner')
print(f"병합 후: {len(merged)}개")

X = merged[ALL_COLS].fillna(0).values.astype(np.float32)
y = (merged['label'] == 'unstable').astype(int).values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, eval_metric='logloss', random_state=42)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

val_ll  = log_loss(y_val, clf.predict_proba(X_val)[:,1])
val_acc = accuracy_score(y_val, clf.predict(X_val))
print(f"\nXGBoost Val LogLoss: {val_ll:.4f} | Acc: {val_acc:.4f}")

# gain 중요도 추출
scores = clf.get_booster().get_score(importance_type='gain')
named  = {}
for k, v in scores.items():
    idx = int(k[1:])
    if idx < len(ALL_COLS):
        named[ALL_COLS[idx]] = v
for c in ALL_COLS:
    if c not in named:
        named[c] = 0.0

df_imp = pd.DataFrame({'feature': list(named.keys()),
                       'gain':    list(named.values())}
                     ).sort_values('gain', ascending=False).reset_index(drop=True)

# 그룹 정의
GROUP_A = {'f_cx_offset','f_mass_upper_ratio','t_compactness','t_cx_offset','t_left_mass_ratio'}
GROUP_B = {'f_cy_ratio','t_frontback_mass_ratio','t_pa_cx_offset','t_pa_cy_offset'}
GROUP_C = {'FS_overturning','kern_ratio','effective_eccentricity','eccentric_combined','p_delta_eccentricity'}
GROUP_D = {'front_grid_detected','front_grid_tilt_angle','front_grid_perspective_ratio',
           'top_grid_detected','top_grid_tilt_angle','top_grid_perspective_ratio'}
GROUP_E = set(NEW_COLS)  # dh 신규

gmap = {}
for f in GROUP_A: gmap[f] = 'A(기존)'
for f in GROUP_B: gmap[f] = 'B(신규1세대)'
for f in GROUP_C: gmap[f] = 'C(2세대기존)'
for f in GROUP_D: gmap[f] = 'D(격자)'
for f in GROUP_E: gmap[f] = 'E(dh신규)'

total = df_imp['gain'].sum()

# 출력
print(f"\n{'순위':>4} {'피처':35} {'gain':>7} {'비율%':>6} {'그룹'}")
print("=" * 65)
for i, row in df_imp.iterrows():
    pct = row['gain'] / total * 100 if total > 0 else 0
    grp = gmap.get(row['feature'], '?')
    marker = '🆕' if row['feature'] in GROUP_E else '  '
    print(f"{i+1:>4}. {marker}{row['feature']:33} {row['gain']:>7.3f} {pct:>5.1f}%  [{grp}]")

print(f"\n{'그룹':15} {'평균gain':>9} {'합gain':>9} {'비율%':>7} {'유효/전체'}")
print("-" * 55)
for grp, gset, name in [
    ('A(기존)',      GROUP_A, '기존 5개'),
    ('B(신규1세대)', GROUP_B, '신규 4개'),
    ('C(2세대기존)', GROUP_C, '2세대기존 5개'),
    ('D(격자)',      GROUP_D, '격자 6개'),
    ('E(dh신규)',    GROUP_E, 'dh신규 8개'),
]:
    gains   = [named.get(f, 0) for f in gset]
    g_sum   = sum(gains)
    g_mean  = g_sum / len(gains)
    g_pct   = g_sum / total * 100 if total > 0 else 0
    nonzero = sum(1 for g in gains if g > 0)
    print(f"{grp:15} {g_mean:>9.3f} {g_sum:>9.3f} {g_pct:>6.1f}%  {nonzero}/{len(gset)}")

# 기존 kern_ratio vs 신규 kern_top / FS_overturning vs FS_top 비교
print(f"\n=== 기존 vs 신규 직접 비교 ===")
pairs = [
    ('kern_ratio', 'kern_top'),
    ('FS_overturning', 'FS_top'),
    ('eccentric_combined', 't_ecc_2d'),
]
for old, new in pairs:
    o = named.get(old, 0)
    n = named.get(new, 0)
    winner = '신규 ✅' if n > o else '기존 ✅'
    print(f"  {old:25} {o:.3f}  vs  {new:25} {n:.3f}  → {winner}")

# 그래프 저장
FIG_DIR = REPO_ROOT / 'reports' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

colors = []
for f in df_imp['feature']:
    g = gmap.get(f, '?')
    colors.append({'A(기존)':'steelblue','B(신규1세대)':'seagreen',
                   'C(2세대기존)':'orange','D(격자)':'mediumpurple',
                   'E(dh신규)':'crimson'}.get(g, 'gray'))

fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(df_imp['feature'][::-1], df_imp['gain'][::-1], color=colors[::-1])
ax.set_xlabel('Gain')
ax.set_title(f'Feature Importance — 28 features (dh branch added, XGBoost gain)')

from matplotlib.patches import Patch
legend = [Patch(color='steelblue', label='A 기존'),
          Patch(color='seagreen',  label='B 신규1세대'),
          Patch(color='orange',    label='C 2세대기존'),
          Patch(color='mediumpurple', label='D 격자'),
          Patch(color='crimson',   label='E dh신규')]
ax.legend(handles=legend, loc='lower right')
plt.tight_layout()
save_path = str(FIG_DIR / 'feature_importance_v3_dh.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n그래프 저장: {save_path}")
plt.show()
print("\n분석 완료.")
