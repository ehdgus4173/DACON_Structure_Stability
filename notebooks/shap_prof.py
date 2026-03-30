"""
교수 자문 반영 후 SHAP 분석
전체 24개 피처 (기존 21개 중 t_compactness_sq 제거 + 신규 3개 추가)
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
FEAT_PATH = PROJECT_ROOT / 'features' / 'combined_features_v3.csv'
ALL_COLS  = PHYS_COLS_V2  # 24개

feature_df = pd.read_csv(FEAT_PATH)
print(f"피처 CSV: {feature_df.shape}")

train_df = pd.read_csv(DATA_DIR / 'train.csv')
dev_df   = pd.read_csv(DATA_DIR / 'dev.csv')
all_df   = pd.concat([train_df, dev_df], ignore_index=True)
all_df['id']     = all_df['id'].astype(str)
feature_df['id'] = feature_df['id'].astype(str)

train_ids  = set(all_df['id'].values)
feat_train = feature_df[feature_df['id'].isin(train_ids)].copy()
merged     = all_df.merge(feat_train[['id'] + ALL_COLS], on='id', how='inner')
print(f"병합 후: {len(merged)}개")

missing = [c for c in ALL_COLS if c not in feature_df.columns]
if missing:
    print(f"WARNING 누락 컬럼: {missing}")
    ALL_COLS = [c for c in ALL_COLS if c in feature_df.columns]

X = merged[ALL_COLS].fillna(0).values.astype(np.float32)
y = (merged['label'] == 'unstable').astype(int).values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, eval_metric='logloss', random_state=42)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

val_ll  = log_loss(y_val, clf.predict_proba(X_val)[:,1])
val_acc = accuracy_score(y_val, clf.predict(X_val))
print(f"\nXGBoost Val LogLoss: {val_ll:.4f} | Acc: {val_acc:.4f}")

# gain 중요도
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

total = df_imp['gain'].sum()

# 그룹 정의
NEW_PROF = {'stability_number', 'moment_ratio', 'pd_risk_index'}
OLD_BAD  = {'elongation_index'}  # t_compactness_sq 대체
FIXED    = {'height_support_risk'}  # 부호 수정됨

def tag(feat):
    if feat in NEW_PROF: return '★교수신규'
    if feat in FIXED:    return '★부호수정'
    if feat in OLD_BAD:  return '★대체'
    return ''

print(f"\n{'순위':>4} {'피처':30} {'gain':>7} {'비율%':>6}  {'비고'}")
print("=" * 65)
cumsum = 0
for i, row in df_imp.iterrows():
    pct = row['gain'] / total * 100 if total > 0 else 0
    cumsum += pct
    t = tag(row['feature'])
    print(f"{i+1:>4}. {row['feature']:30} {row['gain']:>7.3f} {pct:>5.1f}%  {t}  (누적{cumsum:5.1f}%)")

# optimal cutoff 분석
print("\n=== Optimal Cutoff 분석 ===")
print(f"{'dim':>5} {'누적gain%':>10} {'최하위 피처':30} {'최하위 gain':>12} {'추천'}")
print("-" * 75)
checkpoints = [4, 6, 8, 10, 12, 14, 16, 18, 20]
for dim in checkpoints:
    top = df_imp.head(dim)
    cum = top['gain'].sum() / total * 100
    last_feat = top.iloc[-1]
    next_gain = df_imp.iloc[dim]['gain'] if dim < len(df_imp) else 0
    gap = last_feat['gain'] - next_gain
    recommend = "◀ 추천" if gap > 0.5 else ""
    print(f"{dim:>5}  {cum:>9.1f}%  {last_feat['feature']:30} {last_feat['gain']:>12.3f}  {recommend}")

# 그래프 저장
FIG_DIR = REPO_ROOT / 'reports' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

colors = []
for f in df_imp['feature']:
    if f in NEW_PROF: colors.append('crimson')
    elif f in FIXED:  colors.append('darkorange')
    elif f in OLD_BAD: colors.append('seagreen')
    elif f in {'front_grid_detected','top_grid_detected'}: colors.append('lightgray')
    else: colors.append('steelblue')

fig, ax = plt.subplots(figsize=(12, 9))
bars = ax.barh(df_imp['feature'][::-1], df_imp['gain'][::-1], color=colors[::-1])
ax.set_xlabel('Gain (XGBoost)')
ax.set_title('Feature Importance — 교수 자문 반영 후 (24개)\n★crimson=신규, orange=부호수정, green=대체')

from matplotlib.patches import Patch
legend = [Patch(color='crimson',   label='★ 교수 신규 3개'),
          Patch(color='darkorange',label='★ height_support_risk (부호수정)'),
          Patch(color='seagreen',  label='★ elongation_index (대체)'),
          Patch(color='steelblue', label='기존 피처'),
          Patch(color='lightgray', label='gain=0 (제거 권장)')]
ax.legend(handles=legend, loc='lower right')
plt.tight_layout()
save_path = str(FIG_DIR / 'feature_importance_v4_prof.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n그래프 저장: {save_path}")
plt.show()
print("\n분석 완료.")
