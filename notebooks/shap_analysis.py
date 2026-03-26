"""
피처 중요도 분석 — XGBoost 내장 feature_importance 기반
목적: 현재 물리 피처 6개 중 실제 기여도 높은 피처 파악
     팀원 피처 통합 시 선택 기준으로 활용
실행: conda stability 환경에서
      cd DACON_Structure_Stability/notebooks
      python shap_analysis.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from physics_features import extract_physics_features, PHYSICS_FEATURE_NAMES

# ================================================================
# 1. 데이터 로드
# ================================================================
DATA_DIR = Path("C:/Users/userPC/Desktop/Univ/2026/1학기/structure-stability/data")

train_df = pd.read_csv(DATA_DIR / "train.csv")
dev_df   = pd.read_csv(DATA_DIR / "dev.csv")
all_df   = pd.concat([train_df, dev_df], ignore_index=True)
print(f"전체 샘플: {len(all_df)}")

# ================================================================
# 2. 물리 피처 추출
# ================================================================
features_list = []
labels_list   = []
failed = 0

for _, row in tqdm(all_df.iterrows(), total=len(all_df), desc="피처 추출"):
    sample_id = str(row['id'])
    split     = "train" if sample_id.startswith("TRAIN") else "dev"
    front_path = DATA_DIR / split / sample_id / "front.png"
    top_path   = DATA_DIR / split / sample_id / "top.png"
    try:
        front_img = np.array(Image.open(front_path).convert("RGB"))
        top_img   = np.array(Image.open(top_path).convert("RGB"))
        feats     = extract_physics_features(front_img, top_img)
    except Exception:
        feats = np.zeros(6, dtype=np.float32)
        failed += 1
    features_list.append(feats)
    labels_list.append(1 if row['label'] == 'unstable' else 0)

X = np.array(features_list, dtype=np.float32)
y = np.array(labels_list,   dtype=np.int32)
print(f"추출 완료 — 실패: {failed}/{len(all_df)}")

# ================================================================
# 3. XGBoost 학습
# ================================================================
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = XGBClassifier(
    n_estimators=200,
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
# 4. 피처 중요도 분석 (XGBoost 내장 — 3가지 기준)
# ================================================================
importance_types = {
    'weight':  '사용 횟수 (분기 빈도)',
    'gain':    '평균 정보 이득 (예측 기여도)',
    'cover':   '커버하는 샘플 수'
}

FIG_DIR = REPO_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (imp_type, desc) in zip(axes, importance_types.items()):
    scores = clf.get_booster().get_score(importance_type=imp_type)
    # 피처명 매핑 (f0~f5 → PHYSICS_FEATURE_NAMES)
    named_scores = {
        PHYSICS_FEATURE_NAMES[int(k[1:])]: v
        for k, v in scores.items()
    }
    # 없는 피처는 0으로
    for name in PHYSICS_FEATURE_NAMES:
        if name not in named_scores:
            named_scores[name] = 0.0

    imp_df = pd.DataFrame({
        'feature': list(named_scores.keys()),
        'score':   list(named_scores.values())
    }).sort_values('score', ascending=True)

    ax.barh(imp_df['feature'], imp_df['score'], color='steelblue')
    ax.set_title(f'{imp_type}\n({desc})')
    ax.set_xlabel('Score')

plt.suptitle('물리 피처 중요도 분석 (XGBoost)', fontsize=14, y=1.02)
plt.tight_layout()
save_path = str(FIG_DIR / "feature_importance.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n저장: {save_path}")

# 텍스트 요약 출력
print("\n=== gain 기준 피처 중요도 (예측 기여도) ===")
scores = clf.get_booster().get_score(importance_type='gain')
named = {PHYSICS_FEATURE_NAMES[int(k[1:])]: v for k, v in scores.items()}
for name in PHYSICS_FEATURE_NAMES:
    if name not in named:
        named[name] = 0.0
summary = pd.DataFrame({'feature': list(named.keys()), 'gain': list(named.values())})
summary = summary.sort_values('gain', ascending=False)
print(summary.to_string(index=False))

plt.show()
print("\n완료.")
