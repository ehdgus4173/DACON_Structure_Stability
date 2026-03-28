# DACON Structure Stability Pipeline

데이콘 "구조물 안정성 물리 추론 AI 경진대회" 모델 파이프라인.

2가지 시점 이미지(front + top)와 이미지 기반 물리 피처를 결합한
MultiViewResNet으로 구조물의 stable/unstable 여부를 예측합니다.

---

## 디렉토리 구성 가이드

⚠️ 중요: 이 저장소는 코드만 포함합니다. 데이터셋은 용량이 크기 때문에 GitHub에 올리지 않습니다.
반드시 아래 구조대로 로컬에 폴더를 직접 배치해야 코드가 정상 동작합니다.

클론 후 본인 PC의 폴더 구조가 정확히 아래와 같아야 합니다:

```
C:\Users\[내_사용자명]\Desktop\DACON\
│
├── dataset\                        ← 직접 만들어야 합니다 (GitHub에 없음)
│   ├── train\                      ← DACON에서 받은 학습 데이터 폴더
│   ├── dev\                        ← DACON에서 받은 검증 데이터 폴더
│   ├── test\                       ← DACON에서 받은 테스트 데이터 폴더
│   ├── train.csv
│   └── dev.csv
│
└── DACON_Structure_Stability/
    ├── run_pipeline.py
    ├── config.py
    ├── src/
    ...
    └── submission.csv
```

데이터셋 배치 방법

1. DACON 대회 페이지에서 데이터를 다운로드합니다.
2. 압축을 풀고 train, dev, test 폴더와 train.csv, dev.csv를 `C:\Users\[내_사용자명]\Desktop\DACON\dataset\` 안에 넣어주세요.

---

## 📁 디렉토리 구조

```
DACON_Structure_Stability/
├── run_pipeline.py               ← 전체 실행 진입점 (모든 실행은 여기서)
├── config.py                     ← 경로 상수 (DATASET_DIR, CHECKPOINT_DIR 등)
├── src/
│   ├── dataset.py                ← MultiViewDataset (front+top 2-view 로딩)
│   ├── models.py                 ← MultiViewResNet (CNN + Phys MLP fusion)
│   ├── features/
│   │   ├── extract_base.py       ← Step 1: 픽셀 + 이미지 기반 물리 피처 추출
│   │   └── extract_advanced.py   ← Step 2: 2세대 구조공학 피처 계산
│   └── pipeline/
│       ├── train.py              ← Step 4: 학습 루프
│       └── inference.py          ← Step 5: 추론 및 submission.csv 생성
├── features/                     ← [자동 생성] 피처 CSV
├── checkpoints/                  ← [자동 생성] best_model.pth
└── submission.csv                ← [자동 생성] 최종 제출 파일
```

---

## 🚀 실행 방법

```bash
cd C:\Users\(사용자이름)\Desktop\DACON\DACON_Structure_Stability

# 전체 파이프라인 (피처 추출 → 검증 → 학습 → 추론)
python run_pipeline.py --mode all

# 피처 CSV 이미 있으면 추출 스킵
python run_pipeline.py --mode all --skip_extract

# 학습만
python run_pipeline.py --mode train

# 추론만 (학습된 모델 있을 때)
python run_pipeline.py --mode inference
```

### 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--mode` | `all` | `train` / `inference` / `all` |
| `--epochs` | `20` | 학습 에폭 수 |
| `--batch_size` | `32` | 배치 사이즈 |
| `--lr` | `3e-4` | 학습률 |
| `--skip_extract` | `False` | 피처 CSV 존재 시 추출 단계 스킵 |
| `--output` | `submission.csv` | 추론 결과 파일명 |

---

## ⚙️ 파이프라인 내부 실행 순서

```
Step 1. extract_base.py
        → train/dev/test 전체 이미지에서 픽셀 + 물리 피처 직접 추출
        → features/combined_features_v2.csv 생성

Step 2. extract_advanced.py
        → v2 기반으로 구조공학 2세대 피처 9종 계산
           (FS_overturning, kern_ratio, effective_eccentricity 등)
        → features/combined_features_v3.csv 생성

Step 3. 자동 검증 (validate_features)
        → TEST_ 샘플 피처값 정상 여부 확인
        → 이상 감지 시 중단 (f_cx_offset 전부 0 또는 FS_overturning 이상값 다수)

Step 4. train.py
        → combined_features_v3.csv + 이미지로 MultiViewResNet 학습
        → val logloss 최저 시점 checkpoints/best_model.pth 저장

Step 5. inference.py
        → best_model.pth 로드 → test 셋 추론
        → submission.csv 생성
```

---

## 🏗 모델 아키텍처

```
입력: front.png + top.png + 물리 피처 20종 (PHYS_COLS_V2)
                        ↓
MultiViewResNet
├── ResNet18 Backbone (ImageNet pretrained, shared weights)
│   ├── front.png → (Batch, 512)
│   └── top.png   → (Batch, 512)
│                        ↓ concat
│                  img_feat (Batch, 1024)
│
├── Phys MLP
│   20dim → Linear(64) → BN → ReLU → Dropout(0.4) → Linear(32)
│                        ↓ concat
│                  combined (Batch, 1056)
│
└── Classifier
    Linear(256) → ReLU → Dropout(0.4) → Linear(1)
    손실함수: BCEWithLogitsLoss
```

**물리 피처 21종 (PHYS_COLS_V2):**
`f_cx_offset`, `f_cy_ratio`, `t_compactness`, `t_cx_offset`, `t_cy_offset`, `t_left_mass_ratio`, `t_frontback_mass_ratio`,
`FS_top`, `kern_top`, `t_ecc_2d`, `mass_asymmetry_2d`, `support_margin_min`, `height_support_risk`, `compact_ecc`, `t_compactness_sq`,
`front_grid_detected`, `front_grid_tilt_angle`, `front_grid_perspective_ratio`, `top_grid_detected`, `top_grid_tilt_angle`, `top_grid_perspective_ratio`

---

## 📦 설치

```bash
pip install -r requirements.txt
```
