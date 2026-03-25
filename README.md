# 구조물 안정성 물리 추론 AI — DACON_Structure_Stability

DACON "구조물 안정성 물리 추론 AI 경진대회" 팀 저장소입니다.  
front / top 2장의 이미지에서 구조물의 안정(stable) / 불안정(unstable) 여부를 예측합니다.

---

## 📌 목차
1. [로컬 환경 세팅 (처음 한 번만)](#1-로컬-환경-세팅-처음-한-번만)
2. [디렉토리 구성 가이드](#2-디렉토리-구성-가이드)
3. [학습 실행](#3-학습-실행)
4. [추론 및 제출 파일 생성](#4-추론-및-제출-파일-생성)
5. [피처 CSV 재생성 방법](#5-피처-csv-재생성-방법)
6. [모델 구조 요약](#6-모델-구조-요약)
7. [파일별 역할 설명](#7-파일별-역할-설명)

---

## 1. 로컬 환경 세팅 (처음 한 번만)

### 1-1. Git이 없다면 먼저 설치
[https://git-scm.com/downloads](https://git-scm.com/downloads) 에서 Windows용 Git을 다운로드해 설치합니다.  
설치 중 모든 옵션은 기본값(Next → Next → Install)으로 진행해도 됩니다.

### 1-2. 저장소 클론

바탕화면의 `DACON` 폴더 안에 코드를 받아야 합니다.

1. 바탕화면에 `DACON` 폴더가 없다면 만들어두세요.
2. `DACON` 폴더 안에서 **우클릭 → "Git Bash Here"** (또는 파워쉘 열기)
3. 아래 명령어 입력:
```bash
git clone https://github.com/ehdgus4173/DACON_Structure_Stability.git
```

완료되면 `DACON\DACON_Structure_Stability` 폴더가 생깁니다.

### 1-3. Python 패키지 설치

터미널(명령 프롬프트 / PowerShell)에서 아래를 실행합니다:
```bash
cd C:\Users\[내_사용자명]\Desktop\DACON\DACON_Structure_Stability
pip install -r requirements.txt
```

> 💡 `[내_사용자명]` 은 본인 Windows 로그인 이름으로 바꿔주세요. (예: `ehdgu`)

---

## 2. 디렉토리 구성 가이드

> ⚠️ **중요**: 이 저장소는 코드만 포함합니다. 데이터셋은 용량이 크기 때문에 GitHub에 올리지 않습니다.  
> 반드시 아래 구조대로 로컬에 폴더를 직접 배치해야 코드가 정상 동작합니다.

클론 후 본인 PC의 폴더 구조가 **정확히** 아래와 같아야 합니다:
```
C:\Users\[내_사용자명]\Desktop\DACON\
│
├── dataset\                        ← ✅ 직접 만들어야 합니다 (GitHub에 없음)
│   ├── train\                      ← DACON에서 받은 학습 데이터 폴더
│   ├── dev\                        ← DACON에서 받은 검증 데이터 폴더
│   ├── test\                       ← DACON에서 받은 테스트 데이터 폴더
│   ├── train.csv
│   └── dev.csv
│
└── DACON_Structure_Stability\      ← ✅ git clone으로 자동 생성됨
    ├── config.py
    ├── train.py
    ├── inference.py
    ├── ...
    └── features\
        └── combined_features_v2.csv  ← ✅ 팀원에게 별도 공유받아야 합니다
```

### 데이터셋 배치 방법

1. [DACON 대회 페이지](https://dacon.io/competitions/official/236686)에서 데이터를 다운로드합니다.
2. 압축을 풀고 `train`, `dev`, `test` 폴더와 `train.csv`, `dev.csv`를  
   **`C:\Users\[내_사용자명]\Desktop\DACON\dataset\`** 안에 넣어주세요.

### 피처 CSV 배치 방법

`features\combined_features_v2.csv` (또는 v3)는 팀 내 공유 드라이브나 카톡으로 받아서  
**`DACON_Structure_Stability\features\`** 폴더를 직접 만들고 그 안에 넣어주세요.

---

## 3. 학습 실행

터미널에서 `DACON_Structure_Stability` 폴더로 이동 후 실행합니다.
```bash
cd C:\Users\[내_사용자명]\Desktop\DACON\DACON_Structure_Stability

# 기본 실행 (epoch=3, lr=1e-3, batch=32)
python train.py

# 권장 설정으로 실행
python train.py --epochs 20 --batch_size 16 --lr 0.0005
```

학습이 끝나면 `checkpoints\best_model.pth` 파일이 생성됩니다.  
매 epoch마다 Val LogLoss가 개선될 때마다 자동으로 덮어씌워집니다.

---

## 4. 추론 및 제출 파일 생성
```bash
python inference.py
```

실행 완료 후 `submission.csv` 파일이 생성됩니다.  
이 파일을 [DACON 제출 페이지](https://dacon.io/competitions/official/236686)에 업로드하면 됩니다.

---

## 5. 피처 CSV 재생성 방법

이미 `features/combined_features_v2.csv`가 있다면 이 단계는 건너뛰어도 됩니다.  
처음부터 피처를 추출하고 싶다면 아래 순서대로 실행합니다.
```bash
# Step 1: 픽셀 기반 피처 추출 → combined_features_v2.csv 생성
python extract_pixel_features.py

# Step 2: 구조공학 2세대 피처 추가 → combined_features_v3.csv 생성
python extract_physics_v2.py
```

`train.py`와 `inference.py`는 기본적으로 `combined_features_v3.csv`를 사용합니다.

---

## 6. 모델 구조 요약
```
[front 이미지] ──┐
                ├── ResNet18 (shared) ──┐
[top 이미지] ────┘                       ├── Concat (1024-dim)
                                        │                      ├── Classifier → 안정/불안정 확률
[물리 피처 10종] ── MLP (32-dim)  ────────┘
```

- **Backbone**: ResNet18 (ImageNet pretrained, shared weights for front & top)
- **Physics branch**: 10차원 물리 피처 → Linear(64) → Linear(32)
- **Fusion**: CNN 특징(1024) + 물리 특징(32) = 1056차원 → 최종 분류

---

## 7. 파일별 역할 설명

| 파일 | 역할 |
|------|------|
| `config.py` | 데이터셋 경로 등 전역 설정. 경로 문제 발생 시 이 파일 확인 |
| `dataset.py` | 이미지 로딩 및 전처리 (PyTorch Dataset) |
| `models.py` | CNN + 물리 피처 Fusion 모델 정의 |
| `train.py` | 모델 학습 실행 스크립트 |
| `inference.py` | 추론 및 `submission.csv` 생성 스크립트 |
| `extract_pixel_features.py` | front/top 이미지에서 픽셀 기반 피처 추출 |
| `extract_physics_v2.py` | 구조공학 이론 기반 2세대 피처 계산 및 병합 |
| `features/` | 생성된 피처 CSV 저장 폴더 |
| `checkpoints/` | 학습된 모델 가중치 저장 폴더 |
| `requirements.txt` | 필요한 Python 패키지 목록 |

---

## ❓ 자주 발생하는 오류

**`FileNotFoundError: dataset 경로를 찾을 수 없습니다`**  
→ `DACON\dataset\` 폴더 위치를 [섹션 2](#2-디렉토리-구성-가이드)와 비교해서 확인하세요.

**`FileNotFoundError: combined_features_v3.csv`**  
→ `features\` 폴더에 CSV 파일이 없는 것입니다. 팀원에게 파일을 받거나 [섹션 5](#5-피처-csv-재생성-방법)를 실행하세요.

**`ModuleNotFoundError: No module named 'torch'`**  
→ `pip install -r requirements.txt`를 실행하지 않은 것입니다. [섹션 1-3](#1-3-python-패키지-설치)을 다시 따라하세요.

**`CUDA out of memory`**  
→ `--batch_size 8` 또는 `--batch_size 4`로 줄여서 실행해보세요.