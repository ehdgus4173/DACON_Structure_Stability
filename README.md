# DACON: 구조물 안정성 물리 추론 AI 경진대회

본 저장소는 DACON "구조물 안정성 물리 추론 AI 경진대회"를 위한 베이스라인 파이프라인 및 피처 엔지니어링 코드를 포함하고 있습니다. 현재 Baseline (Multi-View ResNet) 아키텍처를 기반으로 모듈화된 학습 및 추론 파이프라인이 구축되어 있습니다.

## 📁 Repository Structure

```text
DACON_Structure_Stability/
├── README.md               # 프로젝트 설명서
├── config.py               # 원본 데이터셋(`dataset/`) 경로 환경 설정
├── dataset.py              # PyTorch Dataset 클래스 (MultiViewDataset)
├── models.py               # 모델 아키텍처 정의 (MultiViewResNet)
├── train.py                # 모델 학습 루프 및 검증, 체크포인트 저장 스크립트
├── inference.py            # 추론 및 submission.csv 파일 생성 스크립트
└── checkpoints/            # 훈련 중 최고 성능을 낸 모델 가중치가 저장되는 폴더
```

## ⚙️ Installation & Setup

1. **저장소 클론하기**

   ```bash
   git clone https://github.com/[YOUR_USERNAME]/DACON_Structure_Stability.git
   cd DACON_Structure_Stability
   ```

2. **데이터셋 연결**

   `config.py` 파일의 데이터 경로가 로컬 환경의 실제 데이터셋 위치를 올바르게 가리키고 있는지 확인하세요. 기본 경로는 본 저장소와 동일한 레벨에 위치한 `건축물진단/codebase/dataset` 폴더를 가리키도록 설정되어 있습니다.

3. **필요 라이브러리 설치**

   원격 또는 로컬 환경에서 아래 패키지들을 설치해 주세요. (PyTorch, Pandas, OpenCV, Scikit-Learn 등)

   ```bash
   pip install -r ../건축물진단/codebase/requirements.txt
   ```

## 🚀 How to Run

### 1. 모델 학습 (Training)

학습을 시작하려면 `train.py` 스크립트를 실행합니다. 하이퍼파라미터(`epochs`, `batch_size`, `lr`)를 인자로 전달할 수 있습니다.

```bash
# 기본 파라미터로 학습 (3 epochs, lr=1e-3, batch=32)
python train.py

# 파라미터 지정 학습
python train.py --epochs 10 --batch_size 16 --lr 0.0005
```

- 학습이 진행되는 동안 Epoch 마다 검증 세트(dev 폴더 데이터)의 Log-Loss를 계산합니다.
- 최고 성능(Lowest Log-Loss)을 달성할 때마다 `checkpoints/best_model.pth` 파일이 갱신(저장)됩니다.

### 2. 추론 및 모델 제출 (Inference)

학습이 끝난 최고 성능 모델(`.pth`)을 바탕으로 Test 데이터셋을 평가합니다.

```bash
python inference.py
```

- 실행이 완료되면 동일 디렉토리에 **`submission.csv`** 파일이 생성되며, 이를 DACON 플랫폼에 바로 제출할 수 있습니다.
