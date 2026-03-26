"""
Data Augmentation Module

학술 근거:
Ben-David et al.(2010) — Domain Adaptation 이론 토대 🟢
Hendrycks et al.(2020) AugMix — 증강 다양성과 분포 이동 강건성 🟢
Shorten & Khoshgoftaar(2019) — 색상 증강 관행 🟡
Simard et al.(2003), Krizhevsky et al.(2012) — 기하 변환 🟢
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
CUSTOM_STATS = {
    "front": {"mean": [0.800, 0.844, 0.919], "std": [0.174, 0.155, 0.143]},
    "top": {"mean": [0.825, 0.875, 0.951], "std": [0.167, 0.127, 0.087]}
}

def get_train_transform(view: str = "front", norm_version: str = "imagenet", aug_params: dict = None) -> A.Compose:
    if aug_params is None:
        aug_params = {
            "brightness_p": 0.7,
            "gamma_p": 0.5,
            "hsv_p": 0.5,
            "shift_scale_p": 0.4,
            "perspective_p": 0.4,
            "flip_p": 0.3
        }
        
    if norm_version == "custom":
        mean = CUSTOM_STATS[view]["mean"]
        std = CUSTOM_STATS[view]["std"]
    else:
        mean = IMAGENET_STATS["mean"]
        std = IMAGENET_STATS["std"]

    return A.Compose([
        A.Resize(224, 224),
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=aug_params["brightness_p"]
        ),  # 🟢 Hendrycks(2020): 밝기 변동이 corruption robustness 향상에 직접 기여 # 🔴 p값 휴리스틱 — Ablation 필요
        A.RandomGamma(
            gamma_limit=(70, 130), p=aug_params["gamma_p"]
        ),  # 🟡 조명 스펙트럼의 비선형 변화 모사 # 🔴 p값 휴리스틱 — Ablation 필요
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=aug_params["hsv_p"]
        ),  # 🟡 Shorten(2019): 색상 증강 관행 및 질감 및 색상 다양성 확보 # 🔴 p값 휴리스틱 — Ablation 필요
        A.Affine(
            translate_percent=0.1, scale=(0.9, 1.1), rotate=(-15, 15), p=aug_params.get('shift_scale_p', 0.4)
        ),  # 🟢 Krizhevsky(2012): 기하 변환. 위치 및 크기 불변성 학습 # 🔴 p값 휴리스틱 — Ablation 필요
        A.Perspective(
            scale=(0.02, 0.05), p=aug_params["perspective_p"]
        ),  # 🟡 카메라 각도 변동 시뮬레이션 관행 — 직접 인용 논문 없음 # 🔴 p값 휴리스틱 — Ablation 필요
        A.HorizontalFlip(
            p=aug_params["flip_p"]
        ),  # 🔴 p값 휴리스틱 — Ablation 필요
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

def get_val_transform(view: str = "front", norm_version: str = "imagenet") -> A.Compose:
    if norm_version == "custom":
        mean = CUSTOM_STATS[view]["mean"]
        std = CUSTOM_STATS[view]["std"]
    else:
        mean = IMAGENET_STATS["mean"]
        std = IMAGENET_STATS["std"]

    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
