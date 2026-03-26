import os
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.features.extract_base import rectify_by_grid


class MultiViewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: str, transform=None,
                 is_test: bool = False, feature_df: pd.DataFrame = None,
                 feature_cols: list = None, rectify: bool = False):
        """
        Args:
            df:           id, label 정보가 있는 메타데이터 DataFrame
            root_dir:     이미지 루트 디렉토리 (train / dev / test)
            transform:    이미지 전처리 transform
            is_test:      True이면 label 미반환
            feature_df:   사전에 추출된 물리 피처 DataFrame
            feature_cols: feature_df에서 모델에 넘길 컬럼 목록 (config.PHYS_COLS_V2)
            rectify:      격자 호모그래피 보정 적용 여부.
                          현재 front 0.2% / top 9.2% 검출률로 비활성화(False)가 기본.
                          검출률 개선 후 True로 전환 가능.
        """
        self.df           = df.reset_index(drop=True)
        self.root_dir     = root_dir
        self.transform    = transform
        self.is_test      = is_test
        self.label_map    = {'stable': 0, 'unstable': 1}
        self.feature_df   = feature_df.set_index('id') if feature_df is not None else None
        self.feature_cols = feature_cols
        self.rectify      = rectify

    def __len__(self):
        return len(self.df)

    def _load_image(self, path: str) -> Image.Image:
        """이미지 로딩. rectify=True일 때만 격자 호모그래피 보정 적용."""
        img_np = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.rectify:
            img_np = rectify_by_grid(img_np)   # 실패 시 원본 반환 (내부 fallback)
        return Image.fromarray(img_np)

    def __getitem__(self, idx):
        sample_id   = str(self.df.iloc[idx]['id'])
        folder_path = os.path.join(self.root_dir, sample_id)

        # 1. 이미지 로딩 (front, top 순서 고정)
        views = []
        for name in ["front", "top"]:
            img = self._load_image(os.path.join(folder_path, f"{name}.png"))
            if self.transform:
                img = self.transform(img)
            views.append(img)

        # 2. 물리 피처 로딩
        feats = None
        if self.feature_df is not None and self.feature_cols is not None:
            feats = torch.tensor(
                self.feature_df.loc[sample_id, self.feature_cols].astype(float).values,
                dtype=torch.float32,
            )

        # 3. 반환
        if feats is not None:
            if self.is_test:
                return views, feats
            return views, feats, self.label_map[self.df.iloc[idx]['label']]
        else:
            if self.is_test:
                return views
            return views, self.label_map[self.df.iloc[idx]['label']]


def get_transforms(img_size: int = 224):
    """학습/검증 이미지 전처리 transform 반환."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, test_transform
