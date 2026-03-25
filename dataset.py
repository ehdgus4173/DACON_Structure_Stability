import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MultiViewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: str, transform=None, 
                 is_test: bool = False, feature_df: pd.DataFrame = None, 
                 feature_cols: list = None):
        """
        Args:
            df: id, label 정보가 있는 메타데이터
            root_dir: 이미지 뷰어 루트 (train/dev/test)
            transform: 이미지 전처리
            is_test: 테스트 여부 (label 미반환)
            feature_df: 사전에 추출된 물리(tabular) 피처가 포함된 데이터프레임
            feature_cols: feature_df에서 모델로 넘겨줄 피처 컬럼들
        """
        # 확실한 인덱싱을 위해 리셋
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.label_map = {'stable': 0, 'unstable': 1}
        
        # 피처 로딩
        self.feature_df = feature_df.set_index('id') if feature_df is not None else None
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_id = str(self.df.iloc[idx]['id'])
        folder_path = os.path.join(self.root_dir, sample_id)
        
        # 1. 이미지 (views) 로딩
        views = []
        for name in ["front", "top"]:
            img_path = os.path.join(folder_path, f"{name}.png")
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            views.append(image)
            
        # 2. 물리 피처 로딩 (있다면)
        feats = None
        if self.feature_df is not None and self.feature_cols is not None:
            # 1D Tensor 형태로 변환
            feats = torch.tensor(
                self.feature_df.loc[sample_id, self.feature_cols].astype(float).values,
                dtype=torch.float32
            )
            
        # 3. 반환 로직 분기
        if feats is not None:
            if self.is_test:
                return views, feats
            label = self.label_map[self.df.iloc[idx]['label']]
            return views, feats, label
            
        else:
            if self.is_test:
                return views
            label = self.label_map[self.df.iloc[idx]['label']]
            return views, label

def get_transforms(img_size: int = 224):
    """기본 이미지 전처리 (Augmentation 강화 반영 예정)"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform
