import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MultiViewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: str, transform=None, is_test: bool = False):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.label_map = {'stable': 0, 'unstable': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_id = str(self.df.iloc[idx]['id'])
        folder_path = os.path.join(self.root_dir, sample_id)
        
        views = []
        for name in ["front", "top"]:
            img_path = os.path.join(folder_path, f"{name}.png")
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            views.append(image)
            
        if self.is_test:
            return views
        
        label = self.label_map[self.df.iloc[idx]['label']]
        return views, label

def get_transforms(img_size: int = 224):
    """기본 이미지 전처리 파이프라인 반환"""
    train_transform = transforms.Compose([
        # 추후 Data Augmentation(ColorJitter 등) 추가 위치
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform
