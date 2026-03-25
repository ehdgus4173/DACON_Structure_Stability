import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import TRAIN_CSV, DEV_CSV, TRAIN_DIR, DEV_DIR, CHECKPOINT_DIR, PROJECT_ROOT
from dataset import MultiViewDataset, get_transforms
from models import MultiViewResNet

PHYS_COLS_V2 = [
    't_compactness', 'f_cx_offset', 't_left_mass_ratio',
    't_cx_offset', 'f_mass_upper_ratio',
    'FS_overturning', 'kern_ratio', 'effective_eccentricity',
    'eccentric_combined', 'p_delta_eccentricity'
]

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for views, feats, labels in pbar:
        views = [v.to(device) for v in views]
        feats = feats.to(device)
        labels = labels.to(device).float()
        
        optimizer.zero_grad()
        # 모델의 forward_pass에 이미지 뷰어 배열과 피처 텐서 모두 전달
        outputs = model(views, feats).view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return train_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for views, feats, labels in tqdm(loader, desc="Validation", leave=False):
            views = [v.to(device) for v in views]
            feats = feats.to(device)
            labels = labels.to(device).float()
            
            outputs = model(views, feats).view(-1)
            probs = torch.sigmoid(outputs)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs, dtype=np.float64)
    all_labels = np.array(all_labels, dtype=np.float64)
    
    eps = 1e-15
    p = np.clip(all_probs, eps, 1 - eps)
    logloss = -np.mean(all_labels * np.log(p) + (1 - all_labels) * np.log(1 - p))
    acc = np.mean((all_probs > 0.5) == all_labels)
    
    return logloss, acc

def main():
    parser = argparse.ArgumentParser()
    # 제출 모델 튜닝 권장사항 (Epoch 20)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loaders
    train_df = pd.read_csv(TRAIN_CSV)
    dev_df = pd.read_csv(DEV_CSV)
    
    # [NEW] 물리 피처 데이터프레임 로드
    feature_csv_path = PROJECT_ROOT / "features" / "combined_features_v3.csv"
    feature_df = pd.read_csv(feature_csv_path)
    
    train_transform, test_transform = get_transforms()
    
    train_dataset = MultiViewDataset(
        df=train_df, root_dir=str(TRAIN_DIR), transform=train_transform, 
        is_test=False, feature_df=feature_df, feature_cols=PHYS_COLS_V2
    )
    val_dataset = MultiViewDataset(
        df=dev_df, root_dir=str(DEV_DIR), transform=test_transform, 
        is_test=False, feature_df=feature_df, feature_cols=PHYS_COLS_V2
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model & Optimization (num_phys_features=10)
    model = MultiViewResNet(num_classes=1, num_phys_features=len(PHYS_COLS_V2)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_logloss = float('inf')
    best_model_path = CHECKPOINT_DIR / "best_model.pth"

    # Training Loop
    print(f"Starting training for {args.epochs} epochs... with Physical Feature Fusion ({len(PHYS_COLS_V2)} dims)")
    for epoch in range(1, args.epochs + 1):
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_logloss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch}/{args.epochs}]")
        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Val LogLoss: {val_logloss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_logloss < best_logloss:
            best_logloss = val_logloss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Best model saved to {best_model_path} (LogLoss: {best_logloss:.4f})")

if __name__ == '__main__':
    main()
