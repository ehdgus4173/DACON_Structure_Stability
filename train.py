import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import TRAIN_CSV, DEV_CSV, TRAIN_DIR, DEV_DIR, CHECKPOINT_DIR
from dataset import MultiViewDataset, get_transforms
from models import MultiViewResNet

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for views, labels in pbar:
        views = [v.to(device) for v in views]
        labels = labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(views).view(-1)
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
        for views, labels in tqdm(loader, desc="Validation", leave=False):
            views = [v.to(device) for v in views]
            labels = labels.to(device).float()
            
            outputs = model(views).view(-1)
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
    
    train_transform, test_transform = get_transforms()
    
    train_dataset = MultiViewDataset(train_df, str(TRAIN_DIR), train_transform, is_test=False)
    val_dataset = MultiViewDataset(dev_df, str(DEV_DIR), test_transform, is_test=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model & Optimization
    model = MultiViewResNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_logloss = float('inf')
    best_model_path = CHECKPOINT_DIR / "best_model.pth"

    # Training Loop
    print(f"Starting training for {args.epochs} epochs...")
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
