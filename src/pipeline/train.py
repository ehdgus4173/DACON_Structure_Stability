import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import TRAIN_CSV, DEV_CSV, TRAIN_DIR, DEV_DIR, CHECKPOINT_DIR, PROJECT_ROOT, PHYS_COLS_V2
from src.dataset import MultiViewDataset, get_transforms
from src.models import MultiViewResNet


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

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args(args if args is not None else [])

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df   = pd.read_csv(TRAIN_CSV)
    dev_df     = pd.read_csv(DEV_CSV)
    feature_df = pd.read_csv(PROJECT_ROOT / "features" / "combined_features_v3.csv")

    train_transform, val_transform = get_transforms()

    train_dataset = MultiViewDataset(
        df=train_df, root_dir=str(TRAIN_DIR), transform=train_transform,
        is_test=False, feature_df=feature_df, feature_cols=PHYS_COLS_V2
    )
    val_dataset = MultiViewDataset(
        df=dev_df, root_dir=str(DEV_DIR), transform=val_transform,
        is_test=False, feature_df=feature_df, feature_cols=PHYS_COLS_V2
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    model     = MultiViewResNet(num_classes=1, num_phys_features=len(PHYS_COLS_V2)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_logloss   = float('inf')
    best_model_path = CHECKPOINT_DIR / "best_model.pth"

    print(f"Training {args.epochs} epochs | phys_features={len(PHYS_COLS_V2)}dim")
    for epoch in range(1, args.epochs + 1):
        train_loss           = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_logloss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{args.epochs}]  "
              f"train_loss={train_loss:.4f}  "
              f"val_logloss={val_logloss:.4f}  val_acc={val_acc:.4f}")

        if val_logloss < best_logloss:
            best_logloss = val_logloss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> best model saved (logloss={best_logloss:.4f})")

        scheduler.step()

if __name__ == '__main__':
    main()
