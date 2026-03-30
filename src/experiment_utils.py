"""
experiment_utils.py — 팀원 dataset 구조 기반 실험 인프라

변경사항 (3/28):
- 팀원 MultiViewDataset 방식으로 전환
  - 피처 CSV 파일(combined_features_v3.csv) 미리 로드 후 전달
  - batch 형태: (views, feats, label) 튜플 방식
- build_datasets / build_test_dataset 대신 _build_loaders 내부 함수로 통합
- run_experiment, run_inference 전면 수정
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import log_loss, accuracy_score
from torchvision import transforms

from dataset import MultiViewDataset, get_transforms
from model import MultiViewNet


# ================================================================
# Seed 고정
# ================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================================================
# EarlyStopping
# ================================================================
class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience    = patience
        self.counter     = 0
        self.best_loss   = float('inf')
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


# ================================================================
# 데이터 로더 빌더
# ================================================================
def _build_loaders(config: dict, device=None):
    """
    팀원 MultiViewDataset 방식으로 train/val DataLoader 생성.

    - features/combined_features_v3.csv 로드
    - PHYS_COLS_V2 기준 피처 컬럼 선택
    - dev_split_ratio에 따라 dev를 train에 일부 포함
    """
    data_dir      = config['data_dir']
    img_size      = config['img_size']
    dev_ratio     = config.get('dev_split_ratio', 0.5)
    norm_version  = config.get('norm_version', 'custom')
    use_physics   = config.get('use_physics', True)
    physics_dim   = config.get('physics_dim', 20)
    num_workers   = config.get('num_workers', 0)
    batch_size    = config.get('batch_size', 16)
    random_state  = config.get('random_state', 42)

    # ── 정규화 설정 ────────────────────────────────────────────
    if norm_version == 'custom':
        mean, std = [0.4611, 0.4359, 0.3905], [0.2193, 0.2150, 0.2109]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # ── CSV 로드 ────────────────────────────────────────────────
    train_csv = pd.read_csv(os.path.join(data_dir, config.get('train_csv', 'train.csv')))
    dev_csv   = pd.read_csv(os.path.join(data_dir, config.get('dev_csv', 'dev.csv')))

    # ── 피처 CSV 로드 ────────────────────────────────────────────
    # Kaggle 환경이면 working/features, 로컬이면 레포/features
    feat_csv_path = config.get('features_csv', None)
    if feat_csv_path is None:
        # 자동 탐색
        candidates = [
            os.path.join(os.path.dirname(data_dir), '..', 'features', 'combined_features_v3.csv'),
            '/kaggle/working/features/combined_features_v3.csv',
        ]
        for c in candidates:
            if os.path.exists(c):
                feat_csv_path = c
                break
    if feat_csv_path is None or not os.path.exists(str(feat_csv_path)):
        raise FileNotFoundError(
            "combined_features_v3.csv를 찾을 수 없습니다.\n"
            "python src/features/extract_base.py 후 extract_advanced.py 실행 필요"
        )
    feature_df = pd.read_csv(feat_csv_path)

    # ── 피처 컬럼 선택 ──────────────────────────────────────────
    # config에 phys_cols 명시되어 있으면 사용, 없으면 자동 선택
    phys_cols = config.get('phys_cols', None)
    if phys_cols is None:
        # 숫자형 컬럼 중 physics_dim개 자동 선택
        num_cols = [c for c in feature_df.columns
                    if c != 'id' and feature_df[c].dtype in [np.float64, np.float32, np.int64]]
        phys_cols = num_cols[:physics_dim]
    phys_cols = phys_cols[:physics_dim]

    # ── dev_split_ratio 처리 ────────────────────────────────────
    n_dev = len(dev_csv)
    n_train_from_dev = int(n_dev * dev_ratio)

    train_dir = os.path.join(data_dir, 'train')
    dev_dir   = os.path.join(data_dir, 'dev')

    # train/dev samples have different root_dirs, so build separate Datasets and concat
    if n_train_from_dev > 0:
        dev_for_train = dev_csv.sample(n=n_train_from_dev, random_state=random_state)
        dev_for_val   = dev_csv.drop(dev_for_train.index).reset_index(drop=True)
        train_only_ds = MultiViewDataset(
            df=train_csv, root_dir=train_dir,
            transform=train_transform,
            feature_df=feature_df, feature_cols=phys_cols,
        )
        dev_train_ds = MultiViewDataset(
            df=dev_for_train, root_dir=dev_dir,
            transform=train_transform,
            feature_df=feature_df, feature_cols=phys_cols,
        )
        from torch.utils.data import ConcatDataset
        train_ds = ConcatDataset([train_only_ds, dev_train_ds])
    else:
        dev_for_val = dev_csv.copy()
        train_ds = MultiViewDataset(
            df=train_csv, root_dir=train_dir,
            transform=train_transform,
            feature_df=feature_df, feature_cols=phys_cols,
        )

    val_ds = MultiViewDataset(
        df=dev_for_val, root_dir=dev_dir,
        transform=val_transform,
        feature_df=feature_df, feature_cols=phys_cols,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def _build_test_loader(config: dict):
    """Test DataLoader 생성."""
    data_dir     = config['data_dir']
    img_size     = config['img_size']
    norm_version = config.get('norm_version', 'custom')
    physics_dim  = config.get('physics_dim', 20)
    num_workers  = config.get('num_workers', 0)
    batch_size   = config.get('batch_size', 32)

    if norm_version == 'custom':
        mean, std = [0.4611, 0.4359, 0.3905], [0.2193, 0.2150, 0.2109]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # 피처 CSV 로드
    feat_csv_path = config.get('features_csv', None)
    if feat_csv_path is None:
        candidates = [
            os.path.join(os.path.dirname(data_dir), '..', 'features', 'combined_features_v3.csv'),
            '/kaggle/working/features/combined_features_v3.csv',
        ]
        for c in candidates:
            if os.path.exists(c):
                feat_csv_path = c
                break

    feature_df = pd.read_csv(feat_csv_path)
    phys_cols  = config.get('phys_cols', None)
    if phys_cols is None:
        num_cols  = [c for c in feature_df.columns
                     if c != 'id' and feature_df[c].dtype in [np.float64, np.float32, np.int64]]
        phys_cols = num_cols[:physics_dim]
    phys_cols = phys_cols[:physics_dim]

    # Test CSV
    test_dir = os.path.join(data_dir, 'test')
    test_ids = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    test_df  = pd.DataFrame({'id': test_ids})

    test_ds = MultiViewDataset(
        df=test_df, root_dir=test_dir,
        transform=val_transform, is_test=True,
        feature_df=feature_df, feature_cols=phys_cols,
    )
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ================================================================
# 학습 / 평가 함수
# ================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, use_physics=True):
    model.train()
    total_loss = 0.0
    for batch in loader:
        views, feats, labels = batch
        front  = views[0].to(device)
        top    = views[1].to(device)
        feats  = feats.to(device)
        labels = labels.float().view(-1, 1).to(device)

        optimizer.zero_grad()
        phys = feats if use_physics else None
        loss = criterion(model(front, top, phys), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * front.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, use_physics=True):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            views, feats, labels = batch
            front = views[0].to(device)
            top   = views[1].to(device)
            feats = feats.to(device)
            phys  = feats if use_physics else None
            probs = np.clip(
                torch.sigmoid(model(front, top, phys)).cpu().numpy().flatten(),
                1e-7, 1-1e-7
            )
            preds.extend(probs)
            targets.extend(labels.numpy())
    preds, targets = np.array(preds), np.array(targets)
    return log_loss(targets, preds, labels=[0, 1]), \
           accuracy_score(targets, (preds >= 0.5).astype(int))


# ================================================================
# run_experiment
# ================================================================
def run_experiment(config: dict, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_id = config['exp_id']
    set_seed(config['random_state'])

    os.makedirs(f"{config['out_dir']}/logs",   exist_ok=True)
    os.makedirs(f"{config['out_dir']}/models", exist_ok=True)

    with open(f"{config['out_dir']}/logs/exp{exp_id}_config.json", 'w', encoding='utf-8') as f:
        json.dump({k: str(v) for k, v in config.items()}, f, indent=2, ensure_ascii=False)

    train_loader, val_loader = _build_loaders(config, device)

    model = MultiViewNet(
        backbone_name   = config['backbone'],
        fusion_mode     = config['fusion_mode'],
        use_physics     = config['use_physics'],
        physics_dim     = config.get('physics_dim', 20),
        use_phys_mlp    = config.get('use_phys_mlp', False),
        shared_backbone = config.get('shared_backbone', False),
        img_size        = config['img_size'],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    scheduler = None
    if config.get('lr_scheduler') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config.get('lr_min', 1e-5)
        )
    elif config.get('lr_scheduler') == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.get('lr_step_size', 10),
            gamma=config.get('lr_gamma', 0.5)
        )

    es         = EarlyStopping(patience=config['early_stopping_patience'])
    logs       = []
    best_loss  = float('inf')
    model_path = f"{config['out_dir']}/models/{config['model_name']}_v{config['model_version']}_best.pth"

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    print(f"\n=== EXP-{exp_id} 시작 === Train: {n_train} | Val: {n_val}")
    if scheduler:
        print(f"LR Scheduler: {config['lr_scheduler']}")

    for epoch in range(config['epochs']):
        train_loss            = train_one_epoch(model, train_loader, optimizer, criterion,
                                                device, config['use_physics'])
        val_logloss, val_acc  = evaluate(model, val_loader, device, config['use_physics'])
        is_best               = es.step(val_logloss)

        if is_best:
            best_loss = val_logloss
            torch.save(model.state_dict(), model_path)

        if scheduler:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        logs.append({'epoch': epoch+1, 'train_loss': round(train_loss, 6),
                     'val_logloss': round(val_logloss, 6), 'val_acc': round(val_acc, 4),
                     'lr': round(current_lr, 7)})

        print(f"Epoch {epoch+1:02d}/{config['epochs']} | "
              f"Train: {train_loss:.4f} | Val LogLoss: {val_logloss:.4f} | "
              f"Acc: {val_acc:.4f} | LR: {current_lr:.2e} {'✅' if is_best else ''}")

        if es.should_stop:
            print(f"Early Stopping at Epoch {epoch+1}")
            break

    log_df  = pd.DataFrame(logs)
    log_df.to_csv(f"{config['out_dir']}/logs/exp{exp_id}_log.csv", index=False)
    best_ep = int(log_df.loc[log_df['val_logloss'].idxmin(), 'epoch'])

    # Loss 곡선
    fig_dir = config.get('fig_dir', f"{config['out_dir']}/../reports/figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(log_df['epoch'], log_df['train_loss'],  color='blue',   label='Train Loss')
    axes[1].plot(log_df['epoch'], log_df['val_logloss'], color='orange', label='Val LogLoss')
    axes[1].plot(best_ep, log_df['val_logloss'].min(), 'ro', label='Best')
    axes[0].set_title(f'EXP-{exp_id} Train Loss')
    axes[1].set_title(f'EXP-{exp_id} Val LogLoss')
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/exp{exp_id}_loss_curve.png", dpi=150)
    plt.close()

    print(f"Best Val LogLoss: {best_loss:.4f} (Epoch {best_ep})")
    return {'exp_id': exp_id, 'best_val_logloss': best_loss,
            'best_epoch': best_ep, 'model_path': model_path}


# ================================================================
# run_inference
# ================================================================
def run_inference(config: dict, model_path: str, device=None) -> np.ndarray:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiViewNet(
        backbone_name   = config['backbone'],
        fusion_mode     = config['fusion_mode'],
        use_physics     = config['use_physics'],
        physics_dim     = config.get('physics_dim', 20),
        use_phys_mlp    = config.get('use_phys_mlp', False),
        shared_backbone = config.get('shared_backbone', False),
        img_size        = config['img_size'],
    ).to(device)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    test_loader = _build_test_loader(config)
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            # test 배치: (views, feats) — label 없음
            views, feats = batch
            front = views[0].to(device)
            top   = views[1].to(device)
            feats = feats.to(device)
            phys  = feats if config['use_physics'] else None
            probs = np.clip(
                torch.sigmoid(model(front, top, phys)).cpu().numpy().flatten(),
                1e-7, 1-1e-7
            )
            preds.extend(probs)

    return np.array(preds)
