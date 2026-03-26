"""
실험 공통 인프라 — experiment_utils.py

포함 내용:
- set_seed
- EarlyStopping
- train_one_epoch
- evaluate
- run_experiment (LR 스케줄러 포함)
- build_loaders

사용법:
    from experiment_utils import set_seed, EarlyStopping, run_experiment
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss, accuracy_score

from dataset import build_datasets, build_test_dataset
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
    """
    Val LogLoss 기준 Early Stopping
    patience 연속으로 개선 없으면 학습 중단
    """
    def __init__(self, patience: int = 5):
        self.patience    = patience
        self.counter     = 0
        self.best_loss   = float('inf')
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            return True   # is_best
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


# ================================================================
# 학습 / 평가 함수
# ================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, use_physics=False):
    model.train()
    total_loss = 0.0
    for batch in loader:
        front  = batch['front'].to(device)
        top    = batch['top'].to(device)
        labels = batch['label'].view(-1, 1).to(device)
        optimizer.zero_grad()
        logits = model(front, top, batch['physics'].to(device) if use_physics else None)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * front.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, use_physics=False):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            front  = batch['front'].to(device)
            top    = batch['top'].to(device)
            labels = batch['label'].cpu().numpy()
            logits = model(front, top, batch['physics'].to(device) if use_physics else None)
            probs  = np.clip(torch.sigmoid(logits).cpu().numpy().flatten(), 1e-7, 1-1e-7)
            preds.extend(probs)
            targets.extend(labels)
    preds, targets = np.array(preds), np.array(targets)
    return log_loss(targets, preds, labels=[0,1]), accuracy_score(targets, (preds>=0.5).astype(int))


# ================================================================
# run_experiment — LR 스케줄러 지원
# ================================================================
def run_experiment(config: dict, device=None):
    """
    config 딕셔너리 하나로 실험 전체 실행.

    추가된 config 키:
        lr_scheduler : 'cosine' | 'step' | None (기본 None)
        lr_min       : CosineAnnealingLR eta_min (기본 1e-5)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_id = config['exp_id']
    set_seed(config['random_state'])

    # config JSON 저장
    os.makedirs(f"{config['out_dir']}/logs", exist_ok=True)
    with open(f"{config['out_dir']}/logs/exp{exp_id}_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 데이터
    train_ds, val_ds = build_datasets(config)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'], shuffle=False,
                              num_workers=config['num_workers'], pin_memory=True)

    # 모델
    model = MultiViewNet(
        backbone_name   = config['backbone'],
        fusion_mode     = config['fusion_mode'],
        use_physics     = config['use_physics'],
        physics_dim     = config.get('physics_dim', 6),
        use_phys_mlp    = config.get('use_phys_mlp', False),
        shared_backbone = config.get('shared_backbone', False),
        img_size        = config['img_size'],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # LR 스케줄러
    scheduler = None
    lr_sched = config.get('lr_scheduler', None)
    if lr_sched == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = config['epochs'],
            eta_min = config.get('lr_min', 1e-5)
        )
    elif lr_sched == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.get('lr_step_size', 10),
            gamma     = config.get('lr_gamma', 0.5)
        )

    es   = EarlyStopping(patience=config['early_stopping_patience'])
    logs = []
    best_val_logloss = float('inf')
    os.makedirs(f"{config['out_dir']}/models", exist_ok=True)
    model_path = f"{config['out_dir']}/models/{config['model_name']}_v{config['model_version']}_best.pth"

    print(f"\n=== EXP-{exp_id} 시작 === Train: {len(train_ds)} | Val: {len(val_ds)}")
    if scheduler:
        print(f"LR Scheduler: {lr_sched}")

    for epoch in range(config['epochs']):
        train_loss              = train_one_epoch(model, train_loader, optimizer, criterion, device, config['use_physics'])
        val_logloss, val_acc    = evaluate(model, val_loader, device, config['use_physics'])
        is_best                 = es.step(val_logloss)

        if is_best:
            best_val_logloss = val_logloss
            torch.save(model.state_dict(), model_path)

        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = config['lr']

        logs.append({'epoch': epoch+1, 'train_loss': round(train_loss,6),
                     'val_logloss': round(val_logloss,6), 'val_acc': round(val_acc,4),
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
    axes[0].plot(log_df['epoch'], log_df['train_loss'], color='blue',   label='Train Loss')
    axes[1].plot(log_df['epoch'], log_df['val_logloss'], color='orange', label='Val LogLoss')
    axes[1].plot(best_ep, log_df['val_logloss'].min(), 'ro', label='Best')
    axes[0].set_title(f'EXP-{exp_id} Train Loss')
    axes[1].set_title(f'EXP-{exp_id} Val LogLoss')
    for ax in axes: ax.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/exp{exp_id}_loss_curve.png", dpi=150)
    plt.close()

    print(f"Best Val LogLoss: {best_val_logloss:.4f} (Epoch {best_ep})")
    return {'exp_id': exp_id, 'best_val_logloss': best_val_logloss,
            'best_epoch': best_ep, 'model_path': model_path}


# ================================================================
# 추론 헬퍼
# ================================================================
def run_inference(config: dict, model_path: str, device=None) -> np.ndarray:
    """
    저장된 모델 가중치로 test 추론 후 확률 배열 반환
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiViewNet(
        backbone_name   = config['backbone'],
        fusion_mode     = config['fusion_mode'],
        use_physics     = config['use_physics'],
        physics_dim     = config.get('physics_dim', 6),
        use_phys_mlp    = config.get('use_phys_mlp', False),
        shared_backbone = config.get('shared_backbone', False),
        img_size        = config['img_size'],
    ).to(device)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    test_ds     = build_test_dataset(config)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             num_workers=config['num_workers'], pin_memory=True)

    preds = []
    with torch.no_grad():
        for batch in test_loader:
            front  = batch['front'].to(device)
            top    = batch['top'].to(device)
            phys   = batch['physics'].to(device) if config['use_physics'] else None
            probs  = np.clip(torch.sigmoid(model(front, top, phys)).cpu().numpy().flatten(), 1e-7, 1-1e-7)
            preds.extend(probs)

    return np.array(preds)
