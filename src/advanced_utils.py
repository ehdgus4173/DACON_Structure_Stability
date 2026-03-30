import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiment_utils import (
    set_seed, EarlyStopping, _build_loaders, _build_test_loader,
    run_experiment, run_inference, train_one_epoch, evaluate
)
from model import MultiViewNet



# ================================================================
# TTA (Test-Time Augmentation)
# ================================================================
# 🟢 Simonyan & Zisserman(2015) VGGNet — horizontal flip TTA로 ILSVRC 개선
# 🟢 Son & Kang(2023, Information Sciences) — 추가 학습 없이 분류 정확도 향상

def run_inference_tta(config: dict, model_path: str, device=None,
                      n_augments: int = 5) -> np.ndarray:
    """
    TTA 추론: 원본 + 4가지 변환의 확률값을 평균낸다.

    n_augments=5: 원본, hflip, vflip, rot90, rot180
    n_augments=1: 원본만 (TTA 비활성화, 일반 추론과 동일)
    """
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

    # TTA 변환 목록 (결정론적 — 랜덤 아님)
    tta_fns = [
        lambda x: x,                           # 원본
        lambda x: x.flip(-1),                  # horizontal flip
        lambda x: x.flip(-2),                  # vertical flip
        lambda x: x.rot90(1, dims=[-2, -1]),   # 90도 회전
        lambda x: x.rot90(2, dims=[-2, -1]),   # 180도 회전
    ][:n_augments]

    test_loader = _build_test_loader(config)
    all_preds = []

    with torch.no_grad():
        for aug_fn in tta_fns:
            aug_preds = []
            for batch in test_loader:
                views, feats = batch
                front = aug_fn(views[0].to(device))
                top   = aug_fn(views[1].to(device))
                feats = feats.to(device)
                phys  = feats if config['use_physics'] else None
                probs = np.clip(
                    torch.sigmoid(model(front, top, phys)).cpu().numpy().flatten(),
                    1e-7, 1-1e-7
                )
                aug_preds.extend(probs)
            all_preds.append(np.array(aug_preds))

    return np.mean(all_preds, axis=0)


# ================================================================
# SWA (Stochastic Weight Averaging)
# ================================================================
# 🟢 Izmailov et al.(2018 UAI) — 후반 checkpoint 평균이 더 넓은 loss basin
# 학습 중 SWA 적용 버전 run_experiment_swa

def run_experiment_swa(config: dict, device=None, swa_start_ratio: float = 0.75):
    """
    SWA 적용 실험.
    전체 epoch의 swa_start_ratio 이후부터 AveragedModel에 가중치를 누적한다.

    🟢 Izmailov et al.(2018 UAI):
    "Averaging Weights Leads to Wider Optima and Better Generalization"
    """
    from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_id = config['exp_id']
    set_seed(config['random_state'])

    os.makedirs(f"{config['out_dir']}/logs",   exist_ok=True)
    os.makedirs(f"{config['out_dir']}/models", exist_ok=True)

    train_loader, val_loader = _build_loaders(config, device)

    model = MultiViewNet(
        backbone_name   = config['backbone'],
        fusion_mode     = config['fusion_mode'],
        use_physics     = config['use_physics'],
        physics_dim     = config.get('physics_dim', 20),
        use_phys_mlp    = config.get('use_phys_mlp', False),
        img_size        = config['img_size'],
    ).to(device)

    swa_model = AveragedModel(model)
    criterion  = nn.BCEWithLogitsLoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=config['lr'])
    swa_start  = int(config['epochs'] * swa_start_ratio)

    es   = EarlyStopping(patience=config['early_stopping_patience'])
    logs = []
    best_loss  = float('inf')
    model_path = f"{config['out_dir']}/models/{config['model_name']}_v{config['model_version']}_swa_best.pth"

    print(f"\n=== EXP-{exp_id} (SWA) === Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    print(f"SWA 시작 Epoch: {swa_start+1}")

    for epoch in range(config['epochs']):
        train_loss           = train_one_epoch(model, train_loader, optimizer, criterion,
                                               device, config['use_physics'])
        val_logloss, val_acc = evaluate(model, val_loader, device, config['use_physics'])
        is_best              = es.step(val_logloss)

        if epoch >= swa_start:
            swa_model.update_parameters(model)

        if is_best:
            best_loss = val_logloss
            torch.save(model.state_dict(), model_path)

        logs.append({'epoch': epoch+1, 'train_loss': round(train_loss, 6),
                     'val_logloss': round(val_logloss, 6), 'val_acc': round(val_acc, 4)})
        print(f"Epoch {epoch+1:02d}/{config['epochs']} | Train: {train_loss:.4f} | "
              f"Val: {val_logloss:.4f} | Acc: {val_acc:.4f} {'[SWA]' if epoch >= swa_start else ''} {'✅' if is_best else ''}")

        if es.should_stop:
            print(f"Early Stopping at Epoch {epoch+1}")
            break

    # SWA BN 업데이트
    update_bn(train_loader, swa_model, device=device)
    swa_path = model_path.replace('_best.pth', '_swa_final.pth')
    torch.save(swa_model.module.state_dict(), swa_path)

    # SWA 모델 Val 평가
    swa_val, swa_acc = evaluate(swa_model.module, val_loader, device, config['use_physics'])
    print(f"\nSWA Val LogLoss: {swa_val:.4f} | Acc: {swa_acc:.4f}")
    print(f"Best single Val LogLoss: {best_loss:.4f}")

    best_ep = int(pd.DataFrame(logs).loc[pd.DataFrame(logs)['val_logloss'].idxmin(), 'epoch'])
    final_loss = min(best_loss, swa_val)
    final_path = swa_path if swa_val < best_loss else model_path

    return {'exp_id': exp_id, 'best_val_logloss': final_loss,
            'best_epoch': best_ep, 'model_path': final_path,
            'swa_val_logloss': swa_val}


# ================================================================
# Label Smoothing
# ================================================================
# 🟢 Müller et al.(2019 NeurIPS) "When Does Label Smoothing Help?"
# LogLoss 평가 지표에서 과신뢰(overconfidence) 감소 효과

class LabelSmoothingBCE(nn.Module):
    """
    Binary Cross-Entropy with Label Smoothing.
    hard label 0/1 → soft label epsilon/(epsilon+1)
    🟢 Müller et al.(2019 NeurIPS)
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: (B, 1) float
        smooth_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits, smooth_targets)


# ================================================================
# Pseudo-Label
# ================================================================
# 🟢 Lee(2013) "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method"
# 타겟 도메인(Test) 데이터를 학습에 활용 → Domain Adaptation 효과

def generate_pseudo_labels(config: dict, model_path: str,
                            confidence_threshold: float = 0.95,
                            device=None) -> pd.DataFrame:
    """
    Test 셋에 대해 pseudo-label을 생성한다.
    confidence_threshold 이상의 확률값을 가진 샘플만 반환.

    🟢 Lee(2013):
    높은 신뢰도 샘플만 선택하면 노이즈 pseudo-label의 영향을 최소화할 수 있다.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    preds = run_inference(config, model_path, device)

    data_dir = config['data_dir']
    test_dir = os.path.join(data_dir, 'test')
    test_ids = sorted([d for d in os.listdir(test_dir)
                       if os.path.isdir(os.path.join(test_dir, d))])

    df = pd.DataFrame({'id': test_ids, 'unstable_prob': preds})
    df['label'] = np.where(df['unstable_prob'] >= 0.5, 'unstable', 'stable')

    # 신뢰도 필터
    high_conf = df[
        (df['unstable_prob'] >= confidence_threshold) |
        (df['unstable_prob'] <= 1 - confidence_threshold)
    ].copy()

    print(f"Pseudo-label 생성: 전체 {len(df)}개 중 {len(high_conf)}개 선택 "
          f"(신뢰도 >= {confidence_threshold})")
    print(f"  - unstable: {(high_conf['label']=='unstable').sum()}개")
    print(f"  - stable:   {(high_conf['label']=='stable').sum()}개")

    return high_conf[['id', 'label']]


def run_experiment_with_pseudo_labels(config: dict,
                                       pseudo_df: pd.DataFrame,
                                       device=None):
    """
    Pseudo-label을 Train에 추가해서 재학습한다.
    pseudo_df의 이미지 경로는 config['data_dir']/test/{id}/front.png 로 찾는다.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pseudo_df에 test 경로 정보 추가
    data_dir = config['data_dir']

    # 기존 train+dev 로더 (dev_split_ratio 적용)
    # _build_loaders 내부에서 train_df를 만들 때 pseudo_df를 추가로 넣는 방식
    # config에 pseudo_df 경로 정보를 임시로 주입

    # pseudo CSV를 임시 파일로 저장
    pseudo_path = os.path.join(config['out_dir'], 'pseudo_labels.csv')
    pseudo_df.to_csv(pseudo_path, index=False)
    config_pl = {**config, 'pseudo_csv': pseudo_path}

    return run_experiment(config_pl, device)
