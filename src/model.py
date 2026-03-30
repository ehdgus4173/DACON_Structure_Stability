"""
MultiViewNet — 통합 모델 클래스

학술 근거:
🟢 Su et al.(2015) MVCNN — Multi-View Late Fusion
🟢 Guo et al.(2017) — FC 벡터 추출 구조 (SHAP 연결용)
🟡 Lerer et al.(2016) PhysNet — 물리 피처 결합
🟢 Dosovitskiy et al.(2021 ICLR) — ViT: 전역 구조 관계 포착

Args:
    backbone_name : 'resnet18' | 'efficientnet_b0' | 'efficientnet_b4' | 'vit_b_16'
    fusion_mode   : 'concat' | 'diff_concat'
    use_physics   : 물리 피처 FC 연결 여부
    physics_dim   : 물리 피처 차원 수
    use_phys_mlp  : True → 별도 MLP로 처리 후 concat (팀원 구조)
                    False → FC 입력에 직접 concat (기존 구조)
    shared_backbone: True → front/top 가중치 공유
    img_size      : 입력 이미지 크기
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B4_Weights,
    ViT_B_16_Weights,
)


# ================================================================
# ViT 인코더 래퍼
# ================================================================
class _ViTEncoder(nn.Module):
    """
    ViT-B/16 backbone 래퍼.
    heads를 Identity로 교체하고 768d CLS 토큰 벡터를 반환한다.

    🟢 Dosovitskiy et al.(2021 ICLR) ViT
    - CNN과 다른 inductive bias (전역 attention)
    - ViT+CNN 앙상블 시 다양성 극대화 효과
    """
    def __init__(self, vit):
        super().__init__()
        self.vit      = vit
        self.vit.heads = nn.Identity()   # classification head 제거 → 768d 출력

    def forward(self, x):
        # ViT forward → (B, 768) → (B, 768, 1, 1) for flatten(1) 호환
        out = self.vit(x)                 # (B, 768)
        return out.unsqueeze(-1).unsqueeze(-1)  # (B, 768, 1, 1)


# ================================================================
# MultiViewNet
# ================================================================
class MultiViewNet(nn.Module):
    def __init__(
        self,
        backbone_name   : str  = 'resnet18',
        fusion_mode     : str  = 'concat',
        use_physics     : bool = False,
        physics_dim     : int  = 6,
        use_phys_mlp    : bool = False,
        shared_backbone : bool = False,
        img_size        : int  = 224,
    ):
        super().__init__()
        self.fusion_mode      = fusion_mode
        self.use_physics      = use_physics
        self.use_phys_mlp     = use_phys_mlp
        self.shared_backbone  = shared_backbone

        # ── 백본 ─────────────────────────────────────────────
        self.front_encoder, feat_dim = self._build_backbone(backbone_name)
        if shared_backbone:
            self.top_encoder = self.front_encoder
        else:
            self.top_encoder, _ = self._build_backbone(backbone_name)
        self.feature_dim = feat_dim

        # ── Fusion 차원 계산 ─────────────────────────────────
        if fusion_mode == 'concat':
            fused_dim = feat_dim * 2
        elif fusion_mode == 'diff_concat':
            fused_dim = feat_dim * 3
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        # ── 물리 피처 처리 ───────────────────────────────────
        phys_out_dim = 0
        if use_physics:
            if use_phys_mlp:
                # 🟡 Lerer et al.(2016) PhysNet — 물리 피처 별도 MLP
                self.phys_mlp = nn.Sequential(
                    nn.Linear(physics_dim, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                )
                phys_out_dim = 32
            else:
                phys_out_dim = physics_dim

        fc_in = fused_dim + phys_out_dim

        # ── Classifier ───────────────────────────────────────
        # 🟢 Guo et al.(2017) — FC 직전 벡터 추출 구조 (SHAP 연결용)
        self.fc_hidden = nn.Sequential(
            nn.Linear(fc_in, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fc_out = nn.Linear(256, 1)

    # ── 백본 빌더 ─────────────────────────────────────────────
    def _build_backbone(self, name: str):
        if name == 'resnet18':
            net     = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            encoder = nn.Sequential(*list(net.children())[:-1])
            return encoder, 512

        elif name == 'efficientnet_b0':
            net     = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            encoder = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1))
            return encoder, 1280

        elif name == 'efficientnet_b4':
            net     = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
            encoder = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1))
            return encoder, 1792

        elif name == 'vit_b_16':
            # 🟢 Dosovitskiy et al.(2021 ICLR) ViT-B/16
            # 전역 self-attention으로 구조물 전체 형태 파악
            # CNN과 다른 inductive bias → 앙상블 다양성 극대화
            net     = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            encoder = _ViTEncoder(net)
            return encoder, 768

        else:
            raise ValueError(f"Unknown backbone: {name}")

    # ── SHAP 연결용 피처 추출 ─────────────────────────────────
    def extract_features(self, front, top, physics=None):
        f1 = self.front_encoder(front).flatten(1)
        f2 = self.top_encoder(top).flatten(1)

        if self.fusion_mode == 'concat':
            fused = torch.cat([f1, f2], dim=1)
        else:
            fused = torch.cat([f1, f2, f1 - f2], dim=1)

        if self.use_physics and physics is not None:
            p = self.phys_mlp(physics) if self.use_phys_mlp else physics
            fused = torch.cat([fused, p], dim=1)

        return fused

    def forward(self, front, top, physics=None):
        return self.fc_out(self.fc_hidden(self.extract_features(front, top, physics)))
