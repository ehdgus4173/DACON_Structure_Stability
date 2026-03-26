import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class MultiViewResNet(nn.Module):
    def __init__(self, num_classes: int = 1, num_phys_features: int = 10):
        super().__init__()

        # ResNet18 backbone (ImageNet pretrained), FC 레이어 제거
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # 물리 피처 압축 브랜치
        self.phys_mlp = nn.Sequential(
            nn.Linear(num_phys_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),        # 0.3 → 0.4
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # 최종 분류기: CNN(512*2) + MLP(32) = 1056
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.4),        # 0.2 → 0.4
            nn.Linear(256, num_classes),
        )

    def forward(self, views, phys_feats=None):
        """
        Args:
            views:      [front_tensor, top_tensor]  shape: (B, 3, H, W)
            phys_feats: 물리 피처 텐서              shape: (B, num_phys_features)
        """
        f1 = self.feature_extractor(views[0]).view(views[0].size(0), -1)  # (B, 512)
        f2 = self.feature_extractor(views[1]).view(views[1].size(0), -1)  # (B, 512)
        img_feat = torch.cat((f1, f2), dim=1)                              # (B, 1024)

        if phys_feats is not None:
            p        = self.phys_mlp(phys_feats)                           # (B, 32)
            combined = torch.cat((img_feat, p), dim=1)                     # (B, 1056)
        else:
            zeros    = torch.zeros(img_feat.size(0), 32, device=img_feat.device)
            combined = torch.cat((img_feat, zeros), dim=1)

        return self.classifier(combined)
