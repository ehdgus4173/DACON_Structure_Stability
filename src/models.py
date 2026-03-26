import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class MultiViewResNet(nn.Module):
    def __init__(self, num_classes: int = 1, num_phys_features: int = 10):
        super(MultiViewResNet, self).__init__()
        # Backbone으로 ResNet18 사용
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # 마지막 FC 레이어 제거하여 특성 추출기 역할
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # [NEW] 물리 피처용 특성 압축 브랜치
        self.phys_mlp = nn.Sequential(
            nn.Linear(num_phys_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 합친 후 통과시킬 최종 분류기 (CNN 512*2 + MLP 32 = 1056)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, views, phys_feats=None):
        # views: [batch, 3, 224, 224] 사이즈의 (front, top) 리스트
        f1 = self.feature_extractor(views[0]).view(views[0].size(0), -1)
        f2 = self.feature_extractor(views[1]).view(views[1].size(0), -1)
        img_feat = torch.cat((f1, f2), dim=1)  # (Batch, 1024)
        
        if phys_feats is not None:
            # 1D 물리 피처 브랜치 통과
            p = self.phys_mlp(phys_feats)     # (Batch, 32)
            # 이미지 특성과 Concat
            combined = torch.cat((img_feat, p), dim=1)  # (Batch, 1056)
        else:
            # 물리 피처가 안 들어왔을 때 에러나지 않게 제로 패딩
            zeros = torch.zeros(img_feat.size(0), 32).to(img_feat.device)
            combined = torch.cat((img_feat, zeros), dim=1)
            
        return self.classifier(combined)
