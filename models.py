import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class MultiViewResNet(nn.Module):
    def __init__(self, num_classes: int = 1):
        super(MultiViewResNet, self).__init__()
        # Backbone으로 ResNet18 사용
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # 마지막 FC 레이어 제거하여 특성 추출기 역할
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 특성 합친 후 통과시킬 분류기 
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, views):
        # views: [batch, 3, 224, 224] 사이즈 텐서 배열 (front, top의 2개)
        f1 = self.feature_extractor(views[0]).view(views[0].size(0), -1)
        f2 = self.feature_extractor(views[1]).view(views[1].size(0), -1)
        
        # 특성 합침 (512 + 512 = 1024)
        combined = torch.cat((f1, f2), dim=1)
        
        # 최종 로짓 반환
        return self.classifier(combined)
