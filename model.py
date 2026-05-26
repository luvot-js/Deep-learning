"""
model.py
--------
GraphoVision ResNet18 기반 모델 정의

구조:
  ResNet18 (ImageNet pretrained, conv1 1채널로 수정)
  → AdaptiveAvgPool2d (ResNet 내장)
  → Dropout + Linear(512, 8)  ← 8가지 심리 지표 출력 (BCEWithLogitsLoss 사용)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class GraphoVisionResNet(nn.Module):
    """
    ResNet18 기반 전이 학습 모델

    입력: (B, 1, 224, 224)  — 그레이스케일 필기 이미지
    출력: (B, 8)            — 8가지 심리 지표 로짓

    전략:
      - conv1: 3채널 → 1채널로 수정 (pretrained 가중치 채널 평균으로 초기화)
      - 마지막 fc: 1000 → 8 출력으로 교체
      - freeze_backbone=True 시 layer1~layer2 고정, layer3~layer4+fc만 학습
    """

    def __init__(self, num_labels: int = 8, dropout: float = 0.3, freeze_backbone: bool = False):
        super().__init__()

        # Pretrained ResNet18 로드
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # ── conv1: 3채널 → 1채널 수정 ──────────────────────────────
        old_conv1 = backbone.conv1  # Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )
        # pretrained 가중치 3채널 평균 → 1채널로 초기화 (정보 손실 최소화)
        new_conv1.weight.data = old_conv1.weight.data.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv1

        # ── fc: 1000 → num_labels 교체 ────────────────────────────
        in_features = backbone.fc.in_features  # 512
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_labels),
        )

        self.model = backbone

        # ── 선택적 백본 고정 ───────────────────────────────────────
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                # layer3, layer4, fc만 학습
                if not any(name.startswith(p) for p in ("layer3", "layer4", "fc")):
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────
# 단독 실행 시: 모델 구조 확인
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = GraphoVisionResNet()

    dummy = torch.randn(4, 1, 224, 224)
    out = model(dummy)
    print(f"입력 shape : {dummy.shape}")
    print(f"출력 shape : {out.shape}")   # (4, 8)

    total_params   = sum(p.numel() for p in model.parameters())
    trainable      = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n전체 파라미터   : {total_params:,}")
    print(f"학습 가능 파라미터: {trainable:,}")
