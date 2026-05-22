"""
model.py
--------
GraphoVision CNN 모델 정의

구조:
  4-Layer Custom CNN
  → Global Average Pooling (GAP)
  → Linear(256, 8)  ← 8가지 심리 지표 출력 (sigmoid 없음, BCEWithLogitsLoss 사용)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU 묶음 블록"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GraphoVisionCNN(nn.Module):
    """
    4-Layer CNN + Global Average Pooling + 8-class 출력

    입력: (B, 1, 224, 224)  — 그레이스케일 필기 이미지
    출력: (B, 8)            — 8가지 심리 지표 로짓

    레이어 구조:
      Conv1: 1  → 32,  MaxPool → (B, 32, 112, 112)
      Conv2: 32 → 64,  MaxPool → (B, 64,  56,  56)
      Conv3: 64 → 128, MaxPool → (B, 128, 28,  28)
      Conv4: 128→ 256, no pool → (B, 256, 28,  28)
      GAP                      → (B, 256, 1,   1)
      Flatten                  → (B, 256)
      Linear(256, 8)           → (B, 8)
    """

    def __init__(self, num_labels: int = 8, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1,   32,  kernel=3, pool=True),   # 224 → 112
            ConvBlock(32,  64,  kernel=3, pool=True),   # 112 → 56
            ConvBlock(64,  128, kernel=3, pool=True),   #  56 → 28
            ConvBlock(128, 256, kernel=3, pool=False),  #  28 → 28 (pool 없음)
        )

        # Global Average Pooling: 공간 차원 전체 평균 → (B, 256, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming He 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)    # (B, 256, 28, 28)
        x = self.gap(x)         # (B, 256, 1, 1)
        x = x.flatten(1)        # (B, 256)
        x = self.classifier(x)  # (B, 8)
        return x


# ─────────────────────────────────────────────
# 단독 실행 시: 모델 구조 확인
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = GraphoVisionCNN()
    print(model)

    dummy = torch.randn(4, 1, 224, 224)
    out = model(dummy)
    print(f"\n입력 shape : {dummy.shape}")
    print(f"출력 shape : {out.shape}")   # (4, 8)

    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n전체 파라미터   : {total_params:,}")
    print(f"학습 가능 파라미터: {trainable:,}")
