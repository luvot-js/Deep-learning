"""
train.py
--------
GraphoVision CNN 학습 루프

실행:
  python train.py

출력:
  - 매 epoch: train/val loss, val accuracy (8개 레이블 평균)
  - best_model.pth     : val_loss 기준 최적 모델
  - history.json       : 학습 히스토리 (evaluate.py에서 그래프용)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_pipeline import get_dataloaders
from model import GraphoVisionResNet


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────

BASE       = Path(__file__).parent
LINES_DIR  = str(BASE / "lines")
XML_DIR    = str(BASE / "xml")
LABEL_TXT  = str(BASE / "label_list.txt")

BATCH_SIZE    = 32
EPOCHS        = 50
LR_HEAD       = 1e-3     # 1단계: fc head만 학습할 때
LR_FINETUNE   = 1e-4     # 2단계: 전체 fine-tuning
FREEZE_EPOCHS = 10       # 백본을 고정하고 head만 학습하는 epoch 수
DROPOUT       = 0.3
PATIENCE      = 10       # Early stopping patience
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# 학습 / 검증 함수
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, threshold: float = 0.5):
    """
    val/test 루프. loss와 레이블별 accuracy를 반환한다.
    threshold: sigmoid 출력을 양성으로 판단하는 기준값 (기본 0.5)
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = (torch.sigmoid(logits) >= threshold).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds,  dim=0).numpy()   # (N, 8)
    all_labels = torch.cat(all_labels, dim=0).numpy()   # (N, 8)

    # 레이블별 정확도
    per_label_acc = (all_preds == all_labels).mean(axis=0)   # (8,)
    mean_acc = per_label_acc.mean()

    return total_loss / len(loader.dataset), mean_acc, per_label_acc


# ─────────────────────────────────────────────
# 클래스 불균형 대응: Focal Loss
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    멀티레이블 이진 분류용 Focal Loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha : (num_labels,) 레이블별 양성 클래스 가중치
            = neg_count / total  → 양성이 희귀할수록 높은 값
    gamma : focusing 파라미터. 클수록 easy example 억제 효과 강함 (기본 2.0)
    """
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits, targets: (N, L)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)                                          # sigmoid(logit)*y + (1-sigmoid)*( 1-y)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


def compute_alpha(train_loader, device, min_a: float = 0.5, max_a: float = 0.99):
    """
    Focal Loss의 per-label alpha 계산.
    alpha = neg_count / total  (양성이 드물수록 alpha 높음 → 양성에 집중)

    클램핑 범위 [min_a, max_a]:
      - 하한(0.5): alpha < 0.5면 음성이 더 희귀한 상황 — 그냥 0.5로 유지
      - 상한(0.99): 극단적 불균형에서도 수치 안정성 유지
    """
    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels)
    all_labels = torch.cat(all_labels, dim=0)       # (N, L)

    total     = all_labels.size(0)
    pos_count = all_labels.sum(dim=0)               # (L,)
    alpha     = (total - pos_count) / total         # neg_rate per label
    alpha     = alpha.clamp(min=min_a, max=max_a)
    print(f"Focal alpha (clamped to [{min_a}, {max_a}]): {alpha.tolist()}")
    return alpha.to(device)


# ─────────────────────────────────────────────
# 메인 학습 루프
# ─────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")

    # 데이터 로드
    train_loader, val_loader, _ = get_dataloaders(
        LINES_DIR, XML_DIR, LABEL_TXT, batch_size=BATCH_SIZE
    )

    # ── 1단계: 백본 고정, fc head만 학습 ──────────────────────────
    print(f"\n[1단계] 백본 고정 — fc head만 학습 ({FREEZE_EPOCHS} epoch)")
    model = GraphoVisionResNet(num_labels=5, dropout=DROPOUT, freeze_backbone=True).to(DEVICE)

    alpha     = compute_alpha(train_loader, DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=1e-4,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    history = {
        "train_loss": [],
        "val_loss":   [],
        "val_acc":    [],
        "per_label_acc": [],
    }

    best_val_loss    = float("inf")
    patience_counter = 0

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>8} | {'Time':>6} | Phase")
    print("-" * 68)

    for epoch in range(1, EPOCHS + 1):

        # ── 2단계 전환: FREEZE_EPOCHS 이후 전체 파라미터 학습 ──────
        if epoch == FREEZE_EPOCHS + 1:
            print(f"\n[2단계] 백본 고정 해제 — 전체 fine-tuning (lr={LR_FINETUNE})")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
            print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>8} | {'Time':>6} | Phase")
            print("-" * 68)

        phase = "freeze" if epoch <= FREEZE_EPOCHS else "finetune"
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, per_label_acc = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(float(val_acc))
        history["per_label_acc"].append(per_label_acc.tolist())

        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | {val_acc:>8.4f} | {elapsed:>5.1f}s | {phase}")

        # Best model 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BASE / "best_model.pth")
            print(f"         → best model 저장 (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
            break

    # 히스토리 저장
    with open(BASE / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n학습 완료!")
    print(f"  best_model.pth 저장됨 (val_loss={best_val_loss:.4f})")
    print(f"  history.json   저장됨")


if __name__ == "__main__":
    main()
