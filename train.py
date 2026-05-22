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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_pipeline import get_dataloaders
from model import GraphoVisionCNN


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────

BASE       = Path(__file__).parent
LINES_DIR  = str(BASE / "lines")
XML_DIR    = str(BASE / "xml")
LABEL_TXT  = str(BASE / "label_list.txt")

BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-3
DROPOUT    = 0.3
PATIENCE   = 10          # Early stopping patience
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


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


def evaluate(model, loader, criterion, device):
    """
    val/test 루프. loss와 레이블별 accuracy를 반환한다.
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

            preds = (torch.sigmoid(logits) >= 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds,  dim=0).numpy()   # (N, 8)
    all_labels = torch.cat(all_labels, dim=0).numpy()   # (N, 8)

    # 레이블별 정확도
    per_label_acc = (all_preds == all_labels).mean(axis=0)   # (8,)
    mean_acc = per_label_acc.mean()

    return total_loss / len(loader.dataset), mean_acc, per_label_acc


# ─────────────────────────────────────────────
# 클래스 불균형 대응: pos_weight 계산
# ─────────────────────────────────────────────

def compute_pos_weight(train_loader, device):
    """
    BCEWithLogitsLoss의 pos_weight 계산.
    양성(1) 비율이 낮은 레이블에 더 높은 가중치를 부여한다.
    """
    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels)
    all_labels = torch.cat(all_labels, dim=0)   # (N, 8)

    pos_count = all_labels.sum(dim=0)           # (8,)
    neg_count = all_labels.size(0) - pos_count
    pos_weight = neg_count / (pos_count + 1e-6)
    return pos_weight.to(device)


# ─────────────────────────────────────────────
# 메인 학습 루프
# ─────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")

    # 데이터 로드
    train_loader, val_loader, _ = get_dataloaders(
        LINES_DIR, XML_DIR, LABEL_TXT, batch_size=BATCH_SIZE
    )

    # 모델
    model = GraphoVisionCNN(dropout=DROPOUT).to(DEVICE)

    # Loss: pos_weight로 클래스 불균형 보정
    pos_weight = compute_pos_weight(train_loader, DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer & Scheduler
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=True)

    # 학습 히스토리
    history = {
        "train_loss": [],
        "val_loss":   [],
        "val_acc":    [],
        "per_label_acc": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>8} | {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, per_label_acc = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(float(val_acc))
        history["per_label_acc"].append(per_label_acc.tolist())

        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | {val_acc:>8.4f} | {elapsed:>5.1f}s")

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
