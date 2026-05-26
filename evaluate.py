"""
evaluate.py
-----------
학습된 모델 평가 및 중간발표용 시각화 생성

실행:
  python evaluate.py

출력 파일:
  - training_curves.png     : Train/Val Loss 학습 곡선
  - per_label_metrics.png   : 8개 지표별 Accuracy/F1 막대 그래프
  - radar_chart_sample.png  : Ground Truth vs AI Prediction 레이더 차트 (예시)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # 헤드리스 환경 대응 (GUI 없어도 PNG 저장 가능)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score, accuracy_score

import torch

from data_pipeline import get_dataloaders
from model import GraphoVisionResNet


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────

BASE      = Path(__file__).parent
LINES_DIR = str(BASE / "lines")
XML_DIR   = str(BASE / "xml")
LABEL_TXT = str(BASE / "label_list.txt")
MODEL_PTH = BASE / "best_model.pth"
HIST_JSON = BASE / "history.json"

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# 8가지 심리 지표 이름 (HBPA 논문 기반 — 실제 명칭 확인 후 수정)
TRAIT_NAMES = [
    "Emotional\nStability",
    "Social\nBehavior",
    "Mental\nEnergy",
    "Willpower",
    "Imagination",
    "Fear /\nAnxiety",
    "Introversion\n/Extroversion",
    "Sensitivity",
]


# ─────────────────────────────────────────────
# 1. 학습 곡선 (Train / Val Loss)
# ─────────────────────────────────────────────

def plot_training_curves(history_path: str = str(HIST_JSON), save_path: str = None):
    """Train Loss / Val Loss를 epochs 축으로 그린다."""
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("GraphoVision — Training Curves", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#F44336")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Validation Accuracy
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="#4CAF50")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (mean over 8 labels)")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = save_path or str(BASE / "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_training_curves] 저장: {out}")


# ─────────────────────────────────────────────
# 2. 8개 지표별 Accuracy / F1 막대 그래프
# ─────────────────────────────────────────────

def print_per_label_metrics(model, test_loader, device=DEVICE, save_path: str = None):
    """
    test_loader 전체에 대해 8개 레이블 각각의
    Accuracy, Precision, Recall, F1-Score를 출력하고 그래프로 저장한다.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = (torch.sigmoid(logits) >= 0.5).float().cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    preds_np  = torch.cat(all_preds,  dim=0).numpy()    # (N, 8)
    labels_np = torch.cat(all_labels, dim=0).numpy()    # (N, 8)

    short_names = [
        "Emot.", "Social", "Energy", "Will",
        "Imag.", "Fear", "Intro.", "Sensit."
    ]

    print(f"\n{'Label':<20} {'Accuracy':>9} {'F1-Score':>9}")
    print("-" * 40)
    accs, f1s = [], []
    for i in range(8):
        acc = accuracy_score(labels_np[:, i], preds_np[:, i])
        f1  = f1_score(labels_np[:, i], preds_np[:, i], zero_division=0)
        accs.append(acc)
        f1s.append(f1)
        print(f"  label_{i} ({short_names[i]:<8})  {acc:>8.4f}   {f1:>8.4f}")
    print(f"\n  {'Mean':<18}  {np.mean(accs):>8.4f}   {np.mean(f1s):>8.4f}")

    # 막대 그래프
    x = np.arange(8)
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, accs, width, label="Accuracy", color="#42A5F5", alpha=0.85)
    ax.bar(x + width/2, f1s,  width, label="F1-Score",  color="#EF5350", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("GraphoVision — Per-Label Accuracy & F1-Score (Test Set)", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = save_path or str(BASE / "per_label_metrics.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[print_per_label_metrics] 저장: {out}")

    return accs, f1s


# ─────────────────────────────────────────────
# 3. 레이더 차트 (중간발표 핵심 시각화)
# ─────────────────────────────────────────────

def plot_radar_chart(
    ground_truth: np.ndarray,
    predicted:    np.ndarray,
    title:        str = "GraphoVision — Personality Prediction",
    save_path:    str = None,
):
    """
    8각 레이더 차트: Ground Truth(파란 실선) vs AI Prediction(빨간 점선)

    Args:
        ground_truth: (8,) 실제 레이블 (0 or 1)
        predicted:    (8,) AI 예측 확률 (0.0 ~ 1.0)
        title:        차트 제목
        save_path:    저장 경로 (None이면 radar_chart_sample.png)
    """
    N = len(TRAIT_NAMES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # 닫힌 다각형

    gt   = ground_truth.tolist() + ground_truth[:1].tolist()
    pred = predicted.tolist()    + predicted[:1].tolist()

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F0F4F8")

    # 배경 격자
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="gray")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(TRAIT_NAMES, fontsize=9, fontweight="bold")

    # Ground Truth
    ax.plot(angles, gt, "o-", linewidth=2.5, color="#1565C0", label="Ground Truth")
    ax.fill(angles, gt, alpha=0.15, color="#1565C0")

    # AI Prediction
    ax.plot(angles, pred, "s--", linewidth=2.5, color="#C62828", label="AI Prediction")
    ax.fill(angles, pred, alpha=0.10, color="#C62828")

    ax.set_title(title, size=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=10)

    out = save_path or str(BASE / "radar_chart_sample.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot_radar_chart] 저장: {out}")


def plot_radar_batch(model, test_loader, n_samples: int = 4, device=DEVICE):
    """
    테스트셋에서 n_samples개를 뽑아 레이더 차트를 한 figure에 그린다.
    (발표 PPT에 여러 사례를 한 장으로 보여주기 위함)
    """
    model.eval()
    all_imgs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs_dev = imgs.to(device)
            logits = model(imgs_dev)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_imgs.append(imgs.numpy())
            all_preds.append(probs)
            all_labels.append(labels.numpy())
            if sum(len(x) for x in all_imgs) >= n_samples:
                break

    all_imgs   = np.concatenate(all_imgs,   axis=0)[:n_samples]
    all_preds  = np.concatenate(all_preds,  axis=0)[:n_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:n_samples]

    N = len(TRAIT_NAMES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    cols = min(n_samples, 2)
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows),
                             subplot_kw={"polar": True})
    fig.suptitle("GraphoVision — Ground Truth vs AI Prediction", fontsize=15, fontweight="bold")
    axes = np.array(axes).flatten()

    for i in range(n_samples):
        ax = axes[i]
        gt   = all_labels[i].tolist() + all_labels[i][:1].tolist()
        pred = all_preds[i].tolist()  + all_preds[i][:1].tolist()

        ax.set_ylim(0, 1)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(["0.5", "1.0"], fontsize=7, color="gray")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(TRAIT_NAMES, fontsize=8)

        ax.plot(angles, gt,   "o-",  lw=2.2, color="#1565C0", label="GT")
        ax.fill(angles, gt,   alpha=0.15, color="#1565C0")
        ax.plot(angles, pred, "s--", lw=2.2, color="#C62828", label="Pred")
        ax.fill(angles, pred, alpha=0.10, color="#C62828")
        ax.set_title(f"Sample {i + 1}", fontsize=10, pad=12)

    gt_patch   = mpatches.Patch(color="#1565C0", alpha=0.7, label="Ground Truth")
    pred_patch = mpatches.Patch(color="#C62828", alpha=0.7, label="AI Prediction")
    fig.legend(handles=[gt_patch, pred_patch], loc="lower center",
               ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.02))

    # 남는 subplot 숨기기
    for j in range(n_samples, len(axes)):
        axes[j].set_visible(False)

    out = str(BASE / "radar_chart_batch.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot_radar_batch] 저장: {out}")


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────

def main():
    # ① 학습 곡선 (history.json이 있어야 실행 가능)
    if HIST_JSON.exists():
        plot_training_curves()
    else:
        print("[SKIP] history.json 없음 — train.py를 먼저 실행하세요")

    # ② 모델 로드
    if not MODEL_PTH.exists():
        print("[ERROR] best_model.pth 없음 — train.py를 먼저 실행하세요")
        return

    model = GraphoVisionResNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PTH, map_location=DEVICE))
    print(f"모델 로드 완료: {MODEL_PTH}")

    # ③ 데이터 로드 (test set만 필요)
    _, _, test_loader = get_dataloaders(
        LINES_DIR, XML_DIR, LABEL_TXT, batch_size=BATCH_SIZE
    )

    # ④ 8개 지표별 Accuracy / F1
    print_per_label_metrics(model, test_loader)

    # ⑤ 레이더 차트 — 단일 샘플 예시
    model.eval()
    imgs, labels = next(iter(test_loader))
    with torch.no_grad():
        probs = torch.sigmoid(model(imgs.to(DEVICE))).cpu().numpy()

    plot_radar_chart(
        ground_truth=labels[0].numpy(),
        predicted=probs[0],
        title=f"GraphoVision — Sample Prediction",
    )

    # ⑥ 레이더 차트 — 4개 샘플 배치 (PPT 핵심 자료)
    plot_radar_batch(model, test_loader, n_samples=4)

    print("\n평가 완료! 생성된 파일:")
    for f in ["training_curves.png", "per_label_metrics.png",
              "radar_chart_sample.png", "radar_chart_batch.png"]:
        path = BASE / f
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {f}")


if __name__ == "__main__":
    main()
