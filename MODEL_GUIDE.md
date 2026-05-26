# GraphoVision — 지금부터 할 일

---

## 현재 상태

| 파일 | 상태 |
|------|------|
| `model.py` | ResNet18 기반으로 교체 완료 |
| `train.py` | 2단계 학습 전략 적용 완료 |
| `data_pipeline.py` | Augmentation 수정 완료 (RandomHorizontalFlip 제거) |
| `evaluate.py` | GraphoVisionResNet으로 import 수정 완료 |

---

## Step 1. 학습 실행

```bash
python train.py
```

콘솔에 아래 형식으로 출력된다.

```
[1단계] 백본 고정 — fc head만 학습 (5 epoch)

 Epoch | Train Loss |   Val Loss |  Val Acc |  Time | Phase
--------------------------------------------------------------------
     1 |     0.6821 |     0.6543 |   0.6120 |  12.3s | freeze
     ...
     5 |     0.5102 |     0.5230 |   0.7011 |  12.1s | freeze

[2단계] 백본 고정 해제 — 전체 fine-tuning (lr=0.0001)

     6 |     0.4890 |     0.4701 |   0.7230 |  18.5s | finetune
     ...
```

학습이 끝나면 `best_model.pth`와 `history.json`이 생성된다.

---

## Step 2. 학습 곡선으로 상태 진단

```bash
python evaluate.py
```

`training_curves.png`가 생성된다. 아래 표로 상태를 판단한다.

| 학습 곡선 패턴 | 판단 | 조처 |
|---------------|------|------|
| train↓ val↓ 같이 내려감 | 정상 수렴 | 그대로 진행 |
| train↓ val↑ 벌어짐 | 과적합 | `train.py`에서 `DROPOUT` 0.3 → 0.5, `weight_decay` 1e-4 → 1e-3 |
| train↓ val 거의 안 변함 | 과소적합 | `FREEZE_EPOCHS` 줄이거나 0으로 설정 |
| val_acc가 항상 동일한 값 | 전부 0 예측 | 아래 **pos_weight 확인** 참고 |
| loss가 NaN | LR 너무 높음 | `LR_FINETUNE` 1e-4 → 1e-5 |

---

## Step 3. pos_weight 확인 (학습이 이상할 때)

`train.py`의 `main()` 첫 줄에 아래를 임시로 추가해서 출력해 본다.

```python
pos_weight = compute_pos_weight(train_loader, DEVICE)
print("pos_weight:", pos_weight)
```

출력값 해석:

```
pos_weight: tensor([1.2, 8.5, 0.9, 14.2, 3.1, 22.0, 1.8, 6.3])
```

- **값이 10 이상**인 레이블 → 양성(1) 샘플이 극히 적다는 뜻
- 이 경우 해당 레이블의 예측 자체가 의미 없을 수 있으므로, 레이블 분포를 다시 확인해야 한다

---

## Step 4. 평가 결과 해석

```bash
python evaluate.py
```

터미널에 출력되는 내용:

```
Label                Accuracy  F1-Score
----------------------------------------
  label_0 (Emot.  )    0.7412     0.6831
  label_1 (Social )    0.6203     0.5120
  ...
  Mean                  0.6900     0.6100
```

생성되는 파일:

| 파일 | 내용 |
|------|------|
| `training_curves.png` | Train/Val Loss 학습 곡선 |
| `per_label_metrics.png` | 8개 지표별 Accuracy·F1 막대 그래프 |
| `radar_chart_sample.png` | 단일 샘플 레이더 차트 |
| `radar_chart_batch.png` | 4개 샘플 레이더 차트 (발표용) |

---

## Step 5. 결과에 따른 튜닝 선택지

### 아직 성능이 낮으면

**옵션 A — FREEZE_EPOCHS 조정** (`train.py`)

```python
FREEZE_EPOCHS = 5   # 기본값
# 1단계에서 val_loss가 빨리 수렴 → 줄이기 (3)
# 아직 head가 덜 학습된 것 같으면 → 늘리기 (10)
```

**옵션 B — Label Smoothing 적용** (`train.py`)

레이블 자체에 전문가 판단 오류가 섞여 있을 수 있다. 모델이 너무 확신하지 않도록 한다.

```python
# train_one_epoch 안에서 loss 계산 전에 추가
labels = labels * 0.9 + 0.05   # 0 → 0.05, 1 → 0.95
loss = criterion(logits, labels)
```

**옵션 C — 학습률 세분화** (`train.py`)

백본과 head에 서로 다른 LR을 주는 방식. 현재보다 정밀하게 제어할 수 있다.

```python
optimizer = Adam([
    {"params": model.model.layer4.parameters(), "lr": 1e-4},
    {"params": model.model.fc.parameters(),     "lr": 1e-3},
], weight_decay=1e-4)
```

### 성능이 충분하면

`per_label_metrics.png`와 `radar_chart_batch.png`를 발표 자료로 사용한다.
