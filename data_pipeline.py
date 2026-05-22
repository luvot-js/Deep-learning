"""
data_pipeline.py
----------------
label_list.txt 파싱 → XML 메타데이터 조인 → lines/ 이미지 경로 매핑
→ PyTorch Dataset / DataLoader 구축
"""

import os
import glob
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ─────────────────────────────────────────────
# 1. label_list.txt 파싱
# ─────────────────────────────────────────────

def parse_label_list(txt_path: str) -> pd.DataFrame:
    """
    label_list.txt 파일을 읽어 필요한 열만 추출한다.

    열 구조 (0-indexed):
      0~6  : 물리적 특징 (사용 안 함)
      7~14 : 8가지 심리 지표 (0 or 1) ← 타겟 레이블
      15   : 파일명 ex) "002-0.png"  ← writer_id + form_index

    반환: DataFrame with columns
      writer_id (str, zero-padded 3자리), form_index (int),
      label_0 ~ label_7 (int)
    """
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 16:
                continue

            filename = parts[15]                        # ex) "002-0.png"
            stem = filename.replace(".png", "")         # ex) "002-0"
            dash_idx = stem.rfind("-")
            suffix = stem[dash_idx + 1:]
            if not suffix.isdigit():   # "598-1-" 같은 비정상 항목 스킵
                continue
            writer_id = stem[:dash_idx].zfill(3)        # "002"
            form_index = int(suffix)                    # 0

            labels = [int(parts[i]) for i in range(7, 15)]  # 8개

            rows.append({
                "writer_id": writer_id,
                "form_index": form_index,
                **{f"label_{j}": labels[j] for j in range(8)},
            })

    df = pd.DataFrame(rows)
    print(f"[parse_label_list] {len(df)}행 로드 완료")
    return df


# ─────────────────────────────────────────────
# 2. XML 파싱 → writer_id → form_id 목록 매핑
# ─────────────────────────────────────────────

def build_writer_form_map(xml_dir: str) -> dict:
    """
    xml/ 디렉터리의 모든 XML 파일을 파싱하여
    writer-id → 정렬된 form_id 목록을 반환한다.

    예) {"002": ["a01-003", "b02-015", ...], ...}
    """
    xml_files = sorted(glob.glob(os.path.join(xml_dir, "*.xml")))
    writer_form_map = defaultdict(list)

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            writer_id = root.get("writer-id", "").zfill(3)
            form_id = root.get("id", "")
            if writer_id and form_id:
                writer_form_map[writer_id].append(form_id)
        except ET.ParseError:
            continue

    # 각 writer의 form 목록을 알파벳순 정렬 (form_index 기준이 됨)
    for wid in writer_form_map:
        writer_form_map[wid] = sorted(writer_form_map[wid])

    print(f"[build_writer_form_map] writer 수: {len(writer_form_map)}")
    return dict(writer_form_map)


# ─────────────────────────────────────────────
# 3. 이미지 경로 + 레이블 샘플 목록 구축
# ─────────────────────────────────────────────

def build_sample_list(
    label_df: pd.DataFrame,
    writer_form_map: dict,
    lines_dir: str,
) -> list:
    """
    label_df의 각 행을 실제 이미지 파일 경로와 연결한다.

    매핑 규칙:
      writer_id + form_index → form_id
      form_id → lines/{form_id[:3]}/{form_id}/*.png
      각 줄 이미지가 하나의 학습 샘플 (같은 form의 모든 줄이 동일한 레이블 공유)

    반환: List of {"image_path": Path, "labels": np.ndarray(8,), "writer_id": str}
    """
    label_cols = [f"label_{i}" for i in range(8)]
    samples = []
    skipped = 0

    for _, row in label_df.iterrows():
        wid = row["writer_id"]
        fidx = int(row["form_index"])

        if wid not in writer_form_map:
            skipped += 1
            continue

        forms = writer_form_map[wid]
        if fidx >= len(forms):
            skipped += 1
            continue

        form_id = forms[fidx]                           # ex) "a01-003"
        parent = form_id[:3]                            # ex) "a01"
        form_dir = Path(lines_dir) / parent / form_id

        if not form_dir.exists():
            skipped += 1
            continue

        png_files = sorted(form_dir.glob("*.png"))
        if not png_files:
            skipped += 1
            continue

        labels = np.array(row[label_cols].values, dtype=np.float32)

        for img_path in png_files:
            samples.append({
                "image_path": img_path,
                "labels": labels,
                "writer_id": wid,
            })

    print(f"[build_sample_list] 샘플 수: {len(samples)}, 스킵: {skipped}")
    return samples


# ─────────────────────────────────────────────
# 4. PyTorch Dataset
# ─────────────────────────────────────────────

class HandwritingDataset(Dataset):
    """
    IAM 필기 줄 이미지를 로드하고 전처리하여 반환하는 Dataset.

    전처리:
      1. 그레이스케일 로드
      2. 세로(H)를 224px로 리사이즈 (가로비율 유지)
      3. 가로 방향에서 224×224 RandomCrop
      4. ToTensor → Normalize(0.5, 0.5)
    """

    TARGET_H = 224
    CROP_SIZE = 224

    def __init__(self, samples: list, augment: bool = True):
        self.samples = samples

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomCrop(self.CROP_SIZE),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(self.CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        img = Image.open(item["image_path"]).convert("L")  # 그레이스케일

        # 세로를 224px로 리사이즈 (비율 유지)
        w, h = img.size
        new_w = max(self.CROP_SIZE, int(w * self.TARGET_H / h))
        img = img.resize((new_w, self.TARGET_H), Image.LANCZOS)

        # 가로가 CROP_SIZE보다 짧으면 패딩
        if img.width < self.CROP_SIZE:
            padded = Image.new("L", (self.CROP_SIZE, self.TARGET_H), color=255)
            padded.paste(img, (0, 0))
            img = padded

        img_tensor = self.transform(img)
        labels = torch.tensor(item["labels"], dtype=torch.float32)
        return img_tensor, labels


# ─────────────────────────────────────────────
# 5. DataLoader 생성
# ─────────────────────────────────────────────

def get_dataloaders(
    lines_dir: str,
    xml_dir: str,
    label_txt: str,
    batch_size: int = 32,
    split: tuple = (0.7, 0.15, 0.15),
    num_workers: int = 0,
    seed: int = 42,
):
    """
    전체 파이프라인을 실행하여 train/val/test DataLoader를 반환한다.

    writer 단위로 분리하여 동일 필기자의 이미지가 train/test에 동시에 포함되지 않도록 한다.
    """
    label_df = parse_label_list(label_txt)
    writer_form_map = build_writer_form_map(xml_dir)
    all_samples = build_sample_list(label_df, writer_form_map, lines_dir)

    # writer 단위 분리
    rng = np.random.default_rng(seed)
    all_writers = sorted(set(s["writer_id"] for s in all_samples))
    rng.shuffle(all_writers)

    n = len(all_writers)
    n_train = int(n * split[0])
    n_val = int(n * split[1])

    train_writers = set(all_writers[:n_train])
    val_writers = set(all_writers[n_train:n_train + n_val])
    test_writers = set(all_writers[n_train + n_val:])

    train_samples = [s for s in all_samples if s["writer_id"] in train_writers]
    val_samples   = [s for s in all_samples if s["writer_id"] in val_writers]
    test_samples  = [s for s in all_samples if s["writer_id"] in test_writers]

    print(f"[get_dataloaders] train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}")

    train_loader = DataLoader(
        HandwritingDataset(train_samples, augment=True),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        HandwritingDataset(val_samples, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        HandwritingDataset(test_samples, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# 단독 실행 시: 파이프라인 검증 출력
# ─────────────────────────────────────────────

if __name__ == "__main__":
    BASE = Path(__file__).parent
    LINES_DIR = str(BASE / "lines")
    XML_DIR   = str(BASE / "xml")
    LABEL_TXT = str(BASE / "label_list.txt")

    label_df = parse_label_list(LABEL_TXT)
    writer_form_map = build_writer_form_map(XML_DIR)
    samples = build_sample_list(label_df, writer_form_map, LINES_DIR)

    # 레이블 분포 확인
    print("\n=== 레이블 분포 (1의 비율) ===")
    labels_all = np.array([s["labels"] for s in samples])
    for i in range(8):
        ratio = labels_all[:, i].mean()
        print(f"  label_{i}: {ratio:.3f} ({labels_all[:, i].sum():.0f} / {len(samples)})")

    # DataLoader 샘플 확인
    train_loader, val_loader, test_loader = get_dataloaders(
        LINES_DIR, XML_DIR, LABEL_TXT, batch_size=8
    )
    imgs, lbls = next(iter(train_loader))
    print(f"\n배치 이미지 shape: {imgs.shape}")   # (8, 1, 224, 224)
    print(f"배치 레이블 shape: {lbls.shape}")    # (8, 8)
    print("파이프라인 검증 완료!")
