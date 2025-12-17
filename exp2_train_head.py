# exp2_train_head.py
# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

CACHE_DIR = ROOT / "cache" / "qwen3vl_train_union_full"
MANIFEST_PATH = CACHE_DIR / "manifest.json"

OUT_DIR = ROOT / "checkpoints" / "exp2_head"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
VAL_RATIO = 0.05

EPOCHS = 5
BATCH_SIZE = 256          # head-only라 크게 잡아도 됨 (RAM/속도 보고 조절)
LR = 1e-3
WEIGHT_DECAY = 1e-4
USE_AMP = True
NUM_WORKERS = 0
# =========================


def set_seed(seed=42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    # scores, labels: (N,)
    pos = labels.sum().item()
    if pos <= 0:
        return float("nan")

    idx = torch.argsort(scores, descending=True)
    y = labels[idx]

    tp = torch.cumsum(y, dim=0)
    fp = torch.cumsum(1 - y, dim=0)
    prec = tp / (tp + fp + 1e-12)

    ap = (prec * y).sum() / (pos + 1e-12)
    return ap.item()


@torch.no_grad()
def compute_map(sigmoid_scores: torch.Tensor, labels: torch.Tensor):
    # sigmoid_scores, labels: (N, 117)
    aps = []
    for c in range(labels.shape[1]):
        aps.append(average_precision(sigmoid_scores[:, c], labels[:, c]))
    valid = [a for a in aps if not (isinstance(a, float) and math.isnan(a))]
    return float(sum(valid) / max(1, len(valid)))


class HeadMLP(nn.Module):
    def __init__(self, dim: int, num_classes: int = 117):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_cache_all():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"manifest not found: {MANIFEST_PATH}\n먼저 exp2_extract_features.py를 실행하세요.")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    shards = manifest.get("shards", [])
    if len(shards) == 0:
        raise RuntimeError("No shards in manifest. exp2_extract_features.py 먼저 실행 필요.")

    feats_list = []
    labels_list = []
    dim = None
    total = 0

    print(f"Loading {len(shards)} shards ...")
    for s in tqdm(shards, desc="Load shards"):
        d = torch.load(s["pt"], map_location="cpu")
        feats = d["feats"]     # (M, D) float16
        labels = d["labels"]   # (M, 117) uint8

        if dim is None:
            dim = feats.shape[1]
        total += feats.shape[0]
        feats_list.append(feats)
        labels_list.append(labels)

    feats = torch.cat(feats_list, dim=0)                 # (N, D) float16
    labels = torch.cat(labels_list, dim=0).float()       # (N, 117) float32
    print(f"Loaded feats={tuple(feats.shape)} labels={tuple(labels.shape)}")
    return feats, labels, dim


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    feats, labels, dim = load_cache_all()

    ds = TensorDataset(feats, labels)

    n_total = len(ds)
    n_val = max(1, int(n_total * VAL_RATIO))
    n_tr = n_total - n_val

    tr_ds, val_ds = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(SEED))

    tr_loader = DataLoader(
        tr_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    model = HeadMLP(dim=dim, num_classes=117).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_map = -1.0
    best_path = OUT_DIR / "best_head.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0

        for xb, yb in tqdm(tr_loader, desc=f"Epoch {epoch} [train]"):
            xb = xb.to(device, non_blocking=True).float()  # float16->float32 (head 학습 안정)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(xb)
                loss = crit(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += loss.item() * xb.size(0)

        tr_loss /= len(tr_ds)

        # val mAP (verb mAP)
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                xb = xb.to(device, non_blocking=True).float()
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    logits = model(xb)
                    scores = torch.sigmoid(logits).cpu()
                all_scores.append(scores)
                all_labels.append(yb.cpu())

        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        val_map = compute_map(all_scores, all_labels)

        print(f"\n[Epoch {epoch}] train_loss={tr_loss:.4f}  val_verb_mAP={val_map:.4f}")

        if val_map > best_map:
            best_map = val_map
            torch.save(
                {
                    "epoch": epoch,
                    "best_map": best_map,
                    "dim": dim,
                    "model": model.state_dict(),
                    "config": {
                        "VAL_RATIO": VAL_RATIO, "EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE,
                        "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY, "USE_AMP": USE_AMP,
                        "CACHE_DIR": str(CACHE_DIR),
                    }
                },
                best_path
            )
            print(f"✅ Saved best: {best_path} (best_map={best_map:.4f})")

    print(f"\nDone. Best val verb mAP = {best_map:.4f}")
    print("Checkpoint:", best_path)


if __name__ == "__main__":
    main()
