# exp3_train_head_geom.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent

# Train cache (union crop 기준, 너가 이미 만든 것)
TRAIN_MANIFEST = ROOT / "cache" / "qwen3vl_train_union_full" / "manifest.json"

# 출력 ckpt
OUT_DIR = ROOT / "checkpoints" / "exp3_head_geom"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_CKPT = OUT_DIR / "best_head_geom.pt"

NUM_VERBS = 117
BASE_DIM = 2048  # Qwen3-VL feature dim (너 캐시 기준)
BATCH_SIZE = 256
EPOCHS = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
USE_AMP = True
VAL_RATIO = 0.1
SEED = 42


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _fix_xyxy(b):
    x1, y1, x2, y2 = map(float, b)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def geom_from_bboxes(hb, ob):
    """
    hb, ob: [x1,y1,x2,y2]
    return: (G,) float32
    이미지 크기 없이도 안정적으로 쓰려고 union 박스 기반으로 정규화.
    """
    eps = 1e-6
    hx1, hy1, hx2, hy2 = _fix_xyxy(hb)
    ox1, oy1, ox2, oy2 = _fix_xyxy(ob)

    hw = max(eps, hx2 - hx1)
    hh = max(eps, hy2 - hy1)
    ow = max(eps, ox2 - ox1)
    oh = max(eps, oy2 - oy1)

    hcx = (hx1 + hx2) * 0.5
    hcy = (hy1 + hy2) * 0.5
    ocx = (ox1 + ox2) * 0.5
    ocy = (oy1 + oy2) * 0.5

    ux1 = min(hx1, ox1)
    uy1 = min(hy1, oy1)
    ux2 = max(hx2, ox2)
    uy2 = max(hy2, oy2)
    uw = max(eps, ux2 - ux1)
    uh = max(eps, uy2 - uy1)

    h_area = hw * hh
    o_area = ow * oh
    u_area = uw * uh

    ix1 = max(hx1, ox1)
    iy1 = max(hy1, oy1)
    ix2 = min(hx2, ox2)
    iy2 = min(hy2, oy2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = h_area + o_area - inter + eps
    iou = inter / union

    # 중심 거리 (union 기준 정규화)
    dcx = (hcx - ocx) / uw
    dcy = (hcy - ocy) / uh

    # 로그/비율 기반(스케일 변화에 덜 민감)
    log_hw = np.log(hw + eps)
    log_hh = np.log(hh + eps)
    log_ow = np.log(ow + eps)
    log_oh = np.log(oh + eps)
    log_ar_h = np.log((hw / hh) + eps)
    log_ar_o = np.log((ow / oh) + eps)
    log_ar_u = np.log((uw / uh) + eps)

    log_h_u = np.log((h_area / u_area) + eps)
    log_o_u = np.log((o_area / u_area) + eps)
    log_h_o = np.log((h_area / o_area) + eps)

    inter_h = inter / (h_area + eps)
    inter_o = inter / (o_area + eps)

    # 위치(0~1 느낌의 값) union 내부 상대 좌표
    rel_hx1 = (hx1 - ux1) / uw
    rel_hy1 = (hy1 - uy1) / uh
    rel_hx2 = (hx2 - ux1) / uw
    rel_hy2 = (hy2 - uy1) / uh
    rel_ox1 = (ox1 - ux1) / uw
    rel_oy1 = (oy1 - uy1) / uh
    rel_ox2 = (ox2 - ux1) / uw
    rel_oy2 = (oy2 - uy1) / uh

    g = np.array([
        dcx, dcy,
        log_hw, log_hh, log_ow, log_oh,
        log_ar_h, log_ar_o, log_ar_u,
        log_h_u, log_o_u, log_h_o,
        iou, inter_h, inter_o,
        rel_hx1, rel_hy1, rel_hx2, rel_hy2,
        rel_ox1, rel_oy1, rel_ox2, rel_oy2,
    ], dtype=np.float32)
    return g


GEOM_DIM = 23


class PairFeatGeomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X  # (N, D)
        self.Y = Y  # (N, 117)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


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


@torch.no_grad()
def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
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
def verb_map(scores: torch.Tensor, labels: torch.Tensor) -> float:
    # scores/labels: (N,117)
    aps = []
    for v in range(scores.shape[1]):
        ap = average_precision(scores[:, v], labels[:, v])
        if not (np.isnan(ap)):
            aps.append(ap)
    if len(aps) == 0:
        return float("nan")
    return float(np.mean(aps))


def load_manifest(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_train_arrays(manifest_path: Path):
    manifest = load_manifest(manifest_path)
    shards = manifest["shards"]
    feats_list, labels_list, geom_list = [], [], []

    pbar = tqdm(shards, desc="Load shards", ncols=200)
    for s in pbar:
        d = torch.load(s["pt"], map_location="cpu")
        feats = d["feats"].float()      # (M,2048)
        labels = d["labels"].float()    # (M,117)

        meta_lines = Path(s["meta"]).read_text(encoding="utf-8").strip().splitlines()
        meta = [json.loads(x) for x in meta_lines]

        # geom 계산
        G = np.stack([geom_from_bboxes(m["human_bbox"], m["object_bbox"]) for m in meta], axis=0)
        G = torch.from_numpy(G).float()

        feats_list.append(feats)
        labels_list.append(labels)
        geom_list.append(G)

    F = torch.cat(feats_list, dim=0)         # (N,2048)
    Y = torch.cat(labels_list, dim=0)        # (N,117)
    G = torch.cat(geom_list, dim=0)          # (N,GEOM_DIM)

    X = torch.cat([F, G], dim=1)             # (N, 2048+GEOM_DIM)
    return X, Y


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("TRAIN_MANIFEST:", TRAIN_MANIFEST)

    if not TRAIN_MANIFEST.exists():
        raise FileNotFoundError(f"Train manifest not found: {TRAIN_MANIFEST}")

    X, Y = load_train_arrays(TRAIN_MANIFEST)
    print(f"Loaded X={tuple(X.shape)} Y={tuple(Y.shape)}  (geom_dim={GEOM_DIM})")

    N = X.shape[0]
    idx = torch.randperm(N)
    n_val = int(N * VAL_RATIO)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xva, Yva = X[val_idx], Y[val_idx]

    train_ds = PairFeatGeomDataset(Xtr, Ytr)
    val_ds = PairFeatGeomDataset(Xva, Yva)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    dim = X.shape[1]
    head = HeadMLP(dim=dim, num_classes=NUM_VERBS).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best = -1.0

    for ep in range(1, EPOCHS + 1):
        head.train()
        tr_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep} [train]", ncols=200)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = head(xb)
                loss = crit(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += loss.item() * xb.size(0)
            pbar.set_postfix(loss=float(loss.item()))

        tr_loss /= max(1, len(train_ds))

        head.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {ep} [val]", ncols=200):
                xb = xb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    logits = head(xb)
                scores = torch.sigmoid(logits).cpu()
                all_scores.append(scores)
                all_labels.append(yb.cpu())

        S = torch.cat(all_scores, dim=0)
        L = torch.cat(all_labels, dim=0)
        vmap = verb_map(S, L)

        print(f"\n[Epoch {ep}] train_loss={tr_loss:.4f}  val_verb_mAP={vmap:.4f}")

        if vmap > best:
            best = vmap
            ckpt = {
                "dim": dim,
                "geom_dim": GEOM_DIM,
                "model": head.state_dict(),
                "best_val_verb_map": best,
            }
            torch.save(ckpt, BEST_CKPT)
            print(f"✅ Saved best: {BEST_CKPT} (best_map={best:.4f})")

    print(f"\nDone. Best val verb mAP = {best:.4f}")
    print("Checkpoint:", BEST_CKPT)


if __name__ == "__main__":
    main()
