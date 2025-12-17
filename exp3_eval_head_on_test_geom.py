# exp3_eval_head_on_test_geom.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

HEAD_CKPT = ROOT / "checkpoints" / "exp3_head_geom" / "best_head_geom.pt"
TEST_MANIFEST = ROOT / "cache" / "qwen3vl_test_union_full" / "manifest.json"

NUM_VERBS = 117
USE_AMP = True


def _fix_xyxy(b):
    x1, y1, x2, y2 = map(float, b)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


def geom_from_bboxes(hb, ob):
    eps = 1e-6
    hx1, hy1, hx2, hy2 = _fix_xyxy(hb)
    ox1, oy1, ox2, oy2 = _fix_xyxy(ob)

    hw = max(eps, hx2 - hx1); hh = max(eps, hy2 - hy1)
    ow = max(eps, ox2 - ox1); oh = max(eps, oy2 - oy1)

    hcx = (hx1 + hx2) * 0.5; hcy = (hy1 + hy2) * 0.5
    ocx = (ox1 + ox2) * 0.5; ocy = (oy1 + oy2) * 0.5

    ux1 = min(hx1, ox1); uy1 = min(hy1, oy1)
    ux2 = max(hx2, ox2); uy2 = max(hy2, oy2)
    uw = max(eps, ux2 - ux1); uh = max(eps, uy2 - uy1)

    h_area = hw * hh; o_area = ow * oh; u_area = uw * uh

    ix1 = max(hx1, ox1); iy1 = max(hy1, oy1)
    ix2 = min(hx2, ox2); iy2 = min(hy2, oy2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = h_area + o_area - inter + eps
    iou = inter / union

    dcx = (hcx - ocx) / uw
    dcy = (hcy - ocy) / uh

    log_hw = np.log(hw + eps); log_hh = np.log(hh + eps)
    log_ow = np.log(ow + eps); log_oh = np.log(oh + eps)
    log_ar_h = np.log((hw / hh) + eps)
    log_ar_o = np.log((ow / oh) + eps)
    log_ar_u = np.log((uw / uh) + eps)

    log_h_u = np.log((h_area / u_area) + eps)
    log_o_u = np.log((o_area / u_area) + eps)
    log_h_o = np.log((h_area / o_area) + eps)

    inter_h = inter / (h_area + eps)
    inter_o = inter / (o_area + eps)

    rel_hx1 = (hx1 - ux1) / uw; rel_hy1 = (hy1 - uy1) / uh
    rel_hx2 = (hx2 - ux1) / uw; rel_hy2 = (hy2 - uy1) / uh
    rel_ox1 = (ox1 - ux1) / uw; rel_oy1 = (oy1 - uy1) / uh
    rel_ox2 = (ox2 - ux1) / uw; rel_oy2 = (oy2 - uy1) / uh

    return np.array([
        dcx, dcy,
        log_hw, log_hh, log_ow, log_oh,
        log_ar_h, log_ar_o, log_ar_u,
        log_h_u, log_o_u, log_h_o,
        iou, inter_h, inter_o,
        rel_hx1, rel_hy1, rel_hx2, rel_hy2,
        rel_ox1, rel_oy1, rel_ox2, rel_oy2,
    ], dtype=np.float32)


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
    def forward(self, x): return self.net(x)


@torch.no_grad()
def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    pos = labels.sum().item()
    if pos <= 0: return float("nan")
    idx = torch.argsort(scores, descending=True)
    y = labels[idx]
    tp = torch.cumsum(y, dim=0)
    fp = torch.cumsum(1 - y, dim=0)
    prec = tp / (tp + fp + 1e-12)
    ap = (prec * y).sum() / (pos + 1e-12)
    return ap.item()


@torch.no_grad()
def verb_map(scores: torch.Tensor, labels: torch.Tensor) -> float:
    aps = []
    for v in range(scores.shape[1]):
        ap = average_precision(scores[:, v], labels[:, v])
        if not np.isnan(ap):
            aps.append(ap)
    return float(np.mean(aps)) if aps else float("nan")


def recall_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    # (N,117)
    topk = torch.topk(scores, k=k, dim=1).indices  # (N,k)
    hit = 0
    N = scores.size(0)
    for i in range(N):
        gt = (labels[i] > 0.5).nonzero(as_tuple=False).view(-1)
        if gt.numel() == 0:
            continue
        if torch.isin(topk[i], gt).any():
            hit += 1
    return hit / max(1, N)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ck = torch.load(HEAD_CKPT, map_location="cpu")
    dim = int(ck["dim"])
    head = HeadMLP(dim=dim, num_classes=NUM_VERBS).to(device).eval()
    head.load_state_dict(ck["model"])
    print("Loaded head:", HEAD_CKPT)

    manifest = json.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    shards = manifest["shards"]

    all_scores, all_labels = [], []

    for s in tqdm(shards, desc="Eval shards", ncols=200):
        d = torch.load(s["pt"], map_location="cpu")
        feats = d["feats"].float()
        labels = d["labels"].float()

        meta_lines = Path(s["meta"]).read_text(encoding="utf-8").strip().splitlines()
        meta = [json.loads(x) for x in meta_lines]
        G = np.stack([geom_from_bboxes(m["human_bbox"], m["object_bbox"]) for m in meta], axis=0)
        G = torch.from_numpy(G).float()

        x = torch.cat([feats, G], dim=1).to(device, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = head(x)
            scores = torch.sigmoid(logits).cpu()

        all_scores.append(scores)
        all_labels.append(labels)

    S = torch.cat(all_scores, dim=0)
    L = torch.cat(all_labels, dim=0)

    r1 = recall_at_k(S, L, 1)
    r5 = recall_at_k(S, L, 5)
    r10 = recall_at_k(S, L, 10)
    vmap = verb_map(S, L)

    print("\n=== Exp3 Test Result (pair-level) ===")
    print(f"Recall@1:  {r1:.4f}  (N={S.size(0)})")
    print(f"Recall@5:  {r5:.4f}  (N={S.size(0)})")
    print(f"Recall@10: {r10:.4f}  (N={S.size(0)})")
    print(f"Verb mAP : {vmap:.4f}  (N={S.size(0)})")


if __name__ == "__main__":
    main()
