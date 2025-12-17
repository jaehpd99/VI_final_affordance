# exp2_eval_head_on_test.py
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
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

# train에서 학습된 head
HEAD_CKPT = ROOT / "checkpoints" / "exp2_head" / "best_head.pt"

# test 캐시
CACHE_DIR = ROOT / "cache" / "qwen3vl_test_union_full"
MANIFEST_PATH = CACHE_DIR / "manifest.json"

# 평가 K
K_LIST = (1, 5, 10)

# (선택) 예측 저장
SAVE_PRED_JSONL = True
PRED_JSONL_PATH = ROOT / "runs" / "preds_exp2_head_test_pairs.jsonl"


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
def compute_map(scores: torch.Tensor, labels: torch.Tensor) -> float:
    aps = []
    for c in range(labels.shape[1]):
        aps.append(average_precision(scores[:, c], labels[:, c]))
    valid = [a for a in aps if not (isinstance(a, float) and math.isnan(a))]
    return float(sum(valid) / max(1, len(valid)))


def recall_at_k(gt_ids, topk_ids) -> float:
    s = set(topk_ids)
    return 1.0 if any(g in s for g in gt_ids) else 0.0


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"manifest not found: {MANIFEST_PATH}\n먼저 exp2_extract_features_test.py 실행하세요.")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    m = load_manifest()
    shards = m["shards"]
    dim = int(shards[0]["dim"])

    ck = torch.load(HEAD_CKPT, map_location="cpu")
    model = HeadMLP(dim=dim, num_classes=117)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print("Loaded head:", HEAD_CKPT)

    # 예측 저장 준비
    if SAVE_PRED_JSONL:
        PRED_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if PRED_JSONL_PATH.exists():
            PRED_JSONL_PATH.unlink()  # 덮어쓰기

    all_scores = []
    all_labels = []

    sum_recall = {k: 0.0 for k in K_LIST}
    n_total = 0

    for s in tqdm(shards, desc="Eval shards"):
        d = torch.load(s["pt"], map_location="cpu")
        feats = d["feats"].float()     # (M, D)
        labels = d["labels"].float()   # (M, 117)

        # meta (image, bbox) 읽기
        meta_path = Path(s["meta"])
        meta_lines = meta_path.read_text(encoding="utf-8").strip().splitlines()

        with torch.no_grad():
            xb = feats.to(device, non_blocking=True)
            logits = model(xb)
            scores = torch.sigmoid(logits).cpu()

        all_scores.append(scores)
        all_labels.append(labels)

        # Recall@K
        topk_vals, topk_ids = torch.topk(scores, k=max(K_LIST), dim=1)
        for i in range(scores.shape[0]):
            gt_ids = torch.nonzero(labels[i] > 0.5).squeeze(1).tolist()
            for k in K_LIST:
                sum_recall[k] += recall_at_k(gt_ids, topk_ids[i, :k].tolist())
        n_total += scores.shape[0]

        # (선택) 예측 jsonl 저장 (pair 단위)
        if SAVE_PRED_JSONL:
            with PRED_JSONL_PATH.open("a", encoding="utf-8") as f:
                for i in range(scores.shape[0]):
                    meta = json.loads(meta_lines[i])
                    # top-10만 저장
                    ids10 = topk_ids[i, :10].tolist()
                    sc10 = topk_vals[i, :10].tolist()
                    meta.update({
                        "top10_verb_ids": ids10,
                        "top10_scores": sc10,
                    })
                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    verb_map = compute_map(all_scores, all_labels)

    print("\n=== Exp2 Test Result (pair-level) ===")
    for k in K_LIST:
        print(f"Recall@{k}: {sum_recall[k] / max(1, n_total):.4f}  (N={n_total})")
    print(f"Verb mAP : {verb_map:.4f}  (N={n_total})")

    if SAVE_PRED_JSONL:
        print("Saved preds:", PRED_JSONL_PATH)


if __name__ == "__main__":
    main()
