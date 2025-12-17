# exp2_eval_hico600_oracle_map.py
# -*- coding: utf-8 -*-
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

# 학습된 head
HEAD_CKPT = ROOT / "checkpoints" / "exp2_head" / "best_head.pt"

# test feature 캐시
TEST_CACHE = ROOT / "cache" / "qwen3vl_test_union_full" / "manifest.json"

# obj_cls 붙인 pair jsonl
TEST_PAIR_OBJ = ROOT / "test_pairs_verb117_objcls.jsonl"
TRAIN_PAIR_OBJ = ROOT / "train_pairs_verb117_objcls.jsonl"

# HICO는 object 80개, verb 117개, HOI 600개지만
# "모든 (verb, obj)" 조합이 HOI인 건 아니라서,
# 간단하게 "조합 존재 여부를 train에서 본 조합으로 제한"해서 600-ish로 평가


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

def key_of(it):
    def nb(b): return tuple(int(round(float(x))) for x in b)
    return (it["image"], nb(it["human_bbox"]), nb(it["object_bbox"]))

def load_objcls_map(path: Path):
    m = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            m[key_of(it)] = int(it.get("obj_cls", -1))
    return m

def compute_coverage(jsonl_path):
    total = 0
    valid = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            it = json.loads(line)
            if int(it.get("obj_cls", -1)) != -1:
                valid += 1
    cov = valid / max(1, total)
    return total, valid, cov
   

def build_hoi_index_from_train(train_path: Path):
    # train에서 등장한 (obj_cls, verb_id) 조합만 HOI로 취급
    # -> "실제 등장 조합" 기준 Full/Rare/Non-rare 분리 가능
    hoi2id = {}
    counts = []
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            obj = int(it.get("obj_cls", -1))
            if obj < 1: 
                continue
            verb_ids = it.get("verb_ids", None)
            if verb_ids is None:
                # verb_labels에서 추출
                verb_ids = [i for i, v in enumerate(it["verb_labels"]) if v == 1]
            for v in verb_ids:
                k = (obj, int(v))
                if k not in hoi2id:
                    hoi2id[k] = len(hoi2id)
                    counts.append(0)
                counts[hoi2id[k]] += 1
    counts = torch.tensor(counts, dtype=torch.long)
    return hoi2id, counts  # counts: train positives per HOI

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not TEST_CACHE.exists():
        raise FileNotFoundError(f"test manifest not found: {TEST_CACHE}")

    # obj_cls map
    test_obj = load_objcls_map(TEST_PAIR_OBJ)

    tot, val, cov = compute_coverage(TEST_PAIR_OBJ)
    print(f"Coverage(test obj_cls!=-1): {val}/{tot} = {cov:.4f}")
    tot2, val2, cov2 = compute_coverage(TRAIN_PAIR_OBJ)
    print(f"Coverage(train obj_cls!=-1): {val2}/{tot2} = {cov2:.4f}")


    # HOI definition from train (obj, verb) combos
    hoi2id, train_counts = build_hoi_index_from_train(TRAIN_PAIR_OBJ)
    num_hoi = len(hoi2id)
    print(f"HOI combos from train: {num_hoi}")

    rare_mask = train_counts < 10
    nonrare_mask = ~rare_mask
    print(f"Rare HOI (<10): {int(rare_mask.sum())}, Non-rare: {int(nonrare_mask.sum())}")

    # load head
    ck = torch.load(HEAD_CKPT, map_location="cpu")
    dim = int(ck["dim"])
    head = HeadMLP(dim=dim, num_classes=117).to(device).eval()
    head.load_state_dict(ck["model"])
    print("Loaded head:", HEAD_CKPT)

    # load test shards
    manifest = json.loads(TEST_CACHE.read_text(encoding="utf-8"))
    shards = manifest["shards"]

    # 각 HOI별로 score/label 모으기 (총합 ~ N*117 정도라 가능)
    scores_list = [ [] for _ in range(num_hoi) ]
    labels_list = [ [] for _ in range(num_hoi) ]

    for s in tqdm(shards, desc="Collect"):
        d = torch.load(s["pt"], map_location="cpu")
        feats = d["feats"].float()
        labels = d["labels"].float()  # (M,117)

        meta_lines = Path(s["meta"]).read_text(encoding="utf-8").strip().splitlines()
        meta = [json.loads(x) for x in meta_lines]
        obj_cls = [test_obj.get(key_of(m), -1) for m in meta]

        with torch.no_grad():
            xb = feats.to(device, non_blocking=True)
            verb_scores = torch.sigmoid(head(xb)).cpu()  # (M,117)

        M = verb_scores.shape[0]
        for i in range(M):
            obj = int(obj_cls[i])
            if obj < 1:
                continue
            # 이 pair에서 모든 verb에 대해 (obj,verb) -> HOI id로 score/label 추가
            for v in range(117):
                k = (obj, v)
                if k not in hoi2id:
                    continue
                hid = hoi2id[k]
                scores_list[hid].append(float(verb_scores[i, v]))
                labels_list[hid].append(float(labels[i, v]))

    # AP 계산
    ap = torch.full((num_hoi,), float("nan"))
    for hid in tqdm(range(num_hoi), desc="AP"):
        s = torch.tensor(scores_list[hid], dtype=torch.float32)
        y = torch.tensor(labels_list[hid], dtype=torch.float32)
        ap[hid] = average_precision(s, y)

    def mean_ap(mask=None):
        x = ap
        if mask is not None:
            x = x[mask]
        x = x[~torch.isnan(x)]
        return float(x.mean().item()) if x.numel() > 0 else float("nan")

    full = mean_ap()
    rare = mean_ap(rare_mask)
    nonrare = mean_ap(nonrare_mask)

    print("\n=== Exp2 HICO-style HOI mAP (GT-pair oracle) ===")
    print(f"Full mAP    : {full:.4f}")
    print(f"Rare mAP    : {rare:.4f}")
    print(f"Non-rare mAP: {nonrare:.4f}")
    print("※ GT pair 고정(oracle) 기반 HOI mAP임. detector mAP과는 세팅이 다름.")

if __name__ == "__main__":
    main()
