# exp3_eval_hico600_oracle_map.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

# 여기서 원하는 ckpt 골라서 평가
HEAD_CKPT = ROOT / "checkpoints" / "exp3_head_geom_textprior" / "best_head_geom_textprior.pt"
#HEAD_CKPT = ROOT / "checkpoints" / "exp3_head_geom" / "best_head_geom.pt"

TEST_CACHE = ROOT / "cache" / "qwen3vl_test_union_full" / "manifest.json"
TEST_PAIR_OBJ = ROOT / "test_pairs_verb117_objcls.jsonl"
TRAIN_PAIR_OBJ = ROOT / "train_pairs_verb117_objcls.jsonl"

NUM_VERBS = 117
USE_AMP = True


# ========= heads =========
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


class HeadGeomTextPrior(nn.Module):
    """
    logits = w * sim_logits + (1-w) * mlp_logits
    sim_logits = scale * (normalize(proj(x)) @ text_emb.T)
    """
    def __init__(self, in_dim: int, text_emb: torch.Tensor):
        super().__init__()
        assert text_emb.ndim == 2 and text_emb.shape[0] == NUM_VERBS
        self.register_buffer("text_emb", text_emb)  # (117,512) normalized

        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, NUM_VERBS),
        )
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        z = self.proj(x)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
        scale = torch.clamp(self.logit_scale, 1.0, 50.0)
        sim_logits = scale * (z @ self.text_emb.t())
        mlp_logits = self.mlp(x)
        w = torch.sigmoid(self.mix_logit)
        logits = w * sim_logits + (1.0 - w) * mlp_logits
        return logits


def load_head_from_ckpt(ckpt_path: Path, device: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    dim = int(ck["dim"])

    sd = ck["model"]
    # text-prior head인지 감지: proj.* 키가 있으면 그거임
    is_textprior = any(k.startswith("proj.") for k in sd.keys()) or any(k.startswith("mlp.") for k in sd.keys())

    if is_textprior:
        # text prior embeds 로드
        text_prior_pt = ck.get("text_prior_pt", None)
        if text_prior_pt is None:
            raise KeyError("text-prior ckpt인데 text_prior_pt 경로가 ckpt에 없음")

        tp = torch.load(text_prior_pt, map_location="cpu")
        text_emb = tp["embeds"].float()  # (117,512)

        head = HeadGeomTextPrior(in_dim=dim, text_emb=text_emb).to(device).eval()
        head.load_state_dict(sd, strict=True)
        head_type = "HeadGeomTextPrior"
        return head, dim, head_type

    # 기본 MLP head
    head = HeadMLP(dim=dim, num_classes=NUM_VERBS).to(device).eval()
    head.load_state_dict(sd, strict=True)
    head_type = "HeadMLP"
    return head, dim, head_type


# ========= utils =========
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


def key_of(it):
    def nb(b):
        return tuple(int(round(float(x))) for x in b)
    return (it["image"], nb(it["human_bbox"]), nb(it["object_bbox"]))


def load_objcls_map(path: Path):
    m = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            m[key_of(it)] = int(it.get("obj_cls", -1))
    return m


def compute_coverage(jsonl_path: Path):
    total = 0
    valid = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            it = json.loads(line)
            if int(it.get("obj_cls", -1)) != -1:
                valid += 1
    cov = valid / max(1, total)
    return total, valid, cov


def build_hoi_index_from_train(train_path: Path):
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
                verb_ids = [i for i, v in enumerate(it["verb_labels"]) if v == 1]
            for v in verb_ids:
                k = (obj, int(v))
                if k not in hoi2id:
                    hoi2id[k] = len(hoi2id)
                    counts.append(0)
                counts[hoi2id[k]] += 1
    counts = torch.tensor(counts, dtype=torch.long)
    return hoi2id, counts


# ===== geom (훈련/평가 동일) =====
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # coverage
    tot, val, cov = compute_coverage(TEST_PAIR_OBJ)
    print(f"Coverage(test obj_cls!=-1): {val}/{tot} = {cov:.4f}")
    tot2, val2, cov2 = compute_coverage(TRAIN_PAIR_OBJ)
    print(f"Coverage(train obj_cls!=-1): {val2}/{tot2} = {cov2:.4f}")

    # obj_cls map
    test_obj = load_objcls_map(TEST_PAIR_OBJ)

    # HOI index from train combos
    hoi2id, train_counts = build_hoi_index_from_train(TRAIN_PAIR_OBJ)
    num_hoi = len(hoi2id)
    print(f"HOI combos from train: {num_hoi}")

    rare_mask = train_counts < 10
    nonrare_mask = ~rare_mask
    print(f"Rare HOI (<10): {int(rare_mask.sum())}, Non-rare: {int(nonrare_mask.sum())}")

    # load head (auto-detect)
    head, dim, head_type = load_head_from_ckpt(HEAD_CKPT, device)
    print(f"Loaded head: {HEAD_CKPT}")
    print(f"Head type : {head_type}, dim={dim}")

    # load test shards
    manifest = json.loads(TEST_CACHE.read_text(encoding="utf-8"))
    shards = manifest["shards"]

    scores_list = [[] for _ in range(num_hoi)]
    labels_list = [[] for _ in range(num_hoi)]
    skipped_pairs = 0

    for s in tqdm(shards, desc="Collect", ncols=200):
        d = torch.load(s["pt"], map_location="cpu")
        feats = d["feats"].float()      # (M,2048)
        labels = d["labels"].float()    # (M,117)

        meta_lines = Path(s["meta"]).read_text(encoding="utf-8").strip().splitlines()
        meta = [json.loads(x) for x in meta_lines]
        obj_cls = [test_obj.get(key_of(m), -1) for m in meta]

        # geom concat
        G = np.stack([geom_from_bboxes(m["human_bbox"], m["object_bbox"]) for m in meta], axis=0)
        G = torch.from_numpy(G).float()

        x = torch.cat([feats, G], dim=1).to(device, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = head(x)
            verb_scores = torch.sigmoid(logits).cpu()

        M = verb_scores.shape[0]
        for i in range(M):
            obj = int(obj_cls[i])
            if obj < 1:
                skipped_pairs += 1
                continue
            for v in range(NUM_VERBS):
                k = (obj, v)
                if k not in hoi2id:
                    continue
                hid = hoi2id[k]
                scores_list[hid].append(float(verb_scores[i, v]))
                labels_list[hid].append(float(labels[i, v]))

    print(f"Skipped pairs in eval (obj_cls<1): {skipped_pairs}")

    ap = torch.full((num_hoi,), float("nan"))
    for hid in tqdm(range(num_hoi), desc="AP", ncols=200):
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

    print("\n=== HICO-style HOI mAP (GT-pair oracle) ===")
    print(f"Full mAP    : {full:.4f}")
    print(f"Rare mAP    : {rare:.4f}")
    print(f"Non-rare mAP: {nonrare:.4f}")
    print("※ GT pair 고정(oracle) 기반 HOI mAP임. detector mAP과는 세팅이 다름.")


if __name__ == "__main__":
    main()
