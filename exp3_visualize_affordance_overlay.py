# exp3_visualize_affordance_overlay.py
# -*- coding: utf-8 -*-
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

# ====== 네 최신 Exp3 체크포인트 ======
HEAD_CKPT = ROOT / "checkpoints" / "exp3_head_geom_textprior" / "best_head_geom_textprior.pt"

# ====== Exp3에서 쓰는 text prior  ======
TEXT_PRIOR_PT = ROOT / "cache" / "text_prior_clip" / "verb_text_embeds.pt"

# ====== test feature cache  ======
TEST_MANIFEST = ROOT / "cache" / "qwen3vl_test_union_full" / "manifest.json"

# ====== 출력 폴더 ======
OUT_DIR = ROOT / "runs" / "vis_exp3_overlay"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ====== 시각화 옵션 ======
NUM_IMAGES = 10          # 몇 장 뽑을지
PAIRS_PER_IMAGE = 1      # 한 이미지에서 몇 pair까지 그릴지(너무 많으면 지저분함)
TOPK_VERBS = 3           # top-k affordance
SEED = 42

# bbox 색
COLOR_HUMAN = (255, 0, 0)    # red
COLOR_OBJECT = (0, 128, 255) # blue

USE_AMP = True
NUM_VERBS = 117


# ========= Head (geom + textprior) =========
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
        return w * sim_logits + (1.0 - w) * mlp_logits


# ========= geometry (훈련/평가와 동일하게) =========
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


def load_font(size=18):
    # 윈도우 기본 폰트 fallback
    for cand in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/맑은 고딕.ttf",
    ]:
        try:
            return ImageFont.truetype(cand, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_box(draw, box, color, width=3):
    x1, y1, x2, y2 = _fix_xyxy(box)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(f"Missing: {TEST_MANIFEST}")

    # load verb names + text embeds
    tp = torch.load(TEXT_PRIOR_PT, map_location="cpu")
    verb_names = tp.get("names", [f"verb_{i}" for i in range(NUM_VERBS)])
    text_emb = tp["embeds"].float()
    # 이미 normalize 돼있지만 안전하게 한 번 더
    text_emb = text_emb / (text_emb.norm(dim=1, keepdim=True) + 1e-6)

    # load head ckpt
    ck = torch.load(HEAD_CKPT, map_location="cpu")
    dim = int(ck["dim"])
    head = HeadGeomTextPrior(in_dim=dim, text_emb=text_emb).to(device).eval()
    head.load_state_dict(ck["model"], strict=True)

    print("Loaded head:", HEAD_CKPT)
    print("Loaded text prior:", TEXT_PRIOR_PT)

    manifest = json.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    shards = manifest["shards"]

    # 1) 우선 샘플 이미지 리스트 만들기 (meta에서 랜덤 추출)
    all_meta = []
    for s in shards:
        meta_lines = Path(s["meta"]).read_text(encoding="utf-8").strip().splitlines()
        if not meta_lines:
            continue
        # 너무 많이 읽을 필요 없어서 일부만 샘플링
        take = min(200, len(meta_lines))
        picks = random.sample(meta_lines, k=take) if len(meta_lines) > take else meta_lines
        for line in picks:
            all_meta.append(json.loads(line))

    # 이미지별로 그룹
    img2pairs = {}
    for m in all_meta:
        img = m["image"]
        img2pairs.setdefault(img, 0)
        img2pairs[img] += 1

    img_list = list(img2pairs.keys())
    random.shuffle(img_list)
    img_list = img_list[:NUM_IMAGES]
    print(f"Pick images: {len(img_list)}")

    font = load_font(18)

    # 2) 각 이미지마다, 전체 shard를 스캔해서 해당 이미지의 pair들을 모으고 예측/오버레이
    for img_path in tqdm(img_list, desc="Visualize", ncols=120):
        found = []  # (feat, hb, ob)
        # shard scan (50개면 충분히 빠름)
        for s in shards:
            d = torch.load(s["pt"], map_location="cpu")
            feats = d["feats"].float()

            meta_lines = Path(s["meta"]).read_text(encoding="utf-8").strip().splitlines()
            if not meta_lines:
                continue
            meta = [json.loads(x) for x in meta_lines]

            for i, m in enumerate(meta):
                if m["image"] == img_path:
                    found.append((feats[i], m["human_bbox"], m["object_bbox"]))
                    if len(found) >= PAIRS_PER_IMAGE:
                        break
            if len(found) >= PAIRS_PER_IMAGE:
                break

        if not found:
            continue

        # load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # 경로 문제 대비: 파일명만으로 재시도
            p = Path(img_path)
            if p.exists():
                img = Image.open(p).convert("RGB")
            else:
                continue

        draw = ImageDraw.Draw(img)

        feats_stack = torch.stack([x[0] for x in found], dim=0)  # (P,2048)
        geom_stack = np.stack([geom_from_bboxes(x[1], x[2]) for x in found], axis=0)  # (P,23)
        geom_stack = torch.from_numpy(geom_stack).float()

        x = torch.cat([feats_stack, geom_stack], dim=1)  # (P, dim)
        assert x.shape[1] == dim, f"dim mismatch: x={x.shape[1]} ckpt_dim={dim}"

        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=USE_AMP and device.startswith("cuda")):
                logits = head(x)
            probs = torch.sigmoid(logits).float().cpu()  # (P,117)

        # draw each pair
        for pi, (_, hb, ob) in enumerate(found):
            draw_box(draw, hb, COLOR_HUMAN, width=3)
            draw_box(draw, ob, COLOR_OBJECT, width=3)

            p = probs[pi]
            topv = torch.topk(p, k=TOPK_VERBS).indices.tolist()
            lines = []
            for v in topv:
                lines.append(f"{verb_names[v]}:{p[v].item():.2f}")

            # label 위치: human box 왼쪽 위
            x1, y1, _, _ = _fix_xyxy(hb)
            tx, ty = int(x1) + 4, max(0, int(y1) - 18 * (len(lines)+1))
            text = "Affordance: " + ", ".join(lines)

            # 텍스트 배경
            pad = 3
            tw, th = draw.textbbox((0,0), text, font=font)[2:]
            draw.rectangle([tx-pad, ty-pad, tx+tw+pad, ty+th+pad], fill=(0,0,0))
            draw.text((tx, ty), text, fill=(255,255,255), font=font)

        out_name = Path(img_path).stem + "_exp3.png"
        out_path = OUT_DIR / out_name
        img.save(out_path)
        print("Saved:", out_path)

    print("\nDone.")
    print("Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
