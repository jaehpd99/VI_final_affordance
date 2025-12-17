# exp2_extract_features_test.py
# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

ROOT = Path(__file__).resolve().parent

JSONL_PATH = ROOT / "test_pairs_verb117.jsonl"
IMAGE_DIR = ROOT / "data" / "hicodet" / "hico_20160224_det" / "images" / "test2015"

USE_UNION = True
MAX_SAMPLES = -1
SHARD_SIZE = 512

OUT_DIR = ROOT / "cache" / "qwen3vl_test_union_full"
MANIFEST_PATH = OUT_DIR / "manifest.json"

PROMPT = "Action?"


def clamp_box_xyxy(box, w, h):
    x1, y1, x2, y2 = map(float, box)
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)
    return (x1, y1, x2, y2)


def union_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))


def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "model_id": MODEL_ID,
        "jsonl": str(JSONL_PATH),
        "image_dir": str(IMAGE_DIR),
        "use_union": USE_UNION,
        "shard_size": SHARD_SIZE,
        "max_samples": MAX_SAMPLES,
        "shards": [],
        "total_saved": 0,
        "done": False,
    }


def save_manifest(m):
    MANIFEST_PATH.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


@torch.no_grad()
def pooled_from_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)


@torch.inference_mode()
def extract_feature_fast(model: Qwen3VLForConditionalGeneration, inputs: dict) -> torch.Tensor:
    base = getattr(model, "model", None)
    if base is not None:
        try:
            out = base(**inputs, return_dict=True)
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                return pooled_from_last_hidden(out.last_hidden_state, inputs["attention_mask"])
        except Exception:
            pass

    out2 = model(**inputs, output_hidden_states=True, return_dict=True)
    return pooled_from_last_hidden(out2.hidden_states[-1], inputs["attention_mask"])


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print("=== Exp2 Extract TEST Features (FAST) ===")
    print("JSONL      :", JSONL_PATH)
    print("IMAGE_DIR  :", IMAGE_DIR)
    print("USE_UNION  :", USE_UNION)
    print("MAX_SAMPLES:", MAX_SAMPLES)
    print("SHARD_SIZE :", SHARD_SIZE)
    print("OUT_DIR    :", OUT_DIR)
    print("========================================\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    if manifest.get("done", False):
        print("✅ Already done (manifest.done=True). Nothing to do.")
        return

    start_idx = int(manifest.get("total_saved", 0))
    print(f"Resume: starting from index {start_idx}\n")

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype="auto", device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    items = load_jsonl(JSONL_PATH)
    if MAX_SAMPLES > 0:
        items = items[:min(MAX_SAMPLES, len(items))]

    feat_buf, label_buf, meta_lines = [], [], []
    shard_id = len(manifest["shards"])
    total = start_idx

    pbar = tqdm(range(start_idx, len(items)), desc="Extract TEST")
    for i in pbar:
        it = items[i]
        img_path = IMAGE_DIR / it["image"]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        hb = clamp_box_xyxy(it["human_bbox"], w, h)
        ob = clamp_box_xyxy(it["object_bbox"], w, h)
        crop = img.crop(union_xyxy(hb, ob)) if USE_UNION else img.crop(ob)

        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": crop}, {"type": "text", "text": PROMPT}],
        }]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt"
        ).to(model.device)

        feat = extract_feature_fast(model, inputs)  # (1, D)
        feat = feat.detach().cpu().to(torch.float16).squeeze(0)
        label = torch.tensor(it["verb_labels"], dtype=torch.uint8)

        feat_buf.append(feat)
        label_buf.append(label)
        meta_lines.append(json.dumps({
            "image": it["image"],
            "human_bbox": it["human_bbox"],
            "object_bbox": it["object_bbox"],
        }, ensure_ascii=False))

        total += 1
        pbar.set_postfix(saved=total, shard=len(feat_buf))

        if len(feat_buf) >= SHARD_SIZE:
            feats = torch.stack(feat_buf, dim=0)
            labels = torch.stack(label_buf, dim=0)
            pt_path = OUT_DIR / f"shard_{shard_id:05d}.pt"
            meta_path = OUT_DIR / f"shard_{shard_id:05d}.jsonl"
            torch.save({"feats": feats, "labels": labels}, pt_path)
            meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

            manifest["shards"].append({"pt": str(pt_path), "meta": str(meta_path),
                                       "n": int(feats.shape[0]), "dim": int(feats.shape[1])})
            manifest["total_saved"] = int(total)
            save_manifest(manifest)

            shard_id += 1
            feat_buf.clear(); label_buf.clear(); meta_lines.clear()

    if len(feat_buf) > 0:
        feats = torch.stack(feat_buf, dim=0)
        labels = torch.stack(label_buf, dim=0)
        pt_path = OUT_DIR / f"shard_{shard_id:05d}.pt"
        meta_path = OUT_DIR / f"shard_{shard_id:05d}.jsonl"
        torch.save({"feats": feats, "labels": labels}, pt_path)
        meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
        manifest["shards"].append({"pt": str(pt_path), "meta": str(meta_path),
                                   "n": int(feats.shape[0]), "dim": int(feats.shape[1])})
        manifest["total_saved"] = int(total)
        save_manifest(manifest)

    manifest["done"] = True
    save_manifest(manifest)
    print("\n✅ TEST feature cache DONE.")
    print(f"Saved shards: {len(manifest['shards'])}, total_saved={manifest['total_saved']}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
