# zeroshot_qwen3vl_recallk.py
# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
from typing import List

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
USE_UNION = True
MAX_SAMPLES = 1000
K_LIST = (1, 5, 10)
MAX_NEW_TOKENS = 64

ROOT = os.path.dirname(os.path.abspath(__file__))

TRAIN_JSONL = os.path.join(ROOT, "train_pairs_verb117.jsonl")
TEST_JSONL  = os.path.join(ROOT, "test_pairs_verb117.jsonl")

IMG_DIR_TRAIN = os.path.join(ROOT, "data", "hicodet", "hico_20160224_det", "images", "train2015")
IMG_DIR_TEST  = os.path.join(ROOT, "data", "hicodet", "hico_20160224_det", "images", "test2015")


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


def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def parse_pred_ids(text: str, k: int) -> List[int]:
    nums = re.findall(r"\d+", text)
    ids = []
    for n in nums:
        v = int(n)
        if 0 <= v < 117:
            ids.append(v)
    seen, uniq = set(), []
    for v in ids:
        if v not in seen:
            uniq.append(v); seen.add(v)
    return uniq[:k]


def recall_at_k(gt_ids: List[int], pred_ids: List[int], k: int) -> float:
    if len(gt_ids) == 0:
        return 0.0
    predk = set(pred_ids[:k])
    return 1.0 if any(g in predk for g in gt_ids) else 0.0


def pick_split_auto():
    # test가 있으면 test를 기본으로, 없으면 train
    if os.path.exists(TEST_JSONL) and os.path.exists(IMG_DIR_TEST):
        return "test", TEST_JSONL, IMG_DIR_TEST
    if os.path.exists(TRAIN_JSONL) and os.path.exists(IMG_DIR_TRAIN):
        return "train", TRAIN_JSONL, IMG_DIR_TRAIN
    # 여기까지 오면 경로가 잘못된 것
    raise FileNotFoundError(
        "Cannot find jsonl/image_dir automatically.\n"
        f"TEST_JSONL={TEST_JSONL}\nIMG_DIR_TEST={IMG_DIR_TEST}\n"
        f"TRAIN_JSONL={TRAIN_JSONL}\nIMG_DIR_TRAIN={IMG_DIR_TRAIN}\n"
        "=> 위 경로가 실제 폴더 구조랑 맞는지 확인하세요."
    )


def main():
    split, jsonl_path, image_dir = pick_split_auto()
    print("=== Exp1: Zero-shot VLM Recall@K ===")
    print("split     :", split)
    print("jsonl     :", jsonl_path)
    print("image_dir :", image_dir)
    print("use_union :", USE_UNION)
    print("max_samp  :", MAX_SAMPLES)
    print("K_list    :", K_LIST)
    print("====================================\n")

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    items = load_jsonl(jsonl_path)
    if MAX_SAMPLES > 0:
        items = items[:min(MAX_SAMPLES, len(items))]

    max_k = max(K_LIST)
    prompt = (
        "You are given an image that contains a human and an object (a human-object pair). "
        "Predict the TOP-K most likely action verbs (from 0..116) describing what the human does to the object. "
        "Return ONLY one line in this exact format:\n"
        "IDs: <id1>, <id2>, <id3>, ...\n"
        "Do not include any other text.\n"
        f"K = {max_k}.\n"
    )

    total = 0
    sum_recall = {k: 0.0 for k in K_LIST}

    for it in tqdm(items, desc="Zero-shot VLM"):
        img_path = os.path.join(image_dir, it["image"])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        hb = clamp_box_xyxy(it["human_bbox"], w, h)
        ob = clamp_box_xyxy(it["object_bbox"], w, h)

        if USE_UNION:
            crop = img.crop(union_xyxy(hb, ob))
        else:
            crop = img.crop(ob)

        gt = it.get("verb_ids", None)
        if gt is None:
            gt = [i for i, v in enumerate(it["verb_labels"]) if int(v) == 1]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": crop},
                {"type": "text", "text": prompt},
            ],
        }]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        pred_ids = parse_pred_ids(out, k=max_k)

        total += 1
        for k in K_LIST:
            sum_recall[k] += recall_at_k(gt, pred_ids, k)

    print("\n=== Result ===")
    for k in K_LIST:
        print(f"Recall@{k}: {sum_recall[k] / max(1, total):.4f}  (N={total})")


if __name__ == "__main__":
    main()
