# attach_objcls_from_annobbox_iou.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent

TRAIN_IN = ROOT / "train_pairs_verb117.jsonl"
TEST_IN  = ROOT / "test_pairs_verb117.jsonl"
TRAIN_OUT = ROOT / "train_pairs_verb117_objcls.jsonl"
TEST_OUT  = ROOT / "test_pairs_verb117_objcls.jsonl"

ANNO_BBOX = ROOT / "data" / "hicodet" / "hico_20160224_det" / "anno_bbox.mat"

# bbox가 진짜 다를 수도 있으니 iou는 유지하되, 핵심은 obj_cls 추출
IOU_THR = 0.75

# obj cls는 1..80 (COCO 80)
OBJ_MIN, OBJ_MAX = 1, 80


def pick_first(*vals):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, np.ndarray):
            if v.size == 0:
                continue
            return v
        return v
    return None


def get_fields(obj):
    return obj.__dict__ if hasattr(obj, "__dict__") else {}


def to_str(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        x = x.reshape(-1)[0]
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.str_):
        return str(x)
    return str(x)


def norm_img_key(s: str):
    # jsonl에는 full path가 들어있고, mat에는 basename만 있음 -> basename로 통일
    if s is None:
        return None
    s = s.replace("\\", "/").strip()
    return Path(s).name


def iter_records(arr):
    a = np.array(arr)
    for x in a.reshape(-1):
        yield x


def to_scalar(v):
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return None
        v = v.reshape(-1)[0]
    try:
        return float(v)
    except Exception:
        return None


def mat_bbox_to_xyxy(b):
    """
    bbox: numeric [x1,x2,y1,y2] 또는 mat_struct(x1,x2,y1,y2) -> [x1,y1,x2,y2]
    """
    if b is None:
        return None
    if isinstance(b, np.ndarray):
        if b.size == 0:
            return None
        if b.size == 4 and np.issubdtype(b.dtype, np.number):
            a = b.reshape(-1).astype(float)
            return [a[0], a[2], a[1], a[3]]
        b = b.reshape(-1)[0]

    d = get_fields(b)
    if d:
        x1 = to_scalar(d.get("x1"))
        x2 = to_scalar(d.get("x2"))
        y1 = to_scalar(d.get("y1"))
        y2 = to_scalar(d.get("y2"))
        if None not in (x1, x2, y1, y2):
            return [x1, y1, x2, y2]
    return None


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-12
    return inter / union


# ===== 핵심: obj_cls 재귀 탐색 =====
SKIP_KEYS = {
    "bboxhuman", "bboxobject", "bbox_h", "bbox_o",
    "id", "verb", "action", "action_id", "hoi", "hois",
}

PRIOR_KEYS = [
    "objcategory", "objectcategory", "obj_cat", "obj_cls",
    "object", "obj", "category", "coco", "cls",
]


def extract_int(v):
    try:
        if isinstance(v, np.ndarray):
            if v.size == 0:
                return None
            v = v.reshape(-1)[0]
        return int(v)
    except Exception:
        return None


def find_objcls_recursive(obj, depth=0, path=""):
    """
    dict/mat_struct/ndarray 안을 재귀로 훑어서 1..80 범위 int 찾기
    우선 PRIOR_KEYS 힌트가 있는 경로를 먼저 탐색.
    """
    if depth > 4:
        return None

    # scalar or ndarray scalar
    iv = extract_int(obj)
    if iv is not None and OBJ_MIN <= iv <= OBJ_MAX:
        return (iv, path)

    # numpy array: iterate elements
    if isinstance(obj, np.ndarray):
        if obj.size == 0:
            return None
        # 너무 크면 첫 몇 개만
        flat = obj.reshape(-1)
        for i in range(min(len(flat), 8)):
            r = find_objcls_recursive(flat[i], depth + 1, f"{path}[{i}]")
            if r is not None:
                return r
        return None

    # mat_struct / object: __dict__
    d = get_fields(obj)
    if d:
        # 1) 우선 PRIOR 키 포함된 것부터
        keys = list(d.keys())
        keys_sorted = sorted(
            keys,
            key=lambda k: (0 if any(p in k.lower() for p in PRIOR_KEYS) else 1, k)
        )

        for k in keys_sorted:
            if k in SKIP_KEYS:
                continue
            v = d.get(k)
            # key 기반 프루닝: 너무 무관한 건 스킵(숫자 플래그 같은거)
            r = find_objcls_recursive(v, depth + 1, f"{path}.{k}" if path else k)
            if r is not None:
                return r
    return None


def build_img2triples():
    mat = sio.loadmat(str(ANNO_BBOX), squeeze_me=False, struct_as_record=False)
    if "bbox_train" not in mat or "bbox_test" not in mat:
        raise KeyError("anno_bbox.mat에 bbox_train/bbox_test가 없습니다.")

    img2 = {}  # basename -> list of (hb_xyxy, ob_xyxy, obj_cls)
    total_hoi = 0
    total_with_obj = 0
    sample_paths = []

    def ingest(key):
        nonlocal total_hoi, total_with_obj
        arr = mat[key]
        for rec in iter_records(arr):
            rd = get_fields(rec)
            fn = pick_first(rd.get("filename"), getattr(rec, "filename", None))
            fn = norm_img_key(to_str(fn))
            if fn is None:
                continue

            hoi = pick_first(rd.get("hoi"), getattr(rec, "hoi", None))
            if hoi is None:
                continue

            for h in iter_records(hoi):
                total_hoi += 1
                hd = get_fields(h)

                bh = pick_first(hd.get("bboxhuman"), getattr(h, "bboxhuman", None))
                bo = pick_first(hd.get("bboxobject"), getattr(h, "bboxobject", None))
                hb = mat_bbox_to_xyxy(bh)
                ob = mat_bbox_to_xyxy(bo)
                if hb is None or ob is None:
                    continue

                # 1) direct key 먼저
                obj_cls = None
                for k in ["objcategory", "objectcategory", "obj_cat", "obj_cls"]:
                    if k in hd:
                        iv = extract_int(hd[k])
                        if iv is not None and OBJ_MIN <= iv <= OBJ_MAX:
                            obj_cls = iv
                            break

                # 2) 재귀 탐색
                if obj_cls is None:
                    r = find_objcls_recursive(h, depth=0, path="hoi")
                    if r is not None:
                        obj_cls, pth = r
                        if len(sample_paths) < 5:
                            sample_paths.append(pth)

                if obj_cls is None:
                    continue

                total_with_obj += 1
                img2.setdefault(fn, []).append((hb, ob, int(obj_cls)))

    ingest("bbox_train")
    ingest("bbox_test")

    print(f"Total HOI entries: {total_hoi}")
    print(f"HOI with obj_cls  : {total_with_obj}")
    if sample_paths:
        print("Sample obj_cls paths:", sample_paths)

    return img2


def attach(in_path: Path, out_path: Path, img2):
    n = hit = miss = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            it = json.loads(line)
            n += 1
            fn = norm_img_key(it["image"])
            hb = [float(x) for x in it["human_bbox"]]
            ob = [float(x) for x in it["object_bbox"]]

            cand = img2.get(fn, None)
            best = (-1.0, None)

            if cand is not None:
                for (hb2, ob2, oc) in cand:
                    s = min(iou_xyxy(hb, hb2), iou_xyxy(ob, ob2))
                    if s > best[0]:
                        best = (s, oc)

            if best[0] >= IOU_THR:
                it["obj_cls"] = int(best[1])
                hit += 1
            else:
                it["obj_cls"] = -1
                miss += 1

            fout.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"[attach IoU] {in_path.name} -> {out_path.name}  matched={hit}/{n} miss={miss} (thr={IOU_THR})")


def main():
    print("Building img index (with recursive obj_cls search)...")
    img2 = build_img2triples()
    print("Images in index:", len(img2))

    attach(TRAIN_IN, TRAIN_OUT, img2)
    attach(TEST_IN, TEST_OUT, img2)

    print("DONE:", TRAIN_OUT, TEST_OUT)


if __name__ == "__main__":
    main()
