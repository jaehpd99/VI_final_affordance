import os
import json
import numpy as np
from scipy.io import loadmat

# =========================
# ✅ 너 환경 경로 (그대로 사용)
# =========================
ROOT = r"C:\Users\jaehp\Desktop\affordance\data\hicodet\hico_20160224_det"
ANNO_BBOX_MAT = os.path.join(ROOT, "anno_bbox.mat")
IMG_TRAIN_DIR = os.path.join(ROOT, "images", "train2015")
IMG_TEST_DIR  = os.path.join(ROOT, "images", "test2015")

OUT_TRAIN = r"C:\Users\jaehp\Desktop\affordance\train_pairs_verb117.jsonl"
OUT_TEST  = r"C:\Users\jaehp\Desktop\affordance\test_pairs_verb117.jsonl"

# 생성량 조절(너무 많으면 오래 걸림)
MAX_PAIRS_TRAIN = 200000   # None이면 전부
MAX_PAIRS_TEST  = 50000    # None이면 전부

# 디버그로 구조 확인하고 싶으면 True
INSPECT_ONLY = False


# =========================
# 유틸
# =========================
def _to_py_str(x):
    """matlab string/char/np types -> python str"""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.ndarray):
        # char array
        if x.dtype.kind in ("U", "S"):
            return str(x.squeeze().tolist())
        # object array with strings
        if x.size == 1:
            return _to_py_str(x.item())
        # sometimes char array comes as uint16 codes
        if x.dtype.kind in ("u", "i") and x.ndim >= 1:
            try:
                return "".join([chr(int(c)) for c in x.squeeze().tolist() if int(c) != 0])
            except:
                pass
    try:
        return str(x)
    except:
        return ""

def _as_struct_fields(s):
    """scipy.io loadmat struct -> list of field names"""
    if hasattr(s, "_fieldnames") and s._fieldnames is not None:
        return list(s._fieldnames)
    if isinstance(s, np.void) and s.dtype.names:
        return list(s.dtype.names)
    return []

def _get_field(s, name, default=None):
    """get field from matlab struct-ish object"""
    if hasattr(s, name):
        return getattr(s, name)
    if isinstance(s, np.void) and s.dtype.names and name in s.dtype.names:
        return s[name]
    return default

def _squeeze_list(x):
    """matlab arrays that may be shape (1,N) etc -> python list"""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        x = np.squeeze(x)
        if x.size == 0:
            return []
        if x.ndim == 0:
            return [x.item()]
        return [xi for xi in x.flat]
    return [x]

def _to_bbox_list(arr):
    """
    HICO anno_bbox.mat에서 bboxhuman/bboxobject는 보통:
      - mat_struct with fields x1,x2,y1,y2 (single bbox)
      - 또는 numpy array of such structs (multiple bboxes)
      - 또는 numeric arrays (rare)
    Return: list of [x1,y1,x2,y2] float
    """
    if arr is None:
        return []

    # 1) mat_struct with x1/x2/y1/y2
    f = _as_struct_fields(arr)
    if all(k in f for k in ["x1", "x2", "y1", "y2"]):
        x1 = float(np.squeeze(np.array(_get_field(arr, "x1"))).item())
        x2 = float(np.squeeze(np.array(_get_field(arr, "x2"))).item())
        y1 = float(np.squeeze(np.array(_get_field(arr, "y1"))).item())
        y2 = float(np.squeeze(np.array(_get_field(arr, "y2"))).item())
        return [[x1, y1, x2, y2]]

    # 2) array/list of mat_structs
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        out = []
        for item in _squeeze_list(arr):
            out.extend(_to_bbox_list(item))
        return out

    # 3) numeric array cases fallback
    a = np.array(arr)
    a = np.squeeze(a)
    if a.size == 0:
        return []
    if a.ndim == 1 and a.shape[0] == 4:
        return [a.astype(float).tolist()]
    if a.ndim == 2 and a.shape[1] == 4:
        return a.astype(float).tolist()
    if a.ndim == 2 and a.shape[0] == 4:
        return a.T.astype(float).tolist()

    return []



def _guess_hoi_id_field(hoi_struct):
    """
    HICO bbox mat에서 hoi struct 내부에 600 HOI class id가 들어있는 필드명이 버전에 따라 다름.
    흔히: 'id', 'hoi_id', 'action_id' 등.
    """
    candidates = ["id", "hoi_id", "action_id", "hoi", "hoi_idx", "id_hoi"]
    fields = _as_struct_fields(hoi_struct)
    for c in candidates:
        if c in fields:
            return c
    # fallback: 숫자 스칼라 필드 찾기
    for f in fields:
        v = _get_field(hoi_struct, f)
        v2 = np.squeeze(np.array(v))
        if v2.ndim == 0 and np.issubdtype(v2.dtype, np.number):
            return f
    return None

def _build_verb_map(list_action):
    """
    list_action: length 600.
    각 action에는 vname이 있음. 이를 unique 117 verbs로 매핑.
    Returns:
      verb_names (len=117)
      verb2idx (dict)
      hoi2verb_idx (len=600)  # HOI class index(1..600) -> verb_idx(0..116)
    """
    vnames = []
    for a in _squeeze_list(list_action):
        fields = _as_struct_fields(a)
        # vname 혹은 vname_ing 등 사용
        if "vname" in fields:
            v = _to_py_str(_get_field(a, "vname"))
        elif "vname_ing" in fields:
            v = _to_py_str(_get_field(a, "vname_ing"))
        else:
            v = "unknown"
        v = v.strip()
        vnames.append(v)

    # unique verbs (stable)
    verb_names = []
    verb2idx = {}
    for v in vnames:
        if v not in verb2idx:
            verb2idx[v] = len(verb_names)
            verb_names.append(v)

    # HOI(600) -> verb idx
    hoi2verb = []
    for v in vnames:
        hoi2verb.append(verb2idx[v])

    if len(verb_names) != 117:
        print(f"[WARN] unique verbs = {len(verb_names)} (expected 117). Still continuing.")
    return verb_names, verb2idx, hoi2verb

def _write_jsonl(out_path, rows, meta):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        # meta를 첫 줄에 남기고 싶으면 주석 형태로 저장하기 어려우니, 별도 파일로 저장
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta_path = os.path.splitext(out_path)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved:", out_path)
    print("Saved meta:", meta_path)

def _extract_pairs(bbox_split, img_dir, hoi2verb_idx, num_verbs, max_pairs=None):
    """
    bbox_split: bbox_train or bbox_test (list of per-image structs)
    Each image struct has fields: filename, size, hoi
    hoi contains list of HOI annotations, each contains human/object bboxes and HOI id
    """
    rows = []
    pair_count = 0

    # find one hoi fieldname for id (best effort) from first non-empty
    id_fieldname = None
    sample_hoi_fields = None

    for img_struct in bbox_split:
        filename = _to_py_str(_get_field(img_struct, "filename"))
        if not filename:
            continue
        img_path = os.path.join(img_dir, filename)
        hoi_list = _get_field(img_struct, "hoi")

        hoi_items = _squeeze_list(hoi_list)
        if len(hoi_items) == 0:
            continue

        # determine id field from first usable hoi
        if id_fieldname is None:
            for h in hoi_items:
                hf = _as_struct_fields(h)
                if hf:
                    id_fieldname = _guess_hoi_id_field(h)
                    sample_hoi_fields = hf
                    break

        for h in hoi_items:
            # HOI class id (1..600)
            if id_fieldname is None:
                continue
            hoi_id_raw = _get_field(h, id_fieldname)
            hoi_id = int(np.squeeze(np.array(hoi_id_raw)).item())
            if hoi_id < 1 or hoi_id > 600:
                continue

            verb_idx = hoi2verb_idx[hoi_id - 1]  # 0..116

            # bboxes
            # 많이 쓰는 필드명 후보들
            h_fields = _as_struct_fields(h)
            human_field = None
            obj_field = None
            for cand in ["bboxhuman", "human_bbox", "bbox_human", "hbbox", "human_box"]:
                if cand in h_fields:
                    human_field = cand
                    break

            for cand in ["bboxobject", "object_bbox", "bbox_object", "obbox", "object_box"]:
                if cand in h_fields:
                    obj_field = cand
                    break


            human_b = _to_bbox_list(_get_field(h, human_field)) if human_field else []
            obj_b   = _to_bbox_list(_get_field(h, obj_field)) if obj_field else []

            # (N,4)로 올 수도 있고 (1,4)일 수도 있음
            # 길이가 다르면 min으로 맞춤
            n = min(len(human_b), len(obj_b))
            if n == 0:
                continue

            for i in range(n):
                verb_labels = [0] * num_verbs
                verb_labels[verb_idx] = 1

                rows.append({
                    "image": img_path,
                    "human_bbox": human_b[i],
                    "object_bbox": obj_b[i],
                    "verb_labels": verb_labels,  
                    "verb_ids": [verb_idx],
                })
                pair_count += 1
                if max_pairs is not None and pair_count >= max_pairs:
                    meta = {
                        "num_pairs": pair_count,
                        "id_fieldname": id_fieldname,
                        "sample_hoi_fields": sample_hoi_fields,
                        "note": "verb_labels currently one-hot per HOI instance; you can merge same (image,human_bbox,object_bbox) to multi-hot later."
                    }
                    return rows, meta

    meta = {
        "num_pairs": pair_count,
        "id_fieldname": id_fieldname,
        "sample_hoi_fields": sample_hoi_fields,
        "note": "verb_labels currently one-hot per HOI instance; you can merge same (image,human_bbox,object_bbox) to multi-hot later."
    }
    return rows, meta


def main():
    print("Loading:", ANNO_BBOX_MAT)
    mat = loadmat(ANNO_BBOX_MAT, squeeze_me=True, struct_as_record=False)

    # keys 확인
    print("KEYS:", [k for k in mat.keys() if not k.startswith("__")])

    list_action = mat.get("list_action", None)
    bbox_train = mat.get("bbox_train", None)
    bbox_test  = mat.get("bbox_test", None)

    if list_action is None or bbox_train is None or bbox_test is None:
        raise RuntimeError("Expected keys list_action, bbox_train, bbox_test not found. Check mat keys.")

    # verb mapping
    verb_names, verb2idx, hoi2verb = _build_verb_map(list_action)
    num_verbs = len(verb_names)
    print("Unique verbs:", num_verbs)

    # bbox_train/bbox_test를 list로
    train_imgs = _squeeze_list(bbox_train)
    test_imgs  = _squeeze_list(bbox_test)

    print("Train images structs:", len(train_imgs))
    print("Test images structs:", len(test_imgs))

    if INSPECT_ONLY:
        # 구조 확인용: 첫 이미지/hoi 한 개 필드 출력
        first = train_imgs[0]
        print("\n[INSPECT] bbox_train[0] fields:", _as_struct_fields(first))
        hoi_items = _squeeze_list(_get_field(first, "hoi"))
        print("[INSPECT] num hoi items:", len(hoi_items))
        if hoi_items:
            print("[INSPECT] hoi[0] fields:", _as_struct_fields(hoi_items[0]))
        return

    # extract train/test pairs
    print("\nBuilding train pairs...")
    train_rows, train_meta = _extract_pairs(train_imgs, IMG_TRAIN_DIR, hoi2verb, num_verbs, MAX_PAIRS_TRAIN)
    _write_jsonl(OUT_TRAIN, train_rows, {
        "split": "train",
        "num_verbs": num_verbs,
        "verb_names": verb_names,
        **train_meta,
    })

    print("\nBuilding test pairs...")
    test_rows, test_meta = _extract_pairs(test_imgs, IMG_TEST_DIR, hoi2verb, num_verbs, MAX_PAIRS_TEST)
    _write_jsonl(OUT_TEST, test_rows, {
        "split": "test",
        "num_verbs": num_verbs,
        "verb_names": verb_names,
        **test_meta,
    })

    print("\nDone.")
    print("Train example:", train_rows[0] if train_rows else None)
    print("Test example:", test_rows[0] if test_rows else None)


if __name__ == "__main__":
    main()
