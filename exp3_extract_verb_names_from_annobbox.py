# exp3_extract_verb_names_from_annobbox.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent

ANNO_CAND = [
    ROOT / "data" / "hicodet" / "hico_20160224_det" / "anno_bbox.mat",
    ROOT / "data" / "hicodet" / "hico_20160224_det" / "annotations" / "anno_bbox.mat",
    ROOT / "anno_bbox.mat",
]

OUT_DIR = ROOT / "cache" / "text_prior_clip"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "verb_names.json"

NUM_VERBS = 117

# hoi에서 verb id로 보일 만한 필드들
VERB_KEYS = ["verb", "verb_id", "vid", "v", "v_id", "action_verb", "verbindex"]


def find_annobbox():
    for p in ANNO_CAND:
        if p.exists():
            return p
    raise FileNotFoundError("anno_bbox.mat를 못 찾았습니다: " + " | ".join(map(str, ANNO_CAND)))


def mat_struct_to_dict(s):
    if hasattr(s, "_fieldnames"):
        return {k: getattr(s, k) for k in s._fieldnames}
    return None


def scalar(x):
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return scalar(x.item())
    return x


def to_int(x):
    x = scalar(x)
    try:
        return int(x)
    except Exception:
        return None


def to_str(x):
    x = scalar(x)
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.ndarray) and x.dtype.kind in ("U", "S"):
        return "".join([str(t) for t in x.reshape(-1)])
    return str(x)


def get_action_id(hoi_dict):
    # 대부분 'id'가 action id(1..600)
    for k in ["id", "action_id", "hoi_id", "act_id"]:
        if k in hoi_dict:
            v = to_int(hoi_dict[k])
            if v is not None:
                return v
    return None


def get_verb_id(hoi_dict):
    # 명확한 키 먼저 찾기
    for k in VERB_KEYS:
        if k in hoi_dict:
            v = to_int(hoi_dict[k])
            if v is not None and 1 <= v <= NUM_VERBS:
                return v

    # 없으면: 값들 중 1..117 범위인 정수 후보를 "하나" 찾아보기(보수적)
    # (너무 공격적으로 하면 엉뚱한 필드 잡을 수 있어서)
    candidates = []
    for k, v in hoi_dict.items():
        vi = to_int(v)
        if vi is not None and 1 <= vi <= NUM_VERBS:
            # bbox 좌표 같은 거랑 충돌 줄이려고 키 이름에 'v'/'verb' 포함된 경우 우선
            score = 0
            kk = k.lower()
            if "verb" in kk or kk in ["v", "vid"]:
                score += 10
            candidates.append((score, vi, k))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def iter_hoi_list(x):
    # hoi가 리스트/배열/단일 struct로 올 수 있음
    if x is None:
        return
    if isinstance(x, np.ndarray):
        for it in x.reshape(-1):
            yield it
    else:
        yield x


def main():
    p = find_annobbox()
    print("Using:", p)

    mat = sio.loadmat(str(p), squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    print("mat keys:", keys)

    if "list_action" not in mat:
        raise KeyError("anno_bbox.mat에 list_action이 없습니다.")

    list_action = mat["list_action"]
    # list_action: 600개 HOI 정의
    try:
        la_items = list_action.reshape(-1).tolist()
    except Exception:
        la_items = [list_action]
    print("list_action len:", len(la_items))

    # action_id(1..600) -> vname
    actionid_to_vname = {}
    for i, it in enumerate(la_items, start=1):
        d = mat_struct_to_dict(it) or {}
        vname = None
        for kk in ["vname", "verbname", "verb_name", "name"]:
            if kk in d:
                vname = to_str(d[kk])
                break
        if not vname:
            vname = f"verb_action_{i}"
        actionid_to_vname[i] = vname.strip()

    # bbox_train/bbox_test에서 hoi를 훑으면서 verb_id -> vname 매핑 수집
    verbid_to_name = {}  # 1..117
    seen_pairs = 0
    used = 0

    def ingest(split_key):
        nonlocal seen_pairs, used
        if split_key not in mat:
            return
        recs = mat[split_key]
        try:
            rec_list = recs.reshape(-1).tolist()
        except Exception:
            rec_list = [recs]
        print(f"[{split_key}] records:", len(rec_list))

        for rd in rec_list:
            rd_d = mat_struct_to_dict(rd)
            if not rd_d:
                continue

            hoi = rd_d.get("hoi", None)
            if hoi is None:
                hoi = rd_d.get("hois", None)
            if hoi is None:
                continue

            for h in iter_hoi_list(hoi):
                hd = mat_struct_to_dict(h)
                if not hd:
                    continue

                aid = get_action_id(hd)  # 1..600
                vid = get_verb_id(hd)    # 1..117 (있어야 함)

                seen_pairs += 1
                if aid is None or not (1 <= aid <= 600):
                    continue
                if vid is None or not (1 <= vid <= NUM_VERBS):
                    continue

                vname = actionid_to_vname.get(aid, None)
                if not vname:
                    continue

                # 한 번 정해지면 그대로 유지(충돌 거의 없음)
                if vid not in verbid_to_name:
                    verbid_to_name[vid] = vname
                    used += 1

    ingest("bbox_train")
    ingest("bbox_test")

    print("mapped verb ids:", len(verbid_to_name), f"/{NUM_VERBS}")
    print("scanned hoi entries:", seen_pairs, "used:", used)

    # 1..117 순서로 이름 생성
    names = []
    missing = 0
    for vid in range(1, NUM_VERBS + 1):
        nm = verbid_to_name.get(vid, None)
        if nm is None:
            nm = f"verb_{vid-1}"
            missing += 1
        names.append(nm)

    OUT_JSON.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", OUT_JSON)
    print(f"Fallback count: {missing}/{NUM_VERBS}")
    print("Sample 15:", names[:15])


if __name__ == "__main__":
    main()
