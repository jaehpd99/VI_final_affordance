# exp3_build_text_prior_clip_jsononly.py
# -*- coding: utf-8 -*-
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

ROOT = Path(__file__).resolve().parent

OUT_DIR = ROOT / "cache" / "text_prior_clip"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 입력: 반드시 이 JSON만 사용 (없으면 에러)
VERB_JSON = OUT_DIR / "verb_names.json"

# 출력: pt만 갱신
OUT_PT = OUT_DIR / "verb_text_embeds.pt"

CLIP_TEXT_ID = "openai/clip-vit-base-patch32"
NUM_VERBS = 117

@torch.no_grad()
def encode_texts(device: str, names: list) -> torch.Tensor:
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_TEXT_ID)
    # torch 이슈 회피: safetensors 강제
    text_model = CLIPTextModel.from_pretrained(CLIP_TEXT_ID, use_safetensors=True).to(device).eval()

    prompts = [f"a photo of a person {v.replace('_',' ')} an object" for v in names]

    bs = 64
    outs = []
    for i in tqdm(range(0, len(prompts), bs), desc="TextEncode", ncols=120):
        batch = prompts[i:i+bs]
        tok = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        out = text_model(**tok)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            e = out.pooler_output
        else:
            eos_id = tokenizer.eos_token_id
            input_ids = tok["input_ids"]
            eos_pos = (input_ids == eos_id).int().argmax(dim=1)
            e = out.last_hidden_state[torch.arange(input_ids.size(0), device=device), eos_pos]

        outs.append(e.detach())

    E = torch.cat(outs, dim=0).float()  # (117,512)
    E = E / (E.norm(dim=1, keepdim=True) + 1e-6)
    return E.cpu()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("ROOT:", ROOT)
    print("Device:", device)
    print("VERB_JSON:", VERB_JSON)
    print("OUT_PT   :", OUT_PT)

    if not VERB_JSON.exists():
        raise FileNotFoundError(
            f"verb_names.json이 없습니다. 먼저 exp3_extract_verb_names_from_annobbox.py를 실행하세요.\n{VERB_JSON}"
        )

    names = json.loads(VERB_JSON.read_text(encoding="utf-8"))
    if (not isinstance(names, list)) or (len(names) != NUM_VERBS):
        raise RuntimeError(f"verb_names.json 형식/길이 이상: len={len(names) if isinstance(names,list) else 'NA'}")

    # 여기서 verb_0..인지 바로 감지
    if all(isinstance(x, str) and x.startswith("verb_") for x in names[:20]):
        print("⚠️ WARNING: verb_names.json이 아직 verb_0.. 형태입니다. (extract 다시 실행 필요)")
    else:
        print("✅ Loaded real verb names. Sample 15:", names[:15])

    E = encode_texts(device, names)
    payload = {"clip_text_id": CLIP_TEXT_ID, "names": names, "embeds": E}
    torch.save(payload, OUT_PT)
    print("✅ Saved:", OUT_PT)
    print("Embeds:", tuple(E.shape))

if __name__ == "__main__":
    main()
