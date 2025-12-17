# exp3_build_text_prior_clip.py
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
OUT_PT = OUT_DIR / "verb_text_embeds.pt"
OUT_JSON = OUT_DIR / "verb_names.json"

CLIP_TEXT_ID = "openai/clip-vit-base-patch32"  # text dim=512
NUM_VERBS = 117


def load_verb_names_json_first() -> list:
    """JSON이 있으면 무조건 JSON을 우선 사용."""
    if OUT_JSON.exists():
        names = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        if (not isinstance(names, list)) or (len(names) != NUM_VERBS):
            raise RuntimeError(f"verb_names.json 형식 이상: {OUT_JSON}")
        print(f"✅ Loaded verb names from JSON: {OUT_JSON}")
    else:
        raise FileNotFoundError(
            f"verb_names.json이 없습니다. 먼저 exp3_extract_verb_names_from_annobbox.py를 실행하세요: {OUT_JSON}"
        )


def fallback_names():
    return [f"verb_{i}" for i in range(NUM_VERBS)]


@torch.no_grad()
def encode_texts(device: str, names: list) -> torch.Tensor:
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_TEXT_ID)
    # torch<2.6 안전: safetensors 강제
    text_model = CLIPTextModel.from_pretrained(CLIP_TEXT_ID, use_safetensors=True).to(device).eval()

    # prompt 템플릿
    prompts = [f"a photo of a person {v.replace('_', ' ')} an object" for v in names]

    embeds = []
    bs = 64
    for i in tqdm(range(0, len(prompts), bs), desc="TextEncode", ncols=140):
        batch = prompts[i:i+bs]
        tok = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        out = text_model(**tok)
        # CLIP text: last_hidden_state의 EOS 토큰 위치를 뽑는 게 정석이지만,
        # 여기선 간단히 pooled 출력(있으면) / 없으면 EOS 사용
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            e = out.pooler_output
        else:
            # EOS token index 사용
            eos_id = tokenizer.eos_token_id
            # 각 샘플에서 eos 위치 찾기 (없으면 마지막 토큰)
            input_ids = tok["input_ids"]
            eos_pos = (input_ids == eos_id).int().argmax(dim=1)
            e = out.last_hidden_state[torch.arange(input_ids.size(0), device=device), eos_pos]
        embeds.append(e.detach())

    E = torch.cat(embeds, dim=0).float()  # (117,512)
    E = E / (E.norm(dim=1, keepdim=True) + 1e-6)
    return E.cpu()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("CLIP:", CLIP_TEXT_ID)

    names = load_verb_names_json_first()
    if names:
        print(f"✅ verb names loaded from JSON: {OUT_JSON}")
    else:
        print("⚠️ verb names JSON not usable -> fallback(verb_0..116)")
        names = fallback_names()
        # fallback도 json으로 저장해두긴 함
        OUT_JSON.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")

    E = encode_texts(device, names)

    payload = {
        "clip_text_id": CLIP_TEXT_ID,
        "names": names,
        "embeds": E,  # torch.Tensor (117,512)
    }
    torch.save(payload, OUT_PT)

    print("Saved:", OUT_PT)
    print("Saved:", OUT_JSON)
    print("Embeds:", tuple(E.shape))


if __name__ == "__main__":
    main()
