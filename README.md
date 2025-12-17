# Affordance Prediction using Vision-Language Models
### 시각지능학습 기말 프로젝트

본 프로젝트는 **Human–Object Interaction (HOI)** 상황에서  
사람이 물체에 대해 수행하는 **Affordance(행동/기능)** 를 예측하는 문제를 다룬다.

Vision-Language Model(VLM)을 기반으로,
zero-shot 예측부터 학습 가능한 분류기 및 추가적인 구조적 정보(geometry, text prior)를 단계적으로 결합하여
성능 향상을 분석하였다.

 - 실험 재현을 위해서는 HICO-DET 원본 데이터가 필요하다. https://github.com/fredzzhang/hicodet

---

## 1. Dataset

- **HICO-DET** 데이터셋 사용
- Human bounding box, Object bounding box, Verb label(117 classes) 포함
- Human–Object pair 단위로 데이터 재구성

본 repository에는 **pair annotation 결과만 포함**되어 있으며,
원본 이미지 데이터는 포함하지 않는다.
제외된 파일은 ()로 표시

## 2. Project Structure
### Included Annotation Files
```text
train_pairs_verb117.jsonl
train_pairs_verb117_meta.json
test_pairs_verb117.jsonl
test_pairs_verb117_meta.json


.
├─ train_pairs_verb117.jsonl          # train human–object pair + 117 verb 레이블
├─ (train_pairs_verb117_objcls.jsonl   # train pair + object class 정보)
├─ train_pairs_verb117_meta.json      # train 메타 정보 (통계, 인덱스 맵 등)

├─ test_pairs_verb117.jsonl           # test human–object pair + 117 verb 레이블
├─ (test_pairs_verb117_objcls.jsonl    # test pair + object class 정보)
├─ test_pairs_verb117_meta.json       # test 메타 정보

├─ build_pairs_verb117.py             # HICO-DET annotation → (pair, 117 verb) 포맷 생성
├─ attach_objcls_from_annobbox_iou.py # bbox IoU로 object class 부착 스크립트

├─ exp1_zeroshot_qwen3vl_recallk.py   # Exp1: Qwen3-VL zero-shot HOI, Recall@K 평가
├─ exp2_extract_features.py           # Exp2: Qwen3-VL 이미지 feature 캐시 생성
├─ exp2_extract_features_test.py      #       (test split용 feature 캐시)
├─ exp2_train_head.py                 # Exp2: frozen VLM + 단순 HOI head 학습
├─ exp2_eval_hico600_oracle_map.py    # Exp2: HICO-style HOI mAP (GT pair oracle)
├─ exp2_eval_head_on_test.py          # Exp2: pair-level Recall@K, verb mAP 평가

├─ exp3_extract_features_text_geom.py # (선택) Exp3용 feature 추출 스크립트
├─ exp3_train_head_geom.py            # Exp3: geometry 정보 추가 head 학습
├─ exp3_train_head_geom_textprior.py  # Exp3: geometry + text prior 학습
├─ exp3_eval_hico600_oracle_map.py    # Exp3: HICO-style HOI mAP (GT pair oracle)
├─ exp3_eval_head_on_test_geom.py     # Exp3: pair-level Recall@K, verb mAP 평가
├─ exp3_build_text_prior_clip.py      # CLIP 기반 verb–object prior 계산'
├─ exp3_build_text_prior_clip_jsononly.py
├─ exp3_extract_verb_names_from_annobbox.py
├─ exp3_visualize_affordance_overlay.py  # 예측 HOI 시각화 (bbox + verb overlay)

├─ (cache/                             # Qwen3-VL feature 캐시(.pt 등))
├─ (data/                              # 원본 HICO-DET 및 전처리 데이터)
├─ (checkpoints/                       # 학습된 HOI head checkpoint)
└─ runs/                              # 로그, tensorboard 등

```

## 3. Experiments & Evaluation Metrics
### Exp1. Zero-shot VLM Baseline

Exp1은 학습 없이 VLM의 순수 zero-shot 능력을 평가하는 기준 실험이다.

- 모델: Qwen3-VL-2B-Instruct

- 입력: Human–Object union 영역 이미지

- 출력: Verb ID top-K

- 학습 여부: (zero-shot)

- 평가 대상: Test set 중 일부 샘플 (N=1000)

- 평가 지표: Recall@K

**목적**

- 사전학습된 VLM이 HOI / affordance 인식에 얼마나 취약한지를 확인
- 이후 실험(Exp2, Exp3)의 개선 폭을 비교하기 위한 기준선(Baseline) 제공


### Exp2. VLM Feature + Trainable Head

Exp2는 VLM을 feature extractor로 사용하고,
행동 분류를 위한 trainable MLP head를 학습하는 방식이다.

 - Feature extractor: Qwen3-VL (고정, frozen)

 - Classifier: MLP Head (117 verb multi-label)

 - 입력 feature: Human–Object pair visual embedding

 - 학습 데이터: HICO-DET train pairs

 - 평가 데이터: HICO-DET test pairs (N=25,136)
   
**목적**

 - Zero-shot의 한계를 넘기 위해 HOI 전용 supervision을 통해 affordance 개념을 학습

 - Pair-level verb 분류 성능과 HICO-style HOI mAP (GT-pair oracle) 평가

**특징**

 - 사람–물체 쌍이 이미 정확히 주어졌다는 가정(GT pair oracle) 하에 평가 → detector 성능과는 분리된 순수 관계 인식 성능

### Exp3. Geometry + Text Prior (Final Model)
Exp3는 Exp2 구조에 두 가지 핵심 정보를 추가한 최종 모델이다.

**(1) Geometry Feature**

 - Human / Object bounding box 간:

   - 상대 위치

   - 크기 비율

   - IoU, 교차 비율 등

 - 행동은 공간적 관계와 강하게 결합됨을 반영

**(2) Text Prior (Verb Semantic Prior)**

 - CLIP 기반으로 verb 이름을 임베딩

 - 시각 feature와 결합하여:

   - 희소한 verb (Rare class)

   - 의미적으로 유사한 행동 간 일반화 강화

**목적**

 - Affordance = 시각 + 공간 + 의미 정보의 결합이라는 가설 검증

 - Rare HOI 성능 개선 여부 확인

---

### Evaluation Metrics
**1. Recall@K (Pair-level)**

정의 : 각 Human–Object pair에 대해 GT verb 중 하나라도 Top-K 예측에 포함되면 성공

사용 이유 : 하나의 쌍에 여러 verb가 가능한 multi-label 특성 반영 Affordance 관점에서 “가능한 행동을 맞췄는가” 평가

**2. Verb mAP (Mean Average Precision)**

정의 : 117개 verb 각각에 대해 AP 계산 후 평균

사용 이유 : 클래스 불균형이 심한 HOI 문제에서 전반적인 분류 품질을 정량적으로 평가

**3. HICO-style HOI mAP (GT-pair Oracle)**

정의 : 사람–물체 쌍을 정답으로 고정한 상태에서 verb 분류 성능만으로 계산한 HOI mAP / Full / Rare / Non-rare로 분리 평가

⚠️ 본 지표는 detector mAP이 아님
→ 관계 인식(affordance reasoning) 능력만을 평가하기 위한 설정
## 4. Experimental Results

### 실험별 성능 요약
| Exp | 설정 | Full mAP | Rare mAP | Non-rare mAP | Recall@1 | Recall@5 | Recall@10 | Verb mAP |
|----:|------|---------:|---------:|-------------:|---------:|---------:|----------:|---------:|
| Exp1 | Zero-shot VLM (Qwen3-VL) | – | – | – | 0.0330 | 0.0800 | 0.1660 | – |
| Exp2 | VLM Feature + Trainable Head | 0.3225 | 0.1184 | 0.3458 | 0.3748 | 0.9350 | 0.9839 | 0.3200 |
| Exp3 | + Geometry | 0.3505 | 0.1636 | 0.3719 | 0.3941 | 0.9482 | 0.9876 | 0.3492 |
| Exp3 | + Geometry + Text Prior | **0.3531** | **0.1745** | **0.3735** | – | – | – | - |

## 5. Affordance Visualization

Exp3 모델의 예측 결과를 이미지 상에 시각적으로 표현한다.

## 6. Conclusion

- Zero-shot VLM은 기본적인 affordance 추론 가능성을 보였으나 성능 한계 존재

- VLM feature 기반 학습 head를 통해 성능이 크게 향상됨
- Geometry 정보는 affordance 예측에 매우 효과적이며, 특히 Rare class 성능을 크게 개선함
- Text prior는 전반적인 성능을 소폭이지만 일관되게 향상시킴
- Geometry 및 Text prior를 추가한 Exp3 모델이 가장 우수한 성능을 달성함
- Affordance 예측에서 시각 정보 + 구조적 관계 + 의미적 prior의 결합이 중요함을 확인하였다
