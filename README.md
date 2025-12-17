# Affordance Prediction using Vision-Language Models
### 시각지능학습 기말 프로젝트

본 프로젝트는 **Human–Object Interaction (HOI)** 상황에서  
사람이 물체에 대해 수행하는 **Affordance(행동/기능)** 를 예측하는 문제를 다룬다.

Vision-Language Model(VLM)을 기반으로,
zero-shot 예측부터 학습 가능한 분류기 및 추가적인 구조적 정보(geometry, text prior)를 단계적으로 결합하여
성능 향상을 분석하였다.

---

## 1. Dataset

- **HICO-DET** 데이터셋 사용
- Human bounding box, Object bounding box, Verb label(117 classes) 포함
- Human–Object pair 단위로 데이터 재구성

본 repository에는 **pair annotation 결과만 포함**되어 있으며,
원본 이미지 데이터는 포함하지 않는다.

### Included Annotation Files
```text
train_pairs_verb117.jsonl
train_pairs_verb117_meta.json
test_pairs_verb117.jsonl
test_pairs_verb117_meta.json


## 2. Project Structure
.
├── build_pairs_verb117.py                 # HOI pair 생성
├── attach_objcls_from_annobbox_iou.py     # object category 연결
│
├── exp1_zeroshot_qwen3vl_recallk.py       # Exp1: Zero-shot VLM
│
├── exp2_extract_features.py               # Exp2: VLM feature 추출
├── exp2_extract_features_test.py
├── exp2_train_head.py                     # 분류 head 학습
├── exp2_eval_head_on_test.py              # Exp2 평가
├── exp2_eval_hico600_oracle_map.py
│
├── exp3_build_text_prior_clip.py          # Verb text prior 생성
├── exp3_build_text_prior_clip_jsononly.py
├── exp3_extract_verb_names_from_annobbox.py
├── exp3_train_head_geom.py                # Geometry head 학습
├── exp3_train_head_geom_textprior.py      # Geometry + Text prior
├── exp3_eval_head_on_test_geom.py          # Exp3 평가
├── exp3_eval_hico600_oracle_map.py
├── exp3_visualize_affordance_overlay.py   # 시각화
│
├── vis_exp3_overlay/                      # 시각화 결과
└── vis/


## 3. Experiments
Exp1. Zero-shot Affordance Prediction

모델: Qwen3-VL

입력: Human–Object union 영역 이미지

출력: Verb ID top-K

평가 지표: Recall@K
