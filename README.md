# CUK_NL_team3
2025-2 Natural Language Processing term project by team 3

# 1. Project Overview

본 프로젝트는 명시적(explicit) 학습 / 암시적(implicit) 학습 / In-Context Learning(ICL) 조건이
GPT2 기반 언어모델의 규칙 내재화 능력 및 일반화 능력에 미치는 영향을 비교하는 실험이다.

총 3개의 실험으로 구성된다:

1. E1 — Fine-tuning Efficiency

2. E2 — Grammaticality Judgment

  - BLiMP-style (문법성 판단, minimal pairs)

  - 5지선다 문법 판단

  - Surprisal Peak Plot (문법 오류 위치 localizing)

  - Prompting variation (explicit-card / explicit-explanation / implicit)

3. E3 — In-context Generalization (0,1,2,4-shot ICL)

# 2. Repository Structure

zip에서 확인한 구조를 기반으로 main 기준으로 이미 정리된 형태로 문서화했음.
feat# 폴더는 실험 중간 산출물이며, 최종 코드는 main/ 하위에 통합된다는 가정으로 정리.

main/
│
├── config/                      # 실험 설정 YAML
│   ├── base.yaml
│   ├── explicit.yaml
│   └── implicit.yaml
│
├── data/
│   ├── dataset.zip              # ENG / ArLang paired datasets
│   └── split.py                 # train/valid/test split + OOD generation
│
├── src/
│   ├── artlang_generator.py     # 인공언어(SOV) 생성기
│   ├── build_datasets.py        # Explicit / Implicit dataset builder
│   ├── create_pairs.py          # OK / Violation minimal pairs 생성
│   ├── dataloader.py            # PyTorch Dataset/Loader
│   ├── model.py                 # GPT2 기반 LM wrapper
│   ├── prompts.py               # Prompt templates (explicit/implicit)
│   ├── run_ft.py                # Fine-tuning 실행 (E1)
│   ├── run_eval_e2.py           # E2 BLiMP/PLL/5지선다/Surprisal
│   ├── run_icl.py               # E3 ICL 0–4 shot evaluation
│   └── utils.py                # 공통 함수 (tokenizer, logger, seed)
│
├── scripts/
│   ├── train.py                 # E1 학습 실행 스크립트
│   ├── evaluate_methods2.py     # E2 실행
│   ├── plot_learning_curves.py  # E1 PPL 그래프 생성
│   └── plot_surprisal.py        # E2 surprisal 시각화
│
└── utils/
    ├── metrics.py               # PLL, accuracy, surprisal 계산
    └── helpers.py               # 파일 처리, config loader

#️⃣ 3. Installation
conda create -n nlproj python=3.10
conda activate nlproj
pip install -r requirements.txt


주요 패키지:

PyTorch

Transformers

SentencePiece

matplotlib / seaborn

pandas

#️⃣ 4. Dataset Preparation
1) Artificial Language (ArLa)

Zipf 기반 어휘 샘플링

규칙 기반 SOV 생성

OK / Violation minimal pairs 생성

2) English (SVO)

Simple English Wikipedia 기반

SpaCy 의존구문 필터링

SVO 문장만 추출 후 minimal pair 생성

데이터 준비 실행
python src/build_datasets.py --config config/base.yaml
python src/create_pairs.py
python data/split.py

#️⃣ 5. Experiments
🔵 E1 — Fine-tuning Efficiency
목적

Explicit vs Implicit 학습 조건에서 GPT2가 얼마나 빠르고 안정적으로 규칙에 적응하는가(PPL 수렴) 비교.

실행
python scripts/train.py --config config/explicit.yaml
python scripts/train.py --config config/implicit.yaml

출력물

outputs/e1/logs/

outputs/e1/ppl_curves.png

🟣 E2 — Grammaticality Judgment (여러 버전 포함)
포함된 하위 실험

BLiMP-style (minimal pair PLL ranking)

5지선다 문법 판단

Surprisal Peak Plot — 문법 오류가 발생하는 지점의 surprisal 상승 체크

Prompt variation tuning

explicit-card (“규칙카드”)

explicit-explanation (“설명형”)

implicit (“문장만 제시”)

실행
python src/run_eval_e2.py

출력물

outputs/e2/accuracy.csv

outputs/e2/surprisal_plots/*.png

outputs/e2/multiple_choice_results.json

🟢 E3 — In-context Learning (0/1/2/4-shot)
목적

학습된 모델이 문맥만 보고 규칙을 추론할 수 있는지 확인.

실행
python src/run_icl.py --shots 0
python src/run_icl.py --shots 1
python src/run_icl.py --shots 2
python src/run_icl.py --shots 4

#️⃣ 6. Results Overview

(자세한 수치는 Evaluation Report에서 제공)

E1: explicit 조건은 초기 수렴이 빠르고, implicit 조건은 안정적이며 자연스러운 규칙 내재화 경향

E2: Prompting 제공 여부가 문법성 판단 정확도에 큰 영향

E3: implicit 학습 모델은 ICL에서 성능이 더 상승하는 경향

#️⃣ 7. Reproducibility

모든 실험은 seed 고정

config 파일 기반으로 실험 반복 가능

데이터 경로는 config/*.yaml에서 수정 가능

#️⃣ 8. Contributors (Team 3 / 가톨릭대학교)

📌 E1 / E2(implicit–explicit prompting) — 유진님

📌 E2(BLiMP / 5지선다 / Surprisal) — 한종님

📌 E1 / ArLa generation / infrastructure — 홍키쿠키쿠

📌 Code integration & documentation — 전원 기여

#️⃣ 9. License

MIT License (필요시)
