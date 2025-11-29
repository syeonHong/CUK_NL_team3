# CUK_NL_team3
2025-2 Natural Language Processing term project by team 3

**NOTE: 중간중간 결과 Graph 삽입 예정 **


## #️⃣ 1. Project Overview

본 프로젝트는 GPT2 기반 언어모델을 대상으로
명시적(explicit) / 암시적(implicit) 학습 방식 / ICL(few-shot) 간
규칙 내재화 및 일반화 능력의 차이를 분석하는 연구이다.

총 3개의 메인 실험:

- E1 — Fine-tuning Efficiency
- E2 — Grammaticality Judgment (여러 버전 포함)
- E3 — In-context Learning (0–4-shot)



## #️⃣ 2. Repository Structure
```
main/
│
├── config/ # 실험 구성 YAML
├── data/ # dataset zip 및 split 코드
├── src/
│ ├── artlang_generator.py # 인공언어 생성기
│ ├── build_datasets.py # ENG/SOV explicit/implicit dataset 생성
│ ├── create_pairs.py # OK/Violation minimal pairs 생성
│ ├── dataloader.py # PyTorch Dataset
│ ├── model.py # GPT2 wrapper model
│ ├── prompts.py # explicit/implicit prompt card
│ ├── run_ft.py # E1 파인튜닝 코드
│ ├── run_eval_e2.py # E2(BLiMP, 5-choice, surprisal)
│ ├── run_icl.py # E3 ICL 코드
│ └── utils.py # 공통 함수
│
├── scripts/ # 그래프, 결과 정리 스크립트
└── utils/ # metrics 및 helper
```

## #️⃣ 3. File Description

### Dataset 관련
- build_datasets.py — 영어(SVO)·인공언어(SOV) 기반 explicit/implicit 데이터 구성  
- create_pairs.py — OK vs Violation minimal pairs 생성  
- split.py — train/dev/test + OOD split  

### Model 관련
- model.py — GPT2 LM + loss 계산  
- prompts.py — explicit-card, explicit-explanation, implicit prompt template  
- dataloader.py — tokenization & batch dataloader  

### Experiment 코드
- run_ft.py — E1 fine-tuning 로직  
- run_eval_e2.py — E2(문법성 판단) 모든 버전 포함  
- run_icl.py — E3(0/1/2/4-shot ICL)  

### 시각화
- plot_learning_curves.py  
- plot_surprisal.py  



## #️⃣ 4. Experiment Structure

### 🔵 E1 — Fine-tuning Efficiency
- explicit vs implicit vs SVO vs SOV 조건 간 PPL 수렴 비교  
- output: loss/logs, PPL curves  

### 🟣 E2 — Grammaticality Judgment
하위 구성:
- BLiMP-style PLL ranking  
- 5-choice 문법 판단  
- Surprisal Peak Plot  
- Prompt variation (explicit-card, explicit-explanation, implicit)  
- output: accuracy, ΔPLL, surprisal 시각화  

### 🟢 E3 — ICL (0–4 shot)
- 학습 없이 few-shot 문맥만으로 규칙 추론하는지 평가  
- output: shot별 accuracy curve  



## #️⃣ 5. How Components Connect

- build_datasets.py → create_pairs.py → data/split.py  
- dataloader.py + model.py  
- E1/E2/E3 실행 코드가 위 빌딩 블록을 조합  
- scripts/ 폴더가 결과를 정리·시각화  



## #️⃣ 6. Team Members (Team 3 — 가톨릭대학교)

- Dataset / ArLa generation: 류재형
- E1 / E2 (ArLa): 이유진
- E1 / E2 (Eng): 최한종
- E3 (ArLa) : 장주은
- 통합 / 문서화 / 구조화: 홍승연
