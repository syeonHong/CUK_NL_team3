# CUK_NL_team3
2025-2 Natural Language Processing term project by team 3


# 📘 프로젝트 개요서 (Team Shared Doc)

**프로젝트**: Explicit vs Implicit Learning in Language Models:
Understanding Learning Behavior & Personalization

**기간**: 2025.11.02 ~ 12.05

**출처**:

- Ma & Xu (2025), Implicit In-Context Learning: Evidence from Artificial Language Experiments

- OpenAI (2025), TrueReason: An Exemplar Personalized Learning System Integrating Reasoning with Foundational Models

## 🎯 1️⃣ 프로젝트 목적 (Project Goal)

본 프로젝트는 언어모델(ex: GPT-2 small) 을 대상으로
명시적(Explicit) 학습과 암시적(Implicit) 학습의 차이가
모델의 학습 효율·일반화 능력·문맥적 추론 능력(ICL) 에 어떤 영향을 미치는지를 실험적으로 검증하고,
그 결과를 기반으로 사용자 맞춤형 학습 추천 시스템 프로토타입을 제시하는 것이 목표입니다.

### 세부 목적

**[A] 명시적 vs 암시적 학습 비교 실험**

규칙 설명이 주어지는 경우(Explicit)와,
예시만 제공되는 경우(Implicit)에서 모델의 학습곡선(PPL), 정확도, 일반화 성능 비교

**[B-1] In-Context Learning(ICL) 검증**

학습된 모델이 프롬프트 예시(k-shot)만으로 새로운 문법 규칙을 추론할 수 있는지 확인

**[B-2] 개인화 추천 프로토타입 (선택적 확장)**

학습 성향(명시형/암시형/혼합형)을 기반으로
사용자에게 최적의 학습 모드를 추천하는 밴딧 알고리즘 기반 웹 데모 구현

## ⚙️ 2️⃣ 프로젝트 진행 구조 (How it works)
| 단계	| 주요 목표 |	입력  데이터 |	모델/도구 |	결과물 |
|------|------|--------|--------|--------|
| A. 학습 비교 실험	|Explicit vs Implicit fine-tuning|	인공어 + 영어 JSONL|	GPT-2 small|	PPL/Accuracy 그래프|
|B-1. ICL 검증|	0,1,2,4-shot 문맥 평가|	인공어|	GPT-2 small (freeze)|	ICL 곡선 (Accuracy vs k)|
|B-2. 개인화 추천 (웹 데모)|	학습자 유형 기반 모드 추천|	사용자 입력 로그	|Streamlit + Bandit 알고리즘	웹 시연| 앱 + 추천 결과 로그|
리포트/발표|	결과 통합·해석	|전체 실험 결과|	Word/PPT	|보고서 및 발표자료|

## 🧱 3️⃣ 세부 실험 설계
#### (1) 데이터 구조

- 형식: .jsonl (한 줄당 하나의 문장 샘플)

- 공통 Key

```json
{
  "id": "exp_0001",
  "type": "explicit",
  "prompt": "Rule: Subject-Object-Verb order. Example:",
  "text": "li neep lu vode klin noyka",
  "label": "ok",
  "meta": {
    "rule": "word_order",
    "language": "artificial"
  }
}
```

> Key	설명
- **id**	샘플 고유 ID
- **type**	explicit / implicit
- **prompt**	규칙 설명 문장 (implicit일 땐 비워둠)
- **text**	학습 문장
- **label**	정답(“ok”) / 위반(“violation”)
- **meta**	규칙, 언어 종류, 문장 길이 등 부가정보

- 데이터셋 구성

|구분	|언어	|샘플 수|	용도|
|train_explicit.jsonl|	인공어 + 영어|	~2000	|명시적 학습용|
|train_implicit.jsonl|	인공어 + 영어|	~2000	|암시적 학습용|
|test.jsonl|	인공어|	~500	|평가 (문법성/일반화)|

#### (2) 모델 구조 (예시)

- 모델: GPT-2 small (pretrained)

> Fine-tuning 조건
- Optimizer: AdamW
- Learning rate: 5e-5
- Weight decay: 1e-2
- Epochs: 8~10
- Scheduler: cosine

> 평가 지표
- Perplexity (PPL)
- Grammaticality Accuracy
- Generalization (unseen combinations)

#### (3) ICL 평가 (B-1)
|조건	|프롬프트 예시	|목표|
|---|---|---|
|0-shot	|문법성 판단만 요청	|baseline|
|1-shot	|예시 1개	|context 학습 시작|
|2-shot	|예시 2개	|문맥 패턴 강화|
|4-shot	|예시 4개	|구조적 일반화 확인|

>예시 프롬프트 예:

li neep lu vode klin → grammatical

li neep klin lu vode → ungrammatical

Test: lu vode li neep klin → ?

#### (4) 개인화 추천 시스템 (B-2)

- 입력: 학습자 진단 결과 (정답률, 반응시간, 오류유형)

- 유형 분류: 규칙형 / 문맥형 / 혼합형 / 불안정형

- 추천 알고리즘: Thompson Sampling Bandit

reward = 0.5 * Δaccuracy + 0.3 * Δgeneralization - 0.2 * Δreaction_time


> UI 구현: Streamlit

- 명시/암시 학습 모드 선택 버튼

- 실시간 보상 변화 그래프

- 사용자 학습 유형 표시

## 📅 4️⃣ 프로젝트 일정 및 담당자
|기간|	파트|	담당자|	주요 산출물|
|--|--|--|--|
|~11/05|	데이터 구축	재형	|인공어 생성기 + 영어 JSONL 세트
|~11/15|	모델 학습(A)	|한종|	GPT-2 explicit/implicit fine-tuning 결과|
|~11/22|	ICL 실험(B-1)	|주은|	k-shot 곡선, ICL 일반화 분석|
|~11/29|	웹 데모(B-2)	|유진|	Streamlit + Bandit 데모|
|~12/05|	총괄·리포트·발표	|승연|	보고서, 발표자료, 전체 통합|
## 📊 5️⃣ 예상 결과물 및 파일 구조
```json
```text
project/
├── data/
│   ├── artificial/
│   │   ├── train_explicit.jsonl
│   │   ├── train_implicit.jsonl
│   │   └── test.jsonl
│   └── english/
│       ├── train_explicit_en.jsonl
│       └── train_implicit_en.jsonl
│
├── models/
│   ├── gpt2_explicit/
│   │   └── best.pth
│   └── gpt2_implicit/
│       └── best.pth
│
├── src/
│   ├── data_gen.py
│   ├── train.py
│   ├── eval.py
│   ├── icl_eval.py
│   ├── bandit_demo.py
│   └── app_streamlit.py
│
├── results/
│   ├── ppl_curve.png
│   ├── icl_curve.png
│   └── eval_summary.csv
│
└── report/
    ├── final_report.docx
    └── slides.pptx
```

## 🧩 6️⃣ 기대 효과

- 실험적:

명시/암시 학습 비교를 통해 AI 언어모델의 학습 전략 차이를 실증적으로 검증

ICL이 “언어 규칙 내재화”와 유사하게 작동함을 보여줌

- 응용적:

개인화 학습 추천(명시적 vs 암시적)을 적용한 AI 튜터 설계 근거 제시

실험 연구를 실제 AI 서비스 프로토타입으로 확장

## ✅ 요약 문장 

우리 프로젝트는 언어모델의 학습 방식(Explicit vs Implicit) 이
언어 일반화 및 문맥 추론(ICL) 에 어떤 차이를 만드는지를 검증하고,
그 차이를 활용해 학습자 맞춤형 AI 학습 추천 시스템으로 확장하는 것을 목표로 한다.

