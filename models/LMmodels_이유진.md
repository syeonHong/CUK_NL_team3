모델 파트는 명시적 학습(Explicit) 과 암시적 학습(Implicit) 조건에서
GPT 기반 언어모델이 문법 규칙을 어떻게 내재화하고 일반화하는지를 비교하기 위해 수행되었다.
실험은 E1 (Fine-tuning Efficiency) 와 E2 (Grammaticality Judgment) 두 단계로 구성된다.

1. 데이터 나누기 
   ## 데이터 경로 : C:\Users\User\PycharmProjects\CUK_NL_team3\data\split_data
   ## 원본 데이터 : train_aria_explicit.jsonl, train_aria_implicit.jsonl
   두 파일을 각각 train : valid : test = 8 : 1 : 1 비율로 분할하여 저장하였다.

2. E1 실험 - Fine-Tuning & PPL 
   ## 사용 데이터셋
   - Explicit (명시적 학습) : train_explicit.jsonl, val_explicit.jsonl
   - Implicit (암시적 학습) : train_implicit.jsonl, val_implicit.jsonl
   - 학습 경로 : C:\Users\User\PycharmProjects\CUK_NL_team3\scripts\train.py
   
   ## 실험 목적
    동일한 GPT-2 아키텍처에서 explicit/implicit 조건으로 각각 fine-tuning을 수행하고,
    두 조건 간의 PPL(perplexity) 하락 속도와 수렴 패턴을 비교한다.
   
   ## 결과 확인
    학습 로그 및 곡선은 TensorBoard에서 시각화한다. 
    결과 그래프 저장 경로: C:\Users\User\PycharmProjects\CUK_NL_team3\E1_aria

   ## 일차적 파인튜닝 내용 요약
   - 명시적 학습 모델(EXPLICIT)은 규칙 설명이 포함된 텍스트를 사용하여 빠르고 효율적으로 수렴하고 있다.
   - 암시적 학습 모델(IMPLICIT)은 규칙 없이 예시 노출만으로 완만하게 수렴하며 안정적 학습 경향을 보이고 있다.

3. E2 실험 - Grammaticality Judgment (PLL Accuracy)

   ## 사용 데이터셋
   - Explicit (명시적 학습) : test_explicit.jsonl
   - Implicit (암시적 학습) : test_implicit.jsonl
   - 실행 경로 : C:\Users\User\PycharmProjects\CUK_NL_team3\scripts\eval.py

   ## 실험 목적 
    E1에서 fine-tuning된 두 모델(EXPLICIT / IMPLICIT)을 대상으로
    Pseudo-Log-Likelihood (PLL) 기반 문법성 판단을 수행한다.
    각 문장에서 모델이 문법적 구조를 얼마나 정확히 내재화했는지를 평가한다.
   
   ## 1. PPL ACCURACY
   - 각 문장 쌍(정문 vs 비문)에 대해 PLL(OK) > PLL(Violation) 이면 정답으로 처리.
   - 전체 쌍 중 정답 비율을 Accuracy (ACC) 로 계산한다.
     - 실험 결과
       explicit : ACC:1.000000 
       implicit : ACC:1.000000 
   - 분석 : 두 모델 모두 ACC = 1.00으로 나타났지만, 이는 데이터셋 규모가 작아 과적합(overfitting) 된 결과로 해석된다.
   
   ## 2. MEAN_PPL_GAP
   - 문법적(Grammatical) 문장과 비문법적(Ungrammatical) 문장 간의 평균 PLL 차이를 나타내며, 값이 클수록 문법성 구분이 강함을 의미한다.
     - 실험 결과
     explicit : MEAN_PLL_GAP:0.773172 
     implicit : MEAN_PLL_GAP:1.021748
   - 분석 : implicit 모델이 더 큰 PLL Gap을 보여 정문/비문 구분력이 더 강함을 확인할 수 있다.
   
  ## 3. ECE 
  - 모델의 **확률 신뢰도(Calibration)**를 평가하는 지표이며, 값이 낮을수록 모델의 확률 출력이 안정적임을 의미한다.
    - 실험 결과
     explicit : ECE:0.322809
     implicit : ECE:0.270597
   - 분석 : 암시적 학습(IMPLICIT) 모델이 더 안정적인 확률 분포를 보이며, 문법적 규칙을 내재적으로 더 잘 학습했음을 시사한다.

## 최종 결과 
1. E1에서는 explicit (명시적 학습)이 더 빠르고 낮은 PPL로 수렴하여 학습 효율이 높음을 보여줌을 확인했다. 
2. E2에서는 implicit (암시적 학습)에서 더 큰 PPL_GAP과 낮은 ECE를 보이면서 규칙을 더 안정적으로 내재화하고 문법 구분력이 강함을 확인했다.
3. 두 문장 모두 1.0으로 데이터 규모의 한계로 오버피팅 가능성을 확인했다 