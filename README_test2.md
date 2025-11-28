모델 파트는 명시적 학습(Explicit) 과 암시적 학습(Implicit) 조건에서
GPT 기반 언어모델이 문법 규칙을 어떻게 내재화하고 일반화하는지를 비교하기 위해 수행되었다.
실험은 E1 (Fine-tuning Efficiency) 와 E2 (Grammaticality Judgment) 두 단계로 구성된다.

1. 데이터 나누기 
   ## 사용 데이터셋
    - Explicit (명시적 학습) : train_explicit.jsonl, val_explicit.jsonl
   ## 실험 목적 
    - 동일한 GPT-2 아키텍처에서 explicit/implicit 조건별로 각각 fine-tuning을 수행한다.
    - 두 조건 간 PPL(perplexity) 하락 속도와 수렴 패턴을 비교한다.
   두 파일을 각각 train : valid : test = 8 : 1 : 1 비율로 분할하여 저장하였다.

2. E1 실험 - Fine-Tuning & PPL 
   ## 사용 데이터셋
   - Explicit (명시적 학습) : train_explicit.jsonl, val_explicit.jsonl
   - Implicit (암시적 학습) : train_implicit.jsonl, val_implicit.jsonl
   - 학습 경로 : C:\Users\User\PycharmProjects\CUK_NL_team3\scripts\train.py
   
   ## 실험 목적
      - explicit 학습에 대해서는 두가지 방법을 사용해서 비교했다.
         - A. 기술형 규칙카드 (SOV 조건 고정 +  -ka 부착 조건 명시)롤 통해 모델이 규칙을 직접 암기하도록 유도 
         - B. 설명형 규칙카드 (자연언어 기반으로 서술형 설명 명시)를 통해 규칙을 이해하도록 유도
      - 두 조건 간의 PPL(perplexity) 하락 속도와 수렴 패턴을 비교한다. 
   
      ## 결과 확인
      - 학습 로그 및 곡선은 TensorBoard에서 시각화한다.
      - 결과 그래프 저장 경로: C:\Users\User\PycharmProjects\CUK_NL_team3\E1_aria_test2

      ## 일차적 파인튜닝 내용 요약
      explicit_a_gpt2  - AVERAGE PPL (OK):        75.87561461595678
      explicit_a_gpt2  - AVERAGE PPL (violation): 85.18195000460598
      explicit_b_gpt2  - AVERAGE PPL (OK):        8.972814093168509
      explicit_b_gpt2  - AVERAGE PPL (violation): 9.848269220502797
      - OK 문장 기준으로 perplexity가 더 낮은 모델: explicit_b_gpt2 임을 확인할 수 있다.


3. E2 실험 - Grammaticality Judgment (PLL Accuracy)
   ## 사용 데이터셋
   - Explicit (명시적 학습) : test_arla.jsonl
   - Implicit (암시적 학습) : test_arla.jsonl
   - 실행 경로 : C:\Users\User\PycharmProjects\CUK_NL_team3\scripts\eval.py
   ## 실험 목적
   - 명시적/암시적 조건에서 학습한 모델의
   - 일반화 성능 · 문법 판별 능력을 확인한다.

   ## 1) AVERGE PPL 
   - 문장이 규칙에 맞을 때 모델이 더 낮은 불확실성(PPL)로 일관되게 평가
   - 실험 결과
      explicit_b : 
         AVERAGE PPL (OK): 8.972814 
         AVERAGE PPL (violation): 9.848269
      implicit : 
         AVERAGE PPL (OK): 2259.769116
         AVERAGE PPL (violation): 2835.644082
   - 분석 : explicit_b 학습에서는  OK < violation 차이가 안정적으로 보여 규칙 기반 학습 효과가 나타나지만, implicit 학습은 두 값이 매우 크고 불안정하여 규칙을 사용하지 못하고 문장 패턴을 제대로 일반화하지 못하고 있다. 
   ## 2) MEAN_PLL_GAP
   - 동일한 OK/violation 쌍 안에서 모델이 정문을 비문보다 얼마나 일관되게 더 높은 확률로 선호하는지를 측정
   - 실험 결과
      explicit_b : 0.093096
      implicit : 0.227006
   - 분석 : Mean_PLL_Gap에서 implicit 학습에 따른 실험 결과 값이 이 더 크게 나타났지만, 전반적 확률 안정성이 낮아 실제 선호 마진으로 보기 어렵다.
   ## 3) ECE
   - Expected Calibration Error 측정.
   - 실험 결과
      explicit_b : 0.293618
      implicit : 0.258584 
  -  분석 : 두 모델 모두 calibration은 좋은 편이 아니며, implicit이 약간 더 낮은 ECE를 보이지만, 전반적으로 확률 예측이 매우 확신 없어서 생기는 효과로 보인다. 

   ## 최종 결과 
   - 전반적으로 explicit_b 학습이 우수하다.
   - 프롬프트에서 "설명형 규칙카드 B"를 통해 explicit_b이 단순 규칙나열이 아닌 모델이 규칙을 텍스트 형태로 이해하기 좋은 형태로 전달되었기 때문이다.
   - 반면 implicit 학습은 규칙 자체가 없이 패턴만 보고 문장 틀림 유무를 추측하기 때문에 비교적 규칙을 내재화하기 어려워서 다음과 같은 실험 결과가 나왔다고 도출할 수 있다.