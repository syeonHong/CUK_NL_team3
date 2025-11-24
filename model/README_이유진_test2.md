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
      - 세 모델 모두 loss와 lr이 안정적으로 수렴한다.
      - Explict A가 Explicit B에 비해 더 loss가 가파르게 감소함을 확인할 수 있다. 
      Explicit A PPL : 0.939271  >  Explicit B PPL 0.912281 


3. E2 실험 - Grammaticality Judgment (PLL Accuracy)

   ## 사용 데이터셋
   - Explicit (명시적 학습) : test_explicit.jsonl
   - Implicit (암시적 학습) : test_implicit.jsonl
   - 실행 경로 : C:\Users\User\PycharmProjects\CUK_NL_team3\scripts\eval.py
   ## 실험 목적
   - 명시적/암시적 조건에서 학습한 모델의
   - 일반화 성능 · 문법 판별 능력을 확인한다.

   ## 1) PPL ACCURACY
   - 전체 쌍 중 정답 비율을 Accuracy (ACC) 로 계산한다.
   - 실험 결과
        explicit : 0.939271 
        implicit : 0.940621 
   - 분석 : 두 모델 모두 ACC가 약 0.94로 매우 높으며, 문법성 쌍(OK/Violation)을 거의 완벽하게 구분했다.

   ## 2) MEAN_PPL_GAP
   - 모델이 정상 문장과 위반 문장 사이에서 가지는 확신(PLL log-likelihood)의 평균 차이를 계산한다.

   - 실험 결과
        explicit : 0.946057 
        implicit : 0.990131
   - 분석 : Implicit 모델이 더 높은 PLL Gap을 기록하여, 정문과 비문을 더 확실하게 구분하는 경향을 보인다. 암시적 학습 모델이 규칙을 직접 제공받지 않았음에도, 문장 패턴을 보다 자연스럽게 내재화했음을 시사한다.

   ## 3) ECE
   - Expected Calibration Error 측정.
   - 실험 결과
        explicit : 0.225057
        implicit : 0.220281
   -  분석 : Implicit 모델이 더 낮은 ECE를 보이며, 예측 확률의 신뢰도(calibration)가 더 안정적이다. 두 값의 차이는 크지 않지만, 암시적 학습 모델이 확률 출력의 일관성 측면에서 약간 더 우수한 경향을 보인다.

## 최종 결과 
- 현재 JSONL 파일은 데이터 구조 자체가 structure, meta 딕셔너리에서 보여주는 두개의 문법 규칙 밖에 없기에 근거 있는 규칙카드를 두개 이상 생성하기 어려운 한계점이 존재했다. 
- E1에서는 Explicit 모델이 규칙 카드를 기반한 학습이 효율 측면에서 더 우세함을 확인할 수 있었다. 
- 그러나 E2에서 실제 문법성 판단을 평가한 결과,Implicit 학습이 더 큰 PPL GAP과 더 낮은 ECE를 보여 Explicit 모델보다 높은 효율을 보였다
- 이는 명시적 규칙 제공이 모델을 특정 패턴에 과도하게 고정시켜 일반화 성능을 낮게하는 반면, 암시적 학습은 데이터 분포에서 규칙을 스스로 탐색해 더 자연스러운 문법 판단 능력을 형성함을 의미한다. 
