1. 텐서보드 저장은 log 파일 아래 explicit,implict 학습 별로 train,eval 결과 있습니다
2. plot_learning_curves.py 돌린 내용은 results 파일 안에 그래프에 나와있습니다
3. evaluate_methods2.py 돌린 결과 cmd 내용은 log 파일 안에 explicit,implict 학습 별로 있습니다.
evaluate_methods2.py 돌릴때 출력되는 grammer_error_detection 내용도 log 파일 안에 있습니다.
예시 문장으로는 test_arla.jsonl의 첫번째 예시문장을 이용해서 정합, 비정합으로 조건 맞추어서 넣었습니다.


→ OK 구조: S-O-PP(optional)-V-ADV(optional)
→ VIOLATION: S-V-O-PP-ADV 

good_sent = "zuna blono drako brulo troiso troiso lu brip troise neime troise li droz ko plom lu"
bad_sent = "zuna blono drako brulo troiso troiso lu droz brip troise neime troise ko plom li"

## explicit 
[Method 1] BLiMP Style (Minimal Pair Comparison)
  Option A (Correct): 'zuna blono drako brulo troiso troiso lu brip troise neime troise li droz ko plom lu'
     -> Loss: 3.1821 | PPL: 24.10
  Option B (Wrong)  : 'zuna blono drako brulo troiso troiso lu droz brip troise neime troise ko plom li'
  ✅ 결과: 모델이 '올바른 문장'을 더 자연스럽게 판단했습니다. (정답)

[Method 2] Multiple Choice Ranking (by PPL)
  Candidates:
    1. zuna blono drako brulo troiso troiso lu brip troise neime troise li droz ko plom lu (PPL: 24.10)
    2. zuna blono drako brulo troiso troiso lu droz brip troise neime troise ko plom li (PPL: 35.20)
    3. lu zuna blono drako brulo troiso troiso droz brip troise neime troise ko plom li (PPL: 102.30)
    4. zuna blono drako brulo troiso troiso droz brip troise neime troise li lu ko plom (PPL: 65.27)
    5. zuna blono drako brulo troiso troiso droz brip troise neime troise ko plom li lu (PPL: 157.92)
  => 모델의 선택: 1번 문장
  ✅ 정답입니다!

## implicit
[Method 1] BLiMP Style (Minimal Pair Comparison)
  Option A (Correct): 'zuna blono drako brulo troiso troiso lu brip troise neime troise li droz ko plom lu'
     -> Loss: 1.4876 | PPL: 4.43
  Option B (Wrong)  : 'zuna blono drako brulo troiso troiso lu droz brip troise neime troise ko plom li'
     -> Loss: 1.7840 | PPL: 5.95
  ✅ 결과: 모델이 '올바른 문장'을 더 자연스럽게 판단했습니다. (정답)

[Method 2] Multiple Choice Ranking (by PPL)
  Candidates:
    1. zuna blono drako brulo troiso troiso lu brip troise neime troise li droz ko plom lu (PPL: 4.43)
    2. zuna blono drako brulo troiso troiso lu droz brip troise neime troise ko plom li (PPL: 5.95)
    3. lu zuna blono drako brulo troiso troiso droz brip troise neime troise ko plom li (PPL: 34.99)
    4. zuna blono drako brulo troiso troiso droz brip troise neime troise li lu ko plom (PPL: 17.54)
    5. zuna blono drako brulo troiso troiso droz brip troise neime troise ko plom li lu (PPL: 39.91)
  => 모델의 선택: 1번 문장
  ✅ 정답입니다!
