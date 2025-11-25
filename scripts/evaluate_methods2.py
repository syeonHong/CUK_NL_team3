import torch, os
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# [설정] 프롬프트 및 모델 경로
# ==============================================================================
# 학습할 때 데이터 앞에 "Sentence: "가 붙었으므로, 평가할 때도 붙여주는 것이 성능에 좋습니다.
PREFIX_PROMPT = "Sentence: "

# 경로가 없으면 자동으로 'gpt2'를 다운로드하여 실행합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(current_dir, "..", "logs", "explicit_gpt2", "final_model")


# ==============================================================================
# [준비] 모델 로딩 및 유틸리티
# ==============================================================================
def load_model(path):
    print(f"Loading model from {path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path)
    except:
        print(f"Warning: 경로({path})를 찾을 수 없어 기본 'gpt2' 모델을 로드합니다.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    return model, tokenizer


def calculate_loss_and_ppl(model, tokenizer, text):
    """문장 전체의 Loss와 PPL을 계산"""
    full_text = PREFIX_PROMPT + text
    inputs = tokenizer(full_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        ppl = math.exp(loss)
    return loss, ppl


# ==============================================================================
# [Method 1] BLiMP 스타일: Minimal Pair 평가 (핵심 지표)
# ==============================================================================
def evaluate_blimp_pair(model, tokenizer, good_sent, bad_sent):
    print(f"\n[Method 1] BLiMP Style (Minimal Pair Comparison)")

    loss_good, ppl_good = calculate_loss_and_ppl(model, tokenizer, good_sent)
    loss_bad, ppl_bad = calculate_loss_and_ppl(model, tokenizer, bad_sent)

    print(f"  Option A (Correct): '{good_sent}'")
    print(f"     -> Loss: {loss_good:.4f} | PPL: {ppl_good:.2f}")

    print(f"  Option B (Wrong)  : '{bad_sent}'")
    print(f"     -> Loss: {loss_bad:.4f} | PPL: {ppl_bad:.2f}")

    # Loss가 낮을수록 모델이 '자연스럽다'고 느끼는 문장입니다.
    if loss_good < loss_bad:
        print("  ✅ 결과: 모델이 '올바른 문장'을 더 자연스럽게 판단했습니다. (정답)")
        return True
    else:
        print("  ❌ 결과: 모델이 '틀린 문장'을 더 좋아합니다. (오답)")
        return False


# ==============================================================================
# [Method 2] 5지선다 랭킹 (PPL Ranking)
# ==============================================================================
def evaluate_mcq_ranking(model, tokenizer, options, correct_idx):
    print(f"\n[Method 2] Multiple Choice Ranking (by PPL)")

    results = []
    print("  Candidates:")
    for i, opt in enumerate(options):
        loss, ppl = calculate_loss_and_ppl(model, tokenizer, opt)
        results.append((i, opt, ppl))
        print(f"    {i + 1}. {opt} (PPL: {ppl:.2f})")

    # PPL이 낮은 순서대로 정렬
    results.sort(key=lambda x: x[2])

    best_option = results[0]
    best_idx = best_option[0]

    print(f"  => 모델의 선택: {best_idx + 1}번 문장")

    if best_idx == correct_idx:
        print("  ✅ 정답입니다!")
    else:
        print(f"  ❌ 틀렸습니다. (정답은 {correct_idx + 1}번)")


# ==============================================================================
# [Method 3] Surprisal Plot (문법 오류 위치 시각화)
# ==============================================================================
def plot_surprisal(model, tokenizer, sentence, title="Surprisal Plot"):
    print(f"\n[Method 3] Generating Surprisal Plot for: '{sentence}'")

    full_text = PREFIX_PROMPT + sentence
    inputs = tokenizer(full_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits

    # Shift: 예측값(logits)과 정답(labels)의 위치 맞추기
    # logits[0, :-1]은 0번째부터 n-1번째 토큰에 대한 예측
    # labels[0, 1:] 은 1번째부터 n번째 토큰 (실제 정답)
    shift_logits = logits[0, :-1, :].contiguous()
    shift_labels = inputs["input_ids"][0, 1:].contiguous()

    # 토큰별 Loss 계산 (reduction='none'으로 개별 loss 획득)
    loss_fct = CrossEntropyLoss(reduction='none')
    token_losses = loss_fct(shift_logits, shift_labels)

    # 토큰 문자열로 변환 (그래프 X축용)
    tokens = tokenizer.convert_ids_to_tokens(shift_labels)
    losses = token_losses.cpu().numpy()

    # 그래프 그리기
    plt.figure(figsize=(12, 6))
    plt.plot(losses, marker='o', linestyle='-', color='r', label='Surprisal (Loss)')

    # X축 레이블 설정 (토큰이 너무 많으면 겹치므로 처리)
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.xlabel('Tokens')
    plt.ylabel('Loss (Surprisal)')
    plt.title(f"{title}\nSentence: {sentence}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("  -> 그래프가 팝업으로 표시되었습니다.")


# ==============================================================================
# [실행부] 여기서 테스트할 문장을 수정하세요
# ==============================================================================
if __name__ == "__main__":
    # 1. 모델 로드
    model, tokenizer = load_model(MODEL_PATH)

    # --- 테스트 데이터 ---

    # Case A: 기본적인 SVO 어순
    good_sent = "The dog eats the bone."
    bad_sent = "Eats the dog the bone."

    # Case B: 5지선다 보기
    mcq_options = [
        "Eats the dog the bone.",
        "The dog eats the bone.",
        "Bone the dog eats the.",
        "The eats bone dog.",
        "Dog the bone eats."
    ]# 정답 (인덱스 1)
    correct_option_index = 1  # 0부터 시작하므로 2번째 보기

    # --- 3가지 평가 실행 ---

    # 1. BLiMP 평가 (맞는 문장 vs 틀린 문장 승부)
    evaluate_blimp_pair(model, tokenizer, good_sent, bad_sent)

    # 2. 5지선다 평가 (가장 PPL이 낮은 문장 찾기)
    evaluate_mcq_ranking(model, tokenizer, mcq_options, correct_option_index)

    # 3. Surprisal 그래프 (틀린 문장에서 어디가 이상한지 시각화)
    # 틀린 문장을 넣어야 그래프가 치솟는(Spike) 것을 볼 수 있어 재밌습니다.
    plot_surprisal(model, tokenizer, bad_sent, title="Grammar Error Detection")