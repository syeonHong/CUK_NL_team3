import torch, os, json
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import matplotlib.pyplot as plt
import numpy as np
from src.prompts import EXPLICIT_RULE_CARD, IMPLICIT_EXAMPLES

# ==============================================================================
# [설정] 프롬프트 및 모델 경로
# ==============================================================================
# 학습할 때 데이터 앞에 "Sentence: "가 붙었으므로, 평가할 때도 붙여주는 것이 성능에 좋습니다.
PREFIX_PROMPT = "Sentence: "
CONDITION = "implicit" #"explicit" || "implicit"
if CONDITION == "explicit":
    model = "explicit_gpt2"
else:
    model = "implicit_gpt2"

# 경로가 없으면 자동으로 'gpt2'를 다운로드하여 실행합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(current_dir, "..", "logs", model, "final_model")


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


# 1) condition에 맞는 프롬프트를 붙임
# 2) 프롬프트 영역은 Loss 계산에서 제외(Masking)하여 순수 문장 PPL만 측정
def calculate_loss_and_ppl(model, tokenizer, text):
    """문장 전체의 Loss와 PPL을 계산 (프롬프트 제외)"""

    # 1. 조건에 따른 전체 텍스트 구성
    if CONDITION == "explicit":
        full_text = f"{EXPLICIT_RULE_CARD}\n\nSentence: {text}"
    else:
        # Implicit 모델 학습 방식에 따라 수정 가능 (예: 예시 포함 or 미포함)
        # 여기서는 학습 데이터와 동일하게 예시를 포함한다고 가정
        full_text = f"{IMPLICIT_EXAMPLES}\n\nSentence: {text}"
        # 만약 "Sentence: "만 넣고 학습했다면 아래 코드 사용:
        # full_text = f"Sentence: {text}"

    inputs = tokenizer(full_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # 2. Labels 생성 및 마스킹 (Prompt 부분 제외)
    labels = inputs["input_ids"].clone()

    # "Sentence: " 패턴을 찾아 그 이전(프롬프트)은 -100으로 마스킹
    if "Sentence: " in full_text:
        # 텍스트 기반으로 프롬프트 부분 길이 추정
        prompt_part = full_text.split("Sentence: ")[0] + "Sentence: "
        prompt_ids = tokenizer(prompt_part)["input_ids"]
        prompt_len = len(prompt_ids)

        # 텐서 크기 안전장치
        if prompt_len < labels.shape[1]:
            labels[0, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.item()
        # Loss가 너무 크면 overflow 방지
        try:
            ppl = math.exp(loss)
        except OverflowError:
            ppl = float('inf')

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

    if CONDITION == "explicit":
        full_text = f"{EXPLICIT_RULE_CARD}\n\nSentence: {sentence}"
    else:
        full_text = f"{IMPLICIT_EXAMPLES}\n\nSentence: {sentence}"

    inputs = tokenizer(full_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits

    # Shift logits & labels
    shift_logits = logits[0, :-1, :].contiguous()
    shift_labels = inputs["input_ids"][0, 1:].contiguous()

    # Token-wise Loss
    loss_fct = CrossEntropyLoss(reduction='none')
    token_losses = loss_fct(shift_logits, shift_labels)

    # Convert to tokens
    tokens = tokenizer.convert_ids_to_tokens(shift_labels)
    losses = token_losses.cpu().numpy()

    # "Sentence:" 이후의 토큰들만 잘라서 보여줌
    target_start_idx = 0
    for i, token in enumerate(tokens):
        # GPT2 토크나이저 특성상 ":"나 " Sentence" 등이 나뉠 수 있어 대략적인 위치 파악
        # 여기서는 단순화를 위해 전체를 출력하되, 너무 길면 뒤쪽(실제 문장) 위주로 봄
        pass

    # 그래프 그리기
    plt.figure(figsize=(12, 6))
    plt.plot(losses, marker='o', linestyle='-', color='r', label='Surprisal (Loss)')

    # X축 레이블 설정 (토큰이 너무 많으면 겹치므로 처리)
    plt.figure(figsize=(12, 6))
    plt.plot(losses, marker='o', linestyle='-', color='r', label='Surprisal (Loss)')

    # X축 레이블 설정 (너무 많으면 간격 조정)
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.xlabel('Tokens')
    plt.ylabel('Loss (Surprisal)')
    plt.title(f"{title} ({CONDITION})\nSentence: {sentence}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("  -> 그래프가 팝업으로 표시되었습니다.")


def evaluate_test_file(model, tokenizer, test_path):
    print(f"\n[Final Evaluation] Running on Test File: {test_path}")
    print(f"Condition: {CONDITION}")

    ok_ppls = []
    viol_ppls = []

    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                ex = json.loads(line)
            except:
                continue

            label = ex.get("label")
            text = ex.get("text")

            if not text or not label: continue

            # PPL 계산
            loss, ppl = calculate_loss_and_ppl(model, tokenizer, text)

            if label == "ok":
                ok_ppls.append(ppl)
            elif label == "violation":
                viol_ppls.append(ppl)

    # 결과 요약
    if not ok_ppls and not viol_ppls:
        print("⚠️ Warning: 평가할 데이터가 없습니다. JSON 파일의 형식이나 경로를 확인하세요.")
        return

    avg_ok_ppl = sum(ok_ppls) / len(ok_ppls) if ok_ppls else 0
    avg_viol_ppl = sum(viol_ppls) / len(viol_ppls) if viol_ppls else 0

    print("-" * 40)
    print(f"Total 'ok' sentences: {len(ok_ppls)}")
    print(f"Total 'violation' sentences: {len(viol_ppls)}")
    print("-" * 40)
    print(f"Average PPL (OK)       : {avg_ok_ppl:.2f}")
    print(f"Average PPL (Violation): {avg_viol_ppl:.2f}")

    if avg_ok_ppl < avg_viol_ppl:
        print("\n✅ Success: 모델이 평균적으로 '맞는 문장'을 더 자연스럽게(낮은 PPL) 판단합니다.")
    else:
        print("\n❌ Fail: 모델이 '틀린 문장'을 더 선호하거나 구분을 못합니다.")

# ==============================================================================
# [실행부] 여기서 테스트할 문장을 수정하세요
# ==============================================================================
if __name__ == "__main__":
    # 1. 모델 로드
    model, tokenizer = load_model(MODEL_PATH)

    # --- 테스트 데이터 ---

    # Case A: 기본적인 SVO 어순
    good_sent = "The boy kicks the ball."  # SVO (Correct)
    bad_sent = "Kicks the boy the ball."  # VSO (Wrong)

    # Case B: 5지선다 보기
    mcq_options = [
        "Kicks the boy the ball.",
        "The boy kicks the ball.",
        "Ball the boy kicks the.",
        "The kicks ball boy.",
        "Boy the ball kicks."
    ]
    correct_option_index = 1  # 0부터 시작하므로 2번째 보기


    test_file_path = "data/test_eng.jsonl"
    evaluate_test_file(model, tokenizer, test_file_path)

    # --- 3가지 평가 실행 ---

    # 1. BLiMP 평가 (맞는 문장 vs 틀린 문장 승부)
    evaluate_blimp_pair(model, tokenizer, good_sent, bad_sent)

    # 2. 5지선다 평가 (가장 PPL이 낮은 문장 찾기)
    evaluate_mcq_ranking(model, tokenizer, mcq_options, correct_option_index)

    # 3. Surprisal 그래프 (틀린 문장에서 어디가 이상한지 시각화)
    # 틀린 문장을 넣어야 그래프가 치솟는(Spike) 것을 볼 수 있어 재밌습니다.
    plot_surprisal(model, tokenizer, bad_sent, title="Grammar Error Detection")

