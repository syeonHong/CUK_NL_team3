import torch
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, roc_curve

# ==============================================================================
# [1] 프롬프트 정의 (사용자 제공 코드)
# ==============================================================================
EXPLICIT_RULE_CARD = """[GRAMMAR RULES]
1) Word Order: Subject-Verb-Object (SVO)
   - The subject comes first
   - The verb comes second  
   - The object comes third
   - Adverbs can appear optionally at the end

2) Examples:
   ✓ Correct: "The dog eats the bone."
   ✓ Correct: "They will hunt birds sometimes."
   ✗ Incorrect: "Eats the dog the bone." (VSO order)
   ✗ Incorrect: "The bone the dog eats." (OSV order)
"""
IMPLICIT_EXAMPLES = """[EXAMPLES]
✓ The dog eats the bone.
✓ They will hunt birds sometimes.
✓ Each zebra has a unique pattern.
✗ Eats the dog the bone.
✗ The bone the dog eats.
"""

CONDITION = "explicit" #"explicit" || "implicit"
DATA_PATH = "data/test_eng.jsonl"
MODEL_PATH = f"logs/{CONDITION}_gpt2/final_model"


def build_prompt(
        ex: dict,
        condition: str = "implicit",
        for_eval: bool = False,
        task_type: str = "generation",
) -> str:
    if for_eval:
        return ex.get("text", "")

    sent = ex.get("text", "")
    condition = (condition or "").lower()

    # 학습 코드와 동일한 프롬프트 구성 로직
    if condition == "explicit":
        rule_section = f"{EXPLICIT_RULE_CARD}\n\n"
    else:
        # Implicit 모델 학습 방식에 따라 수정 (여기서는 예시 포함)
        rule_section = f"{IMPLICIT_EXAMPLES}\n\n"
        # 만약 Implicit은 예시 없이 문장만 학습했다면 아래 줄 사용:
        # rule_section = ""

    if task_type == "generation":
        # PPL 계산용 (Sentence: 뒤에 문장이 옴)
        prompt = f"{rule_section}Sentence: {sent}"

    elif task_type == "grammaticality":
        # Calibration 측정용 (Yes/No 질문)
        # 주의: 이 포맷은 학습되지 않았을 수 있음 (Zero-shot)
        prompt = (
            f"{rule_section}"
            f"Judge whether the following sentence is grammatically correct.\n"
            f"Sentence: {sent}\n"
            f"Answer (Yes/No):"
        )
    else:
        prompt = f"{rule_section}Sentence: {sent}"

    return prompt


# ==============================================================================
# [3] 지표 계산 클래스
# ==============================================================================
class MetricCalculator:
    def __init__(self, model_path, condition="explicit"):
        print(f"Loading model from {model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        except OSError:
            raise OSError(f"경로를 찾을 수 없습니다: {model_path}\nMODEL_PATH 변수를 확인해주세요.")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.condition = condition

    def get_sentence_score(self, text):
        """
        문장의 Log-Likelihood (LL) 점수를 계산합니다.
        점수가 0에 가까울수록(음수 값이 클수록) 모델이 자연스럽게 느끼는 문장입니다.
        """
        # Generation 태스크 프롬프트 생성
        prompt = build_prompt({"text": text}, condition=self.condition, task_type="generation")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # labels를 주면 모델이 자동으로 Loss를 계산
            outputs = self.model(**inputs, labels=inputs["input_ids"])

        loss = outputs.loss.item()
        # LL = -Loss * Length
        return -loss * inputs["input_ids"].shape[1]

    def get_calibration_prob(self, text):
        """
        Yes/No 문제에 대한 모델의 확신도(Confidence)를 계산합니다.
        """
        # Grammaticality 태스크 프롬프트 생성
        prompt = build_prompt({"text": text}, condition=self.condition, task_type="grammaticality")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # 마지막 토큰(Yes/No를 예측해야 하는 위치)의 Logits
            logits = outputs.logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        # 'Yes'와 'No' 토큰 ID 찾기
        yes_candidates = ["Yes", " Yes", "yes", " yes"]
        no_candidates = ["No", " No", "no", " no"]

        # 토크나이저에 있는 첫 번째 후보 토큰 ID 사용
        yes_id = self.tokenizer.convert_tokens_to_ids(yes_candidates[0])
        no_id = self.tokenizer.convert_tokens_to_ids(no_candidates[0])

        prob_yes = probs[0, yes_id].item()
        prob_no = probs[0, no_id].item()

        # 정규화
        total = prob_yes + prob_no + 1e-12
        prob_yes /= total
        prob_no /= total

        # 예측 및 확신도
        pred_label = "ok" if prob_yes >= 0.5 else "violation"
        confidence = prob_yes if pred_label == "ok" else prob_no

        return pred_label, confidence


# ==============================================================================
# [4] 메인 실행 함수
# ==============================================================================
def main():
    print(f"=== Configuration ===")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Data Path : {DATA_PATH}")
    print(f"Condition : {CONDITION}")
    print(f"=====================\n")

    calculator = MetricCalculator(MODEL_PATH, condition=CONDITION)

    ok_scores = []
    viol_scores = []
    data_ok = []
    data_viol = []
    calib_data = []

    print(f"Running evaluation...")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        if not line.strip(): continue
        try:
            ex = json.loads(line)
        except:
            continue

        text = ex.get("text")
        label = ex.get("label")  # "ok" or "violation"

        if not text or not label: continue

        score = calculator.get_sentence_score(text)

        if label == "ok":
            ok_scores.append(score)
        elif label == "violation":
            viol_scores.append(score)
        pred_label, conf = calculator.get_calibration_prob(text)
        is_correct = (pred_label == label)
        calib_data.append((is_correct, conf))


    # ==========================================================================
    # [Metric 1] AUC (정확도 대체 지표)
    # ==========================================================================
    y_ture = [1] * len(ok_scores) + [0] * len(viol_scores)
    y_scores = ok_scores + viol_scores

    auc = roc_auc_score(y_ture, y_scores)
    print("\n" + "=" * 40)
    print(f" [Metric 1] AUC (Separability)")
    print("=" * 40)
    print(f" OK Samples      : {len(ok_scores)}")
    print(f" Viol Samples    : {len(viol_scores)}")
    print(f" ⭐ AUC Score     : {auc:.4f}")


    # ==========================================================================
    # [Metric 2] PLL Gap (자연스러움 점수 차이)
    # ==========================================================================
    avg_ok = np.mean(ok_scores) if ok_scores else 0
    avg_viol = np.mean(viol_scores) if viol_scores else 0
    pll_gap = avg_ok - avg_viol

    print("\n" + "=" * 40)
    print(f" [Metric 2] PLL Gap Analysis ({CONDITION})")
    print("=" * 40)
    print(f" Average LL (OK)       : {avg_ok:.4f} (Higher is better)")
    print(f" Average LL (Violation): {avg_viol:.4f}")
    print(f" ----------------------------------------")
    print(f" ⭐ PLL Gap           : {pll_gap:.4f}")

    if pll_gap > 0:
        print(" ✅ Result: Success (모델이 정문을 더 자연스럽게 느낌)")
    else:
        print(" ❌ Result: Fail (모델이 비문을 더 선호함)")

    # ==========================================================================
    # [Metric 3] Calibration (ECE)
    # ==========================================================================
    confidences = np.array([x[1] for x in calib_data])
    corrects = np.array([x[0] for x in calib_data])
    accuracy = np.mean(corrects)

    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accuracies = []
    bin_confs = []

    for i in range(n_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(corrects[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            bin_accuracies.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_weight = np.sum(bin_mask) / len(confidences)
            ece += bin_weight * np.abs(bin_acc - bin_conf)

    print("\n" + "=" * 40)
    print(f" [Metric 3] Calibration Analysis")
    print("=" * 40)
    print(f" Accuracy (Yes/No Task): {accuracy * 100:.2f}%")
    print(f" ⭐ ECE                 : {ece:.4f} (Lower is better)")

    # 그래프 그리기
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.plot(bin_confs, bin_accuracies, marker="o", color="blue", label=f"Model (ECE={ece:.2f})")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram ({CONDITION})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = f"calibration_plot_{CONDITION}.png"
    plt.savefig(save_path)
    print(f" 📊 그래프 저장됨: {save_path}")


if __name__ == "__main__":
    main()