import math
import numpy as np
from transformers import TrainerCallback
def compute_ppl_metrics(eval_pred):
    # HF Trainer는 eval_loss를 넘겨주지 않으므로, 통상 metrics_callback에서 가져오거나
    # 여기서는 placeholder로 남겨둠. (Trainer가 loss를 dict로 줄 때 활용)
    # 단순화: eval_loss를 Trainer가 자동 로깅하면 TB로 확인. 필요한 경우 커스텀 TrainerCallback 사용.
    # 스켈레톤: 빈 dict 반환
    return {}

def compute_grammaticality_accuracy(pair_results):
    correct = 0
    total = 0

    for pair_id, results in pair_results.items():
        if "ok" in results and "violation" in results:
            continue
        total += 1
        if results["ok"] < results["violation"]:
            correct += 1
    return correct / total if total > 0 else 0.0

def compute_perplexity_from_loss(loss):
    return math.exp(loss)

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" in metrics:
            ppl = math.exp(metrics["eval_loss"])
            metrics["eval_perplexity"] = ppl
            print(f"\n[Eval] Perplexity: {ppl:.4f}\n")