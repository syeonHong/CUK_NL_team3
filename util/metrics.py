# -*- coding: utf-8 -*-
"""
compute_metrics: HuggingFace Trainer가 evaluation 시 호출
- eval_loss를 받아 PPL(perplexity)로 변환하여 반환.
- implicit-ins 논문처럼 eval loss -> PPL 비교를 명확히 기록하기 위함.
"""

import math

def compute_ppl_metrics(eval_pred):
    """
    eval_pred: EvalPrediction(loss, logits) 형태
    Trainer는 일반적으로 eval_loss를 logs에 기록하므로,
    compute_metrics에서 직접 전달받지는 않음.
    따라서 eval_pred.predictions가 있을 때 손실 계산 대신,
    logs에 기록된 loss를 이용해 PPL 계산하는 버전.
    """
    # eval_pred에는 (predictions, labels) 또는 (loss, logits) 등 다양하게 올 수 있으므로
    # 안전하게 placeholder로 둠
    metrics = {}

    # Trainer가 eval_loss를 log dict로 넘겨주는 경우
    if isinstance(eval_pred, dict) and "eval_loss" in eval_pred:
        loss = eval_pred["eval_loss"]
        metrics["eval_loss"] = loss
        try:
            metrics["eval_ppl"] = math.exp(loss)
        except OverflowError:
            metrics["eval_ppl"] = float("inf")

    # Trainer 내부에서 compute_metrics 호출 시, 일반적으로 (predictions, labels)
    # 만 전달되므로, 위 케이스 외에는 빈 dict 반환
    return metrics
