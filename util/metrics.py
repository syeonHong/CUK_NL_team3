# -*- coding: utf-8 -*-
"""
compute_metrics: HuggingFace Trainer가 evaluation 시 호출
- eval_loss를 받아 PPL(perplexity)로 변환하여 반환.
- implicit-ins 논문처럼 eval loss -> PPL 비교를 명확히 기록하기 위함.
"""

import math

def compute_ppl_metrics(eval_pred):
    metrics = {}

    if isinstance(eval_pred, dict) and "eval_loss" in eval_pred:
        loss = eval_pred["eval_loss"]
        metrics["eval_loss"] = loss
        try:
            metrics["eval_ppl"] = math.exp(loss)
        except OverflowError:
            metrics["eval_ppl"] = float("inf")

    return metrics
