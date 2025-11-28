import math
import torch
import sys


def pll(model, tok, text, device, max_length=512):
    """
    프롬프트 전체에 대한 평균 log p(= -loss) 계산.
    """
    with torch.no_grad():
        ids = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**ids, labels=ids["input_ids"])
        loss = out.loss.item()
        return -loss


def key_from_meta(row):
    """
    pair_id가 없을 때 ok/vi 매칭에 쓸 메타 키 생성.
    """
    meta = row.get("meta", {})
    keys = []
    for k, v in meta.items():
        keys.append(f"{k}:{v}")
    return "|".join(keys)
