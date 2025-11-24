# scripts/train.py
# -*- coding: utf-8 -*-

import argparse
import json
import yaml
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from util.utils import require_cuda
from scripts.prompts import build_prompt

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


class PromptDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        condition,
        max_length=512,
        ok_only=True,
        min_len=6,
        max_len_t=25,
    ):
        self.samples = []
        self.tok = tokenizer
        self.max_length = max_length

        for _ in range(1, 2 + 1):
            print(f"[Dataset Load] {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)

                if ok_only and ex.get("label") not in (None, "ok"):
                    continue

                L = ex.get("meta", {}).get("length")
                if L is not None and not (min_len <= L <= max_len_t):
                    continue

                text = build_prompt(
                    ex,
                    condition=condition,
                    for_eval=False,
                    task_type="generation"
                )
                if text:
                    self.samples.append(text)

        print(f"[Dataset] Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tok(
            self.samples[idx],
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get(
                "attention_mask", [1] * len(enc["input_ids"])
            ),
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    print(">>> [CUDA CHECK]", file=sys.stderr, flush=True)
    require_cuda()

    args = parse_args()
    cfg = load_cfg(args.config)

    expected_ckpt = os.path.join(cfg["train"]["output_dir"], "checkpoint-<global_step>")
    print(f"[CHECKPOINT PATTERN] {expected_ckpt}", file=sys.stderr, flush=True)
    print(f"\n🔍 Expected checkpoint path pattern: {expected_ckpt}\n", file=sys.stderr, flush=True)

    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    print(f"[MODEL] Loaded {model_name}", file=sys.stderr, flush=True)

    condition = cfg["data"]["condition"]

    train_ds = PromptDataset(
        tokenizer=tokenizer,
        data_path=cfg["data"]["train_path"],
        condition=condition,
        max_length=cfg["train"]["max_length"],
        ok_only=cfg["data"].get("ok_only", True),
        min_len=cfg["data"].get("min_len", 6),
        max_len_t=cfg["data"].get("max_len", 25),
    )

    val_ds = None
    print("[DATA] Validation is DISABLED for training (E1).", file=sys.stderr, flush=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    targs = TrainingArguments(
        output_dir=cfg["train"]["output_dir"],
        num_train_epochs=cfg["train"]["epochs"],
        per_device_train_batch_size=cfg["train"]["batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        warmup_steps=cfg["train"]["warmup_steps"],
        logging_steps=cfg["train"]["logging_steps"],
        weight_decay=cfg["train"].get("weight_decay", 0.01),
        fp16=cfg["train"].get("fp16", False),
        bf16=cfg["train"].get("bf16", False),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=None,
    )

    trainer.train()
    trainer.save_model(cfg["train"]["output_dir"])
    print(f"[SAVE] Final models saved to {cfg['train']['output_dir']}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
