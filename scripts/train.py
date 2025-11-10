import argparse, json, yaml, os, sys
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.prompts import build_prompt
from util.metrics import compute_ppl_metrics

class PromptDataset(Dataset):
    def __init__(self, tokenizer, data_path, condition, max_length=512, ok_only=True, min_len=6, max_len_t=25):
        self.samples = []
        self.tok = tokenizer
        self.max_length = max_length
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                if ok_only and ex.get("label") not in (None, "ok"):
                    continue
                L = ex.get("meta", {}).get("length")
                if L is not None and not (min_len <= L <= max_len_t):
                    continue
                text = build_prompt(ex, condition=condition, for_eval=False)
                if text:
                    self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tok(
            self.samples[idx],
            truncation=True,
            max_length=self.max_length,
            # padding은 collator가 처리하므로 여기선 지정하지 않음
        )
        # ✅ 텐서로 만들지 않음. 리스트/파이썬 int 그대로 반환해야 collator가 pad 가능
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get("attention_mask", [1] * len(enc["input_ids"])),
            # ❌ labels는 여기서 넣지 않음 (collator가 자동 생성; mlm=False → LM 학습)
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"])

    train_ds = PromptDataset(
        tokenizer=tokenizer,
        data_path=cfg["data"]["train_path"],
        condition=cfg["data"]["condition"],
        max_length=cfg["train"]["max_length"],
        ok_only=cfg["data"].get("ok_only", True),
        min_len=cfg["data"].get("min_len", 6),
        max_len_t=cfg["data"].get("max_len", 25),
    )
    val_ds = None
    if "val_path" in cfg["data"] and cfg["data"]["val_path"]:
        val_ds = PromptDataset(
            tokenizer=tokenizer,
            data_path=cfg["data"]["val_path"],
            condition=cfg["data"]["condition"],
            max_length=cfg["train"]["max_length"],
            ok_only=cfg["data"].get("ok_only", True),
            min_len=cfg["data"].get("min_len", 6),
            max_len_t=cfg["data"].get("max_len", 25),
        )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    targs = TrainingArguments(
        output_dir=cfg["train"]["output_dir"],
        num_train_epochs=cfg["train"]["epochs"],
        per_device_train_batch_size=cfg["train"]["batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        warmup_steps=cfg["train"]["warmup_steps"],
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch" if val_ds is not None else "steps",
        logging_steps=cfg["train"]["logging_steps"],
        save_total_limit=2,
        fp16=cfg["train"].get("fp16", False),
        bf16=cfg["train"].get("bf16", False),
        weight_decay=cfg["train"].get("weight_decay", 0.01),
        report_to=["tensorboard"],
        load_best_model_at_end=True if val_ds is not None else False,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_ppl_metrics if val_ds is not None else None,
    )

    trainer.train()
    trainer.save_model(cfg["train"]["output_dir"])

if __name__ == "__main__":
    main()
