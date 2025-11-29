# tensorboard --logdir logs/explicit_gpt2/logs
import argparse, yaml, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback
from src.data import PromptDataset, build_collator
from util.metrics import compute_grammaticality_accuracy, compute_perplexity_from_loss, PerplexityCallback


# PPL 계산 콜백 (로그 확인용)
class PerplexityCallback(EarlyStoppingCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" in metrics:
            loss = metrics["eval_loss"]
            try:
                ppl = torch.exp(torch.tensor(loss)).item()
                print(f"\n[Epoch {state.epoch:.2f}] Eval PPL: {ppl:.4f}")
            except OverflowError:
                print(f"\n[Epoch {state.epoch:.2f}] Eval PPL: Infinity")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="YAML 경로 (base/explicit/implicit)")
    return p.parse_args()

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    # 1) Tokenizer/Model 로드
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    # GPT-2 계열은 pad_token이 없어서 에러가 잦음 -> pad_token을 eos로 지정(이유: casual LM에서는 안전)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"])

    condition = cfg["data"]["condition"]

    # 2) Dataset 로드 (explicit/implicit은 데이터 파일은 같아도 "prompt_builder"로 구분)
    train_ds = PromptDataset(
        tokenizer=tokenizer,
        data_path=cfg["data"]["train_path"],
        condition=condition,     # "explicit" | "implicit"
        max_length=cfg["train"]["max_length"]
    )

    if "val_path" in cfg["data"] and cfg["data"]["val_path"]:
        val_ds = PromptDataset(
            tokenizer=tokenizer,
            data_path=cfg["data"]["val_path"],
            condition=condition,
            max_length=cfg["train"]["max_length"]
        )
    else:
        val_ds = None  # dev가 없을 때는 evaluation_strategy="no"로 학습하거나 내부 split을 구현

    data_collator = build_collator(tokenizer, max_length=cfg["train"]["max_length"])

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3) TrainingArguments 설정 (암묵적/명시적 조건 동일 유지 -> 공정성)
    targs = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["train"]["epochs"],
        per_device_train_batch_size=cfg["train"]["batch_size"],
        per_device_eval_batch_size=cfg["train"]["batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        warmup_steps=cfg["train"]["warmup_steps"],

        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True if val_ds is not None else False,
        metric_for_best_model="loss",

        logging_dir=str(output_dir / "logs"),
        logging_steps=cfg["train"]["logging_steps"],
        report_to=["tensorboard"],

        fp16=cfg["train"].get("fp16", False),
        bf16=cfg["train"].get("bf16", False),
        weight_decay=cfg["train"].get("weight_decay", 0.01),
        max_grad_norm=1.0,

        seed=42,
        dataloader_num_workers=0,
        dataloader_pin_memory=False
    )

    # 4) Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3),
                   PerplexityCallback()] if val_ds else []
    )

    trainer.train()
    final_path = output_dir / "final_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

if __name__ == "__main__":
    main()
