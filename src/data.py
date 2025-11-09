import json, torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class PromptDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, data_path: str, condition: str, max_length: int = 512):
        self.tokenizer = tokenizer
        self.data = []
        self.condition = condition
        self.max_length = max_length

        p = Path(data_path)
        assert p.exists(), f"데이터 파일이 없습니다: {p}"
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    ex = json.loads(line)
                except json.JSONDecoderError:
                    print(f"JSON 피싱 에러: {line[:100]}")
                    continue

                if "text" not in ex or "label" not in ex:
                    continue

                if ex.get("type") != condition:
                    continue
                prompt_text = ex.get("prompt", "")

                if condition == "explicit":
                    input_text = f"{prompt_text}\n\nSentence: {ex['text']}"
                else:
                    if prompt_text and "Example:" in prompt_text:
                        implicit_prompt = prompt_text.split("Example:", 1)[1]
                        input_text = f"Example: {implicit_prompt}\n\nSentence: {ex['text']}"
                    else:
                        input_text = f"Sentence: {ex['text']}"

                self.data.append({
                    "input": input_text,
                    "label": ex['label'],
                    "pair_id": ex.get("pair_id", ""),
                    "id": ex.get("id", ""),
                    "original_text": ex['text']
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["input"]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["input_ids"]
        }

def build_collator(tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
    def collate(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        max_len = min(max_len, max_length)

        padded_batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        for item in batch:
            input_ids = item["input_ids"][:max_len]
            attention_mask = item["attention_mask"][:max_len]
            labels = item["labels"][:max_len]

            pad_len = max_len - len(input_ids)
            padded_batch["input_ids"].append(input_ids + [tokenizer.pad_token_id] * pad_len)
            padded_batch["attention_mask"].append(attention_mask + [0] * pad_len)
            padded_batch["labels"].append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_batch["input_ids"]),
            "attention_mask": torch.tensor(padded_batch["attention_mask"]),
            "labels": torch.tensor(padded_batch["labels"])
        }
    return collate
