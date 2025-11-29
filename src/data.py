import json, torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from src.prompts import build_prompt

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

                if ex.get("type") != condition:
                    continue

                if ex.get("label") != "ok":
                    continue
                full_text = build_prompt(ex, condition)

                self.data.append({
                    "full_text": full_text,
                    "original_text": ex['text']
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        full_text = item["full_text"]

        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]

        labels = input_ids.clone()

        if "Sentence: " in full_text:
            prompt_part = full_text.split("Sentence: ")[0] + "Sentence: "

            prompt_ids = self.tokenizer(prompt_part, truncation=True, max_length=self.max_length)["input_ids"]
            prompt_len = len(prompt_ids)

            if prompt_len < len(labels):
                labels[:prompt_len] = -100
            else:
                labels[:] = -100
        return{
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def build_collator(tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
    def collate(batch):
        # 배치 내 가장 긴 시퀀스 길이에 맞춰 패딩
        max_batch_len = max(len(item["input_ids"]) for item in batch)
        max_batch_len = min(max_batch_len, max_length)

        padded_batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        for item in batch:
            # 텐서를 리스트로 변환
            curr_input_ids = item["input_ids"].tolist()
            curr_mask = item["attention_mask"].tolist()
            curr_labels = item["labels"].tolist()

            # Truncate (혹시 모를 안전장치)
            curr_input_ids = curr_input_ids[:max_batch_len]
            curr_mask = curr_mask[:max_batch_len]
            curr_labels = curr_labels[:max_batch_len]

            # Padding
            pad_len = max_batch_len - len(curr_input_ids)

            # input_ids는 pad_token으로, labels는 -100으로 패딩
            padded_batch["input_ids"].append(curr_input_ids + [tokenizer.pad_token_id] * pad_len)
            padded_batch["attention_mask"].append(curr_mask + [0] * pad_len)
            padded_batch["labels"].append(curr_labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_batch["input_ids"]),
            "attention_mask": torch.tensor(padded_batch["attention_mask"]),
            "labels": torch.tensor(padded_batch["labels"])
        }
    return collate
