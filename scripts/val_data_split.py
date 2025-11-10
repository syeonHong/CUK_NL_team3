import os
import json
import random

# ===== 설정 =====
BASE_DIR = r"C:\Users\User\PycharmProjects\CUK_NL_team3\data"
OUTPUT_DIR = os.path.join(BASE_DIR, "split_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILES = {
    "explicit": os.path.join(BASE_DIR, "train_arla_explicit.jsonl"),
    "implicit": os.path.join(BASE_DIR, "train_arla_implicit.jsonl"),
}

SPLIT_RATIO = (0.8, 0.1, 0.1)  # (train, val, test)
SEED = 42
random.seed(SEED)


def split_jsonl(input_path, train_out, val_out, test_out, ratio=(0.8, 0.1, 0.1)):
    """JSONL 파일을 주어진 비율로 train/val/test 분할"""
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    random.shuffle(lines)
    n_total = len(lines)
    n_train = int(n_total * ratio[0])
    n_val = int(n_total * ratio[1])
    n_test = n_total - n_train - n_val

    train_data = lines[:n_train]
    val_data = lines[n_train:n_train + n_val]
    test_data = lines[n_train + n_val:]

    # 저장
    with open(train_out, "w", encoding="utf-8") as ft:
        for ex in train_data:
            ft.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_out, "w", encoding="utf-8") as fv:
        for ex in val_data:
            fv.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(test_out, "w", encoding="utf-8") as fs:
        for ex in test_data:
            fs.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[✓] {os.path.basename(input_path)} 분할 완료:")
    print(f"    Train → {len(train_data)}개 → {train_out}")
    print(f"    Val   → {len(val_data)}개 → {val_out}")
    print(f"    Test  → {len(test_data)}개 → {test_out}\n")


def main():
    for cond, path in INPUT_FILES.items():
        train_out = os.path.join(OUTPUT_DIR, f"train_{cond}.jsonl")
        val_out = os.path.join(OUTPUT_DIR, f"val_{cond}.jsonl")
        test_out = os.path.join(OUTPUT_DIR, f"test_{cond}.jsonl")
        split_jsonl(path, train_out, val_out, test_out, ratio=SPLIT_RATIO)


if __name__ == "__main__":
    main()
