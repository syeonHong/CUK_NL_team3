import json
from sklearn.model_selection import train_test_split
import os

INPUT_FILE = 'data/train_eng_implicit.jsonl'
TRAIN_FILE = 'data/eng/implicit/train.jsonl'
VAL_FILE = 'data/eng/implicit/val.jsonl'
TEST_FILE = 'data/eng/implicit/test.jsonl'
RANDOM_STATE = 42


def save_jsonl(data_list, filename):

    output_dir = os.path.dirname(filename)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item) + '\n')


# --- 1. .jsonl 파일 로딩 ---
all_data = []
labels = []

print(f"'{INPUT_FILE}' 파일 로딩 중...")
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data_item = json.loads(line)
            all_data.append(data_item)
            labels.append(data_item['label'])

    print(f"파일 로딩 완료. 총 {len(all_data)}개의 샘플.")

except FileNotFoundError:
    print(f"오류: '{INPUT_FILE}'을 찾을 수 없습니다. 파일 경로를 확인하세요.")
    exit()
except Exception as e:
    print(f"파일 로딩 중 오류 발생: {e}")
    exit()

if not all_data:
    print("오류: 파일은 있으나 데이터가 비어있습니다.")
    exit()

# --- 2. 데이터 분할 (1단계: Test 세트 분리) ---
train_val_data, test_data, train_val_labels, _ = train_test_split(
    all_data,
    labels,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=labels
)

# --- 3. 데이터 분할 (2단계: Train, Validation 분리) ---
train_data, val_data = train_test_split(
    train_val_data,
    test_size=(1 / 9),
    random_state=RANDOM_STATE,
    stratify=train_val_labels
)

print("데이터 분할 완료.")

# --- 4. 분할된 데이터를 .jsonl 파일로 저장 ---
try:
    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(val_data, VAL_FILE)
    save_jsonl(test_data, TEST_FILE)

    print(f"\n--- 최종 파일 저장 완료 ---")
    print(f"Train 세트: {TRAIN_FILE} (총 {len(train_data)}개)")
    print(f"Validation 세트: {VAL_FILE} (총 {len(val_data)}개)")
    print(f"Test 세트: {TEST_FILE} (총 {len(test_data)}개)")

    total_len = len(all_data)
    print(f"\n--- 최종 비율 ---")
    print(f"Train: {len(train_data) / total_len:.1%}")
    print(f"Validation: {len(val_data) / total_len:.1%}")
    print(f"Test: {len(test_data) / total_len:.1%}")

except Exception as e:
    print(f"파일 저장 중 오류 발생: {e}")
