import json
import random

# --- 설정 (Constants) ---
INPUT_ENGLISH_PAIRS = "english_annotated_pairs.jsonl"
INPUT_ARTLANG_PAIRS = "artlang_annotated_pairs.jsonl"

# 출력 파일 이름을 4개로 분리
OUTPUT_ENG_EXP = "train_eng_explicit.jsonl"
OUTPUT_ARLA_EXP = "train_arla_explicit.jsonl"
OUTPUT_ENG_IMP = "train_eng_implicit.jsonl"
OUTPUT_ARLA_IMP = "train_arla_implicit.jsonl"
OUTPUT_TEST = "test.jsonl"

# 요구사항: 각 파일당 2000개
NUM_ENG_TRAIN = 2000  # 1000 'ok' + 1000 'violation'
NUM_ARLA_TRAIN = 2000  # 1000 'ok' + 1000 'violation'
NUM_ARLA_TEST = 500  # 250 'ok' + 250 'violation'

# --- 프롬프트 (Hints) 정의 ---
ENGLISH_PROMPT = "Rule: Subject-Verb-Object order (adverb optional, sentence-final). Example: The dog eats the bone."
ARLA_PROMPT = "Rule: Subject-Object-Verb order (adverb optional, sentence-final). Example: pleck li vode lu praz noyka"


def load_annotated_pairs(filepath):
    """
    역할: 스크립트 1, 2에서 생성된 .jsonl 파일을 읽어 'ok'와 'violation' 리스트로 분리합니다.
    """
    pairs_ok = []
    pairs_violation = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry["label"] == "ok":
                    pairs_ok.append(entry)
                else:
                    pairs_violation.append(entry)
    except FileNotFoundError:
        print(f"Error: Input file not found: {filepath}")
        print(f"Please run 'simple_wiki_parser.py' and 'artlang_generator.py' first.")
        exit()

    random.shuffle(pairs_ok)
    random.shuffle(pairs_violation)
    return pairs_ok, pairs_violation


def build_explicit_files(eng_ok, eng_vio, arla_ok, arla_vio):
    """
 명시적 데이터를 영어와 인공어로 분리하여 2개의 파일로 생성합니다.
    """
    eng_dataset = []
    arla_dataset = []

    # 1. 영어 샘플 추가 (2000개)
    for i in range(NUM_ENG_TRAIN // 2):
        ok_entry = eng_ok[i]
        ok_entry["type"] = "explicit"
        ok_entry["prompt"] = ENGLISH_PROMPT
        # 영어 샘플에 일관된 SVO 규칙 할당
        ok_entry["meta"] = {
            "rule": "SVO_word_order", "language": "english",
            "length": len(ok_entry["tokens"]), "source": ok_entry["meta"]["source"],
            "parser_confidence": 0.95
        }
        eng_dataset.append(ok_entry)

        vio_entry = eng_vio[i]
        vio_entry["type"] = "explicit"
        vio_entry["prompt"] = ENGLISH_PROMPT
        vio_entry["meta"] = {
            "rule": "SVO_word_order", "language": "english",
            "length": len(vio_entry["tokens"]), "source": vio_entry["meta"]["source"],
            "perturbation": "swap(O,V)"
        }
        eng_dataset.append(vio_entry)

    # 2. 인공어 샘플 추가 (2000개)
    for i in range(NUM_ARLA_TRAIN // 2):
        ok_entry = arla_ok[i]
        ok_entry["type"] = "explicit"
        ok_entry["prompt"] = ARLA_PROMPT
        ok_entry["meta"]["rule"] = "SOV_word_order"
        arla_dataset.append({
            "id": ok_entry["id"], "type": "explicit", "prompt": ARLA_PROMPT,
            "text": ok_entry["text"], "label": "ok", "meta": ok_entry["meta"]
        })

        vio_entry = arla_vio[i]
        vio_entry["type"] = "explicit"
        vio_entry["prompt"] = ARLA_PROMPT
        vio_entry["meta"]["rule"] = "SOV_word_order"
        arla_dataset.append({
            "id": vio_entry["id"], "type": "explicit", "prompt": ARLA_PROMPT,
            "text": vio_entry["text"], "label": "violation", "meta": vio_entry["meta"]
        })

    # 3. 분리하여 셔플 및 저장
    random.shuffle(eng_dataset)
    random.shuffle(arla_dataset)

    with open(OUTPUT_ENG_EXP, "w", encoding="utf-8") as f:
        for entry in eng_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Built {OUTPUT_ENG_EXP} with {len(eng_dataset)} samples.")

    with open(OUTPUT_ARLA_EXP, "w", encoding="utf-8") as f:
        for entry in arla_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Built {OUTPUT_ARLA_EXP} with {len(arla_dataset)} samples.")


def build_implicit_files(eng_ok, eng_vio, arla_ok, arla_vio):
    """
암시적 데이터를 영어와 인공어로 분리하여 2개의 파일로 생성합니다.
    """
    eng_dataset = []
    arla_dataset = []

    # 1. 영어 샘플 추가 (2000개) - explicit에서 사용한 데이터 제외
    start_idx = NUM_ENG_TRAIN // 2
    end_idx = start_idx + (NUM_ENG_TRAIN // 2)
    for i in range(start_idx, end_idx):
        ok_entry = eng_ok[i]
        eng_dataset.append({
            "id": ok_entry["id"].replace("_ok", "_imp"), "type": "implicit", "prompt": "",
            "text": ok_entry["text"], "label": "ok",
            "meta": {"language": "english", "length": len(ok_entry["tokens"])}
        })

        vio_entry = eng_vio[i]
        eng_dataset.append({
            "id": vio_entry["id"].replace("_vi", "_imp"), "type": "implicit", "prompt": "",
            "text": vio_entry["text"], "label": "violation",
            "meta": {"language": "english", "length": len(vio_entry["tokens"])}
        })

    # 2. 인공어 샘플 추가 (2000개)
    start_idx = NUM_ARLA_TRAIN // 2
    end_idx = start_idx + (NUM_ARLA_TRAIN // 2)
    for i in range(start_idx, end_idx):
        ok_entry = arla_ok[i]
        arla_dataset.append({
            "id": ok_entry["id"].replace("_ok", "_imp"), "type": "implicit", "prompt": "",
            "text": ok_entry["text"], "label": "ok", "meta": ok_entry["meta"]
        })

        vio_entry = arla_vio[i]
        arla_dataset.append({
            "id": vio_entry["id"].replace("_vi", "_imp"), "type": "implicit", "prompt": "",
            "text": vio_entry["text"], "label": "violation", "meta": vio_entry["meta"]
        })

    # 3. 분리하여 셔플 및 저장
    random.shuffle(eng_dataset)
    random.shuffle(arla_dataset)

    with open(OUTPUT_ENG_IMP, "w", encoding="utf-8") as f:
        for entry in eng_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Built {OUTPUT_ENG_IMP} with {len(eng_dataset)} samples.")

    with open(OUTPUT_ARLA_IMP, "w", encoding="utf-8") as f:
        for entry in arla_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Built {OUTPUT_ARLA_IMP} with {len(arla_dataset)} samples.")


def build_test_dataset(arla_ok, arla_vio):
    """
    역할: test.jsonl (총 500개, 인공어) 파일을 생성합니다. (변경 없음)
    """
    dataset = []

    # 학습에 사용되지 않은 인덱스에서 데이터 추출
    start_idx = (NUM_ARLA_TRAIN // 2) * 2  # explicit(1000쌍) + implicit(1000쌍) 이후
    end_idx = start_idx + (NUM_ARLA_TEST // 2)

    # 인덱스 범위 확인
    if len(arla_ok) < end_idx or len(arla_vio) < end_idx:
        print(f"Warning: Not enough ArLa samples for TEST set. Need {end_idx} pairs.")
        # 사용 가능한 최대치만 사용
        available_ok = arla_ok[start_idx:]
        available_vio = arla_vio[start_idx:]

        for ok_entry in available_ok:
            dataset.append({
                "id": ok_entry["id"].replace("_ok", "_test"), "type": "implicit", "prompt": "",
                "text": ok_entry["text"], "label": "ok", "meta": ok_entry["meta"]
            })
        for vio_entry in available_vio:
            dataset.append({
                "id": vio_entry["id"].replace("_vi", "_test"), "type": "implicit", "prompt": "",
                "text": vio_entry["text"], "label": "violation", "meta": vio_entry["meta"]
            })
    else:
        # 정상 범위
        for i in range(start_idx, end_idx):
            ok_entry = arla_ok[i]
            dataset.append({
                "id": ok_entry["id"].replace("_ok", "_test"), "type": "implicit", "prompt": "",
                "text": ok_entry["text"], "label": "ok", "meta": ok_entry["meta"]
            })

            vio_entry = arla_vio[i]
            dataset.append({
                "id": vio_entry["id"].replace("_vi", "_test"), "type": "implicit", "prompt": "",
                "text": vio_entry["text"], "label": "violation", "meta": vio_entry["meta"]
            })

    random.shuffle(dataset)

    with open(OUTPUT_TEST, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Built {OUTPUT_TEST} with {len(dataset)} samples.")


def main():
    print("Loading annotated English pairs...")
    eng_ok, eng_vio = load_annotated_pairs(INPUT_ENGLISH_PAIRS)
    print(f"Loaded {len(eng_ok)} 'ok' and {len(eng_vio)} 'violation' English samples.")

    print("Loading annotated Artificial Language pairs...")
    arla_ok, arla_vio = load_annotated_pairs(INPUT_ARTLANG_PAIRS)
    print(f"Loaded {len(arla_ok)} 'ok' and {len(arla_vio)} 'violation' ArLa samples.")

    # 샘플 수가 충분한지 확인
    # 필요한 영어 샘플 수 = 2000 (explicit) + 2000 (implicit) = 4000 (2000쌍)
    total_eng_needed = NUM_ENG_TRAIN  # 2000개 (1000쌍 ok/vio)

    # 필요한 인공어 샘플 수 = 2000 (explicit) + 2000 (implicit) + 500 (test) = 4500 (2250쌍)
    total_arla_needed = (NUM_ARLA_TRAIN // 2) * 2 + (NUM_ARLA_TEST // 2)  # 1000 + 1000 + 250 = 2250 쌍

    if len(eng_ok) < (NUM_ENG_TRAIN // 2) * 2:  # 1000(exp ok) + 1000(imp ok)
        print(f"Warning: Not enough English 'ok' samples. Need {(NUM_ENG_TRAIN // 2) * 2}, found {len(eng_ok)}.")
        return
    if len(arla_ok) < total_arla_needed:
        print(f"Warning: Not enough ArLa 'ok' samples. Need {total_arla_needed}, found {len(arla_ok)}.")
        return

    # 함수 호출 및 프린트문 변경
    print("\nBuilding EXPLICIT files...")
    build_explicit_files(eng_ok, eng_vio, arla_ok, arla_vio)

    print("\nBuilding IMPLICIT files...")
    build_implicit_files(eng_ok, eng_vio, arla_ok, arla_vio)

    print("\nBuilding TEST file...")
    build_test_dataset(arla_ok, arla_vio)

    print("\nAll datasets built successfully!")


if __name__ == "__main__":
    main()