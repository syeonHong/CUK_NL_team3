import json
import random
import math
import os
from collections import Counter

# --- 1. 파일 경로 설정 (수정) ---
INPUT_ENGLISH_PAIRS = "english_annotated_pairs.jsonl"
INPUT_ARTLANG_PAIRS = "artlang_annotated_pairs.jsonl"

# 베이스 출력 디렉터리 정의
OUTPUT_DIR = "output/dataset"

# Train/Dev 파일 (8개) - 유지
OUTPUT_ENG_EXP_TRAIN = os.path.join(OUTPUT_DIR, "train_eng_explicit.jsonl")
OUTPUT_ENG_EXP_DEV = os.path.join(OUTPUT_DIR, "dev_eng_explicit.jsonl")
OUTPUT_ARLA_EXP_TRAIN = os.path.join(OUTPUT_DIR, "train_arla_explicit.jsonl")
OUTPUT_ARLA_EXP_DEV = os.path.join(OUTPUT_DIR, "dev_arla_explicit.jsonl")

OUTPUT_ENG_IMP_TRAIN = os.path.join(OUTPUT_DIR, "train_eng_implicit.jsonl")
OUTPUT_ENG_IMP_DEV = os.path.join(OUTPUT_DIR, "dev_eng_implicit.jsonl")
OUTPUT_ARLA_IMP_TRAIN = os.path.join(OUTPUT_DIR, "train_arla_implicit.jsonl")
OUTPUT_ARLA_IMP_DEV = os.path.join(OUTPUT_DIR, "dev_arla_implicit.jsonl")

# Test 파일 (2개로 단순화)
OUTPUT_TEST_ENG = os.path.join(OUTPUT_DIR, "test_eng.jsonl")  # --- (수정) 파일명 변경 ---
OUTPUT_TEST_ARLA = os.path.join(OUTPUT_DIR, "test_arla.jsonl")  # --- (수정) 파일명 변경 ---

# --- 2. 스키마 정의  ---
SCHEMA_DEFS = {
    "ENG_EXPLICIT_FIELDS": [
        "id", "pair_id", "type", "prompt", "text", "label", "tokens", "spans", "tags", "order", "meta"
    ],
    "ENG_IMPLICIT_FIELDS": [
        "id", "type", "prompt", "text", "label", "meta"
    ],
    "ARLA_EXPLICIT_FIELDS": [
        "id", "type", "prompt", "text", "label", "meta"
    ],
    "ARLA_IMPLICIT_FIELDS": [
        "id", "type", "prompt", "text", "label", "meta"
    ],
}

# --- 3. 목표 크기 및 비율 상수  ---
TARGET_TOTAL_PAIRS = 2000
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1
assert math.isclose(TRAIN_RATIO + DEV_RATIO + TEST_RATIO, 1.0), "Ratios must sum to 1.0"

# --- 4. 형태 변이 확률  ---
REGULAR_PROB = 0.50
IRREG1_PROB = 0.30
IRREG2_PROB = 0.20
MORPH_DEVIATION_THRESHOLD = 0.05

# --- 5. 기타 설정  ---
MIN_TOKENS = 6
MAX_TOKENS = 25

ENGLISH_PROMPT = "Rule: Subject-Verb-Object order (adverb optional, sentence-final). Example: The dog eats the bone."
ARLA_PROMPT = "Rule: Subject-Object-Verb order (adverb optional, sentence-final). Example: pleck li vode lu praz noyka"


# --- load_annotated_pairs 함수  ---
def load_annotated_pairs(filepath, is_english=False):
    """
    .jsonl 파일을 읽어 'ok'와 'violation' 리스트로 분리합니다.
    is_english=True일 때, 6-25 토큰 길이 필터를 적용합니다.
    is_english=False일 때, 형태(plural_type) 비율을 집계합니다.
    """
    pairs_ok = []
    pairs_violation = []
    skipped_count = 0
    morph_counter = Counter()

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)

                # 영어 길이 필터
                if is_english:
                    token_count = len(entry.get("tokens", []))
                    if not (MIN_TOKENS <= token_count <= MAX_TOKENS):
                        skipped_count += 1
                        continue

                # 인공어 형태 집계 (정문 'ok' 기준)
                if not is_english and entry["label"] == "ok":
                    meta = entry.get("meta", {})
                    s_plural = meta.get("s", {}).get("plural_type")
                    o_plural = meta.get("o", {}).get("plural_type")
                    if s_plural: morph_counter[s_plural] += 1
                    if o_plural: morph_counter[o_plural] += 1

                if entry["label"] == "ok":
                    pairs_ok.append(entry)
                else:
                    pairs_violation.append(entry)

    except FileNotFoundError:
        print(f"Error: Input file not found: {filepath}")
        exit(1)

    if is_english and skipped_count > 0:
        print(f"  (Filtered out {skipped_count} English samples due to length constraints [6-25 tokens])")

    return pairs_ok, pairs_violation, morph_counter


# --- verify_morphology 함수  ---
def verify_morphology(morph_counter):
    """
    로드된 인공어 데이터의 형태 비율이 목표치와 맞는지 검증합니다.
    """
    print("\nVerifying ArLa morphology ratios (based on 'ok' samples)...")
    total = sum(morph_counter.values())
    if total == 0:
        print("  Warning: No morphology data found to verify.")
        return

    real_reg_ratio = morph_counter.get("regular", 0) / total
    real_irreg1_ratio = morph_counter.get("irreg1", 0) / total
    real_irreg2_ratio = morph_counter.get("irreg2", 0) / total

    print(f"  Target: REG={REGULAR_PROB:.2%} | IRREG1={IRREG1_PROB:.2%} | IRREG2={IRREG2_PROB:.2%}")
    print(
        f"  Actual: REG={real_reg_ratio:.2%} | IRREG1={real_irreg1_ratio:.2%} | IRREG2={real_irreg2_ratio:.2%} (N={total})")

    if abs(real_reg_ratio - REGULAR_PROB) > MORPH_DEVIATION_THRESHOLD:
        print(f"  WARNING: Regular ratio deviation > {MORPH_DEVIATION_THRESHOLD:.0%}")
    if abs(real_irreg1_ratio - IRREG1_PROB) > MORPH_DEVIATION_THRESHOLD:
        print(f"  WARNING: Irreg1 ratio deviation > {MORPH_DEVIATION_THRESHOLD:.0%}")


# --- write_dataset_files 함수  ---
def write_dataset_files(
        ok_pairs: list,
        vio_pairs: list,
        lang: str,
        num_train_pairs: int,
        num_dev_pairs: int
):
    """
    역할: 전달된 (ok, vio) 페어 리스트 (Train+Dev 풀)를 받아서
          Train/Dev, Explicit/Implicit 파일 4개를 생성합니다.
    """

    # 0. 데이터가 충분한지 확인 및 슬라이싱
    total_needed = num_train_pairs + num_dev_pairs
    if len(ok_pairs) < total_needed or len(vio_pairs) < total_needed:
        print(
            f"Error: Not enough pairs for {lang} Train/Dev split. Needed {total_needed}, Found {len(ok_pairs)} ok / {len(vio_pairs)} vio")
        exit(1)

    # 1. Train / Dev 스플릿 (전달된 풀에서 슬라이싱)
    train_ok = ok_pairs[:num_train_pairs]
    train_vio = vio_pairs[:num_train_pairs]

    dev_ok = ok_pairs[num_train_pairs: total_needed]
    dev_vio = vio_pairs[num_train_pairs: total_needed]

    # 2. 파일 경로 및 프롬프트 설정
    if lang == "eng":
        prompt = ENGLISH_PROMPT
        rule = "SVO_word_order"
        out_exp_train = OUTPUT_ENG_EXP_TRAIN
        out_exp_dev = OUTPUT_ENG_EXP_DEV
        out_imp_train = OUTPUT_ENG_IMP_TRAIN
        out_imp_dev = OUTPUT_ENG_IMP_DEV
    else:  # arla
        prompt = ARLA_PROMPT
        rule = "SOV_word_order"
        out_exp_train = OUTPUT_ARLA_EXP_TRAIN
        out_exp_dev = OUTPUT_ARLA_EXP_DEV
        out_imp_train = OUTPUT_ARLA_IMP_TRAIN
        out_imp_dev = OUTPUT_ARLA_IMP_DEV

    # 3. 파일 생성 함수 (내부 헬퍼)
    def build_file(out_path, ok_list, vio_list, is_explicit):
        dataset = []
        for entry in (ok_list + vio_list):
            new_entry = {}

            # 1. 공통 필드
            new_entry["id"] = entry["id"]
            if not is_explicit:
                new_entry["id"] = new_entry["id"].replace("_ok", "_imp").replace("_vi", "_imp")

            new_entry["type"] = "explicit" if is_explicit else "implicit"
            new_entry["prompt"] = prompt if is_explicit else ""
            new_entry["text"] = entry["text"]
            new_entry["label"] = entry["label"]

            # 2. 메타데이터 (lang별 분기)
            if lang == "eng":
                new_entry["meta"] = {
                    "rule": rule,
                    "language": "english",
                    "length": len(entry.get("tokens", [])),
                    "source": entry.get("meta", {}).get("source", "simplewiki")
                }
                if entry["label"] == "violation":
                    new_entry["meta"]["perturbation"] = "swap(O,V)"
            else:  # arla
                new_entry["meta"] = entry.get("meta", {}).copy()  # 원본 상속 (복사)
                new_entry["meta"]["rule"] = rule

            # 3. Explicit 전용 필드 (Eng)
            if is_explicit and lang == "eng":
                new_entry["pair_id"] = entry.get("pair_id")
                new_entry["tokens"] = entry.get("tokens")
                new_entry["spans"] = entry.get("spans")
                new_entry["tags"] = entry.get("tags")
                new_entry["order"] = entry.get("order")

            dataset.append(new_entry)

        random.shuffle(dataset)
        with open(out_path, "w", encoding="utf-8") as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Built {out_path} with {len(dataset)} samples.")

    # 4. 4개 파일 빌드
    print(f"\nBuilding {lang.upper()} Train/Dev files...")
    build_file(out_exp_train, train_ok, train_vio, is_explicit=True)
    build_file(out_exp_dev, dev_ok, dev_vio, is_explicit=True)
    build_file(out_imp_train, train_ok, train_vio, is_explicit=False)
    build_file(out_imp_dev, dev_ok, dev_vio, is_explicit=False)


# --- (수정) write_test_files 함수: 단순화 ---
def write_test_files(
        eng_test_ok: list, eng_test_vio: list,
        arla_test_ok: list, arla_test_vio: list
):
    """
    역할: 영어 테스트 셋과 인공어 테스트 셋 파일 2개를 생성합니다.
    모든 테스트 파일은 'implicit' 포맷입니다.
    """

    def _format_test_entry(entry, lang):
        """ Test 샘플용 'implicit' 포맷을 생성하는 내부 헬퍼 """
        meta = entry.get("meta", {}).copy()

        # 언어 정보 설정
        if lang == "eng":
            meta["language"] = "english"
            id_suffix = "_test_eng"
        else:
            meta["language"] = meta.get("language", "artificial")
            meta["rule"] = "SOV_word_order"
            id_suffix = "_test_arla"

        return {
            "id": entry["id"].replace("_ok", id_suffix).replace("_vi", id_suffix),
            "type": "implicit",
            "prompt": "",
            "text": entry["text"],
            "label": entry["label"],
            "meta": meta
        }

    def build_single_test_file(out_path, ok_list, vio_list, lang):
        dataset = []
        for entry in (ok_list + vio_list):
            dataset.append(_format_test_entry(entry, lang))

        random.shuffle(dataset)
        with open(out_path, "w", encoding="utf-8") as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Built {out_path} with {len(dataset)} samples.")

    print("\nBuilding Simplified TEST files (Eng and ArLa)...")

    # 영어 테스트 셋 (400 샘플)
    build_single_test_file(OUTPUT_TEST_ENG, eng_test_ok, eng_test_vio, "eng")

    # 인공어 테스트 셋 (400 샘플)
    build_single_test_file(OUTPUT_TEST_ARLA, arla_test_ok, arla_test_vio, "arla")


# --- main 함수 (수정) ---
def main():
    # 출력 디렉터리 생성 로직
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Ensuring output directory exists: {OUTPUT_DIR}")

    # 1. 영어 로드 (길이 필터 적용)
    print("Loading annotated English pairs...")
    eng_ok, eng_vio, _ = load_annotated_pairs(INPUT_ENGLISH_PAIRS, is_english=True)
    print(f"Loaded {len(eng_ok)} 'ok' and {len(eng_vio)} 'violation' English samples (after filtering).")

    # 2. 인공어 로드 (형태 집계)
    print("Loading annotated Artificial Language pairs...")
    arla_ok_all, arla_vio_all, morph_counts = load_annotated_pairs(INPUT_ARTLANG_PAIRS, is_english=False)
    print(f"Loaded {len(arla_ok_all)} 'ok' and {len(arla_vio_all)} 'violation' ArLa samples.")

    # 3. 인공어 형태 비율 검증
    verify_morphology(morph_counts)

    # 4. 🚨 OOD 스플릿 로직 제거
    # 이제 OOD 데이터는 사용하지 않고 전체 ArLa 데이터를 IID 풀로 간주합니다.
    arla_ok_iid_pool = arla_ok_all
    arla_vio_iid_pool = arla_vio_all

    print(f"ArLa IID Pool size (Total): {len(arla_ok_iid_pool)} ok, {len(arla_vio_iid_pool)} vio")

    # 5. 모든 풀 셔플
    random.shuffle(eng_ok)
    random.shuffle(eng_vio)
    random.shuffle(arla_ok_iid_pool)
    random.shuffle(arla_vio_iid_pool)

    # 6. 고정 개수 계산 (TARGET_TOTAL_PAIRS 기반)

    # 필요한 최소 개수 확인
    if min(len(eng_ok), len(eng_vio)) < TARGET_TOTAL_PAIRS:
        print(
            f"Fatal Error: Not enough balanced English pairs. Needed {TARGET_TOTAL_PAIRS}, Found {min(len(eng_ok), len(eng_vio))}.")
        exit(1)
    if min(len(arla_ok_iid_pool), len(arla_vio_iid_pool)) < TARGET_TOTAL_PAIRS:
        print(
            f"Fatal Error: Not enough balanced ArLa IID pairs. Needed {TARGET_TOTAL_PAIRS}, Found {min(len(arla_ok_iid_pool), len(arla_vio_iid_pool))}.")
        exit(1)

    # --- 고정 개수 계산 ---
    NUM_TOTAL = TARGET_TOTAL_PAIRS  # 2000
    NUM_TEST = int(NUM_TOTAL * TEST_RATIO)  # 200
    NUM_TRAIN_DEV = NUM_TOTAL - NUM_TEST  # 1800

    NUM_DEV = int(NUM_TRAIN_DEV * (DEV_RATIO / (TRAIN_RATIO + DEV_RATIO)))  # 200
    NUM_TRAIN = NUM_TRAIN_DEV - NUM_DEV  # 1600

    print(f"\n--- Final Fixed Dataset Counts ({NUM_TOTAL} pairs total per language) ---")
    print(f"  Train: {NUM_TRAIN} pairs ({NUM_TRAIN * 2} samples)")
    print(f"  Dev:   {NUM_DEV} pairs ({NUM_DEV * 2} samples)")
    print(f"  Test:  {NUM_TEST} pairs ({NUM_TEST * 2} samples)")

    # 7. 데이터 풀 슬라이싱 (고정 개수 추출 및 분할)

    # 7-A. 영어 (Eng) - NUM_TOTAL 만큼만 사용
    eng_used_ok = eng_ok[:NUM_TOTAL]
    eng_used_vio = eng_vio[:NUM_TOTAL]

    # Test 풀 (10% - NUM_TEST)
    eng_test_ok = eng_used_ok[:NUM_TEST]
    eng_test_vio = eng_used_vio[:NUM_TEST]

    # Train/Dev 풀 (90% - 나머지)
    eng_train_dev_ok = eng_used_ok[NUM_TEST:]
    eng_train_dev_vio = eng_used_vio[NUM_TEST:]

    # 7-B. 인공어 (ArLa) IID - NUM_TOTAL 만큼만 사용
    arla_used_ok = arla_ok_iid_pool[:NUM_TOTAL]
    arla_used_vio = arla_vio_iid_pool[:NUM_TOTAL]

    # Test 풀 (10% - NUM_TEST)
    arla_test_ok = arla_used_ok[:NUM_TEST]
    arla_test_vio = arla_used_vio[:NUM_TEST]

    # Train/Dev 풀 (90% - 나머지)
    arla_train_dev_ok = arla_used_ok[NUM_TEST:]
    arla_train_dev_vio = arla_used_vio[NUM_TEST:]

    # 8. Train/Dev 파일 빌드 실행 (90% 데이터 사용)
    write_dataset_files(
        eng_train_dev_ok, eng_train_dev_vio, "eng",
        num_train_pairs=NUM_TRAIN,
        num_dev_pairs=NUM_DEV
    )

    write_dataset_files(
        arla_train_dev_ok, arla_train_dev_vio, "arla",
        num_train_pairs=NUM_TRAIN,
        num_dev_pairs=NUM_DEV
    )

    # 9. Test 파일 빌드 실행 (10% 데이터 사용)
    write_test_files(
        eng_test_ok=eng_test_ok,
        eng_test_vio=eng_test_vio,
        arla_test_ok=arla_test_ok,
        arla_test_vio=arla_test_vio
    )

    print("\nAll datasets built successfully!")


if __name__ == "__main__":
    main()