import random
import json
import math
from tqdm import tqdm
import os

# --- 설정 (Constants) ---
OUTPUT_JSONL = "artlang_annotated_pairs.jsonl"
# 1. 목표 샘플 수 수정 (Train 2000(1000쌍) + Implicit 2000(1000쌍) + Test 500(250쌍))
TARGET_PAIRS = 2250  # 총 2250쌍 (4500 샘플)
MIN_TOKENS = 6
MAX_TOKENS = 25

# --- (어휘 정의는 이전과 동일) ---
NOUNS = {"pleck": "m", "vode": "f", "klim": "m", "zuna": "f", "brip": "m", "lorf": "f", "niff": "m", "glim": "f"}
ADJECTIVES = {"trois": {"m": "troise", "f": "troiso"}, "neim": {"m": "neime", "f": "neimo"},
              "peli": {"m": "pelie", "f": "pelio"}, "glok": {"m": "gloke", "f": "gloko"}}
ARTICLES = {"m": "li", "f": "lu"}
VERBS = ["klin", "nim", "yab", "praz", "flig", "droz"]
ADVERBS = ["noyka", "zayma", "dema", "vogo"]
PLURAL_RULES = {"regular": "-ka", "irreg1": "-po", "irreg2": "-lee"}


def get_zipfian_word(vocab_list):
    return random.choice(vocab_list)


def make_np():
    noun_base = get_zipfian_word(list(NOUNS.keys()))
    gender = NOUNS[noun_base]

    is_plural = random.random() < 0.3
    noun = noun_base
    plural_type = None
    if is_plural:
        rand_plural = random.random()
        if rand_plural < 0.7:
            noun += PLURAL_RULES["regular"]; plural_type = "regular"
        elif rand_plural < 0.85:
            noun += PLURAL_RULES["irreg1"]; plural_type = "irreg1"
        else:
            noun += PLURAL_RULES["irreg2"]; plural_type = "irreg2"

    np_tokens = [noun]

    use_adj = random.random() < 0.5
    adj_stem = None
    if use_adj:
        adj_stem = get_zipfian_word(list(ADJECTIVES.keys()))
        np_tokens.append(ADJECTIVES[adj_stem][gender])

    np_tokens.append(ARTICLES[gender])

    meta = {"base": noun_base, "gender": gender, "is_plural": is_plural, "plural_type": plural_type, "adj": adj_stem}
    return np_tokens, meta


def make_sentence_pair():
    np_s_tokens, meta_s = make_np()
    while True:
        np_o_tokens, meta_o = make_np()
        if meta_s["base"] != meta_o["base"]: break

    verb_token = [get_zipfian_word(VERBS)]

    adv_token = []
    if random.random() < 0.5:
        adv_token = [get_zipfian_word(ADVERBS)]

    ok_tokens = np_s_tokens + np_o_tokens + verb_token + adv_token
    ok_text = " ".join(ok_tokens)
    ok_meta = {"structure": "SOV+ADV" if adv_token else "SOV", "s": meta_s, "o": meta_o}

    vio_tokens = np_s_tokens + verb_token + np_o_tokens + adv_token
    vio_text = " ".join(vio_tokens)
    vio_meta = {"structure": "SVO+ADV" if adv_token else "SVO", "s": meta_s, "o": meta_o}

    return (ok_text, ok_meta, ok_tokens), (vio_text, vio_meta, vio_tokens)


def get_existing_pair_count(filepath: str) -> int:
    """
    역할: 스크립트 재시작 시, 이미 생성된 샘플 수를 셉니다.
    """
    if not os.path.exists(filepath):
        return 0
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        # 각 쌍(pair)은 'ok'와 'violation' 2줄로 구성됨
        return line_count // 2
    except Exception as e:
        print(f"Warning: Could not read existing file {filepath}. Overwriting. Error: {e}")
        return 0


def main():
    # 2. 재시작/이어하기 기능 (Checkpointing)
    existing_pairs = get_existing_pair_count(OUTPUT_JSONL)
    pairs_to_generate = TARGET_PAIRS - existing_pairs

    if pairs_to_generate <= 0:
        print(f"Target of {TARGET_PAIRS} pairs already met or exceeded in {OUTPUT_JSONL}.")
        print(f"Found {existing_pairs} existing pairs.")
        return

    print(f"Found {existing_pairs} existing pairs. Generating {pairs_to_generate} new pairs...")

    # 3. 파일을 'a'(append, 이어쓰기) 모드로 엽니다.
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f_out:
        pbar = tqdm(desc="Generating Artificial Language (SOV) pairs", total=pairs_to_generate)

        generated_count = 0
        while generated_count < pairs_to_generate:
            (ok_text, ok_meta, ok_tokens), (vio_text, vio_meta, vio_tokens) = make_sentence_pair()

            if not (MIN_TOKENS <= len(ok_tokens) <= MAX_TOKENS):
                continue

            # 'ok'와 'violation'은 항상 쌍으로 저장
            pair_id_num = existing_pairs + generated_count + 1
            pair_id = f"art_{pair_id_num:06d}"

            # 'ok' (SOV)
            ok_entry = {
                "id": f"{pair_id}_ok", "pair_id": pair_id, "text": ok_text, "label": "ok",
                "meta": {**ok_meta, "length": len(ok_tokens), "language": "artificial"}
            }

            # 'violation' (SVO)
            vio_entry = {
                "id": f"{pair_id}_vi", "pair_id": pair_id, "text": vio_text, "label": "violation",
                "meta": {**vio_meta, "length": len(vio_tokens), "language": "artificial"}
            }

            # 4. 실시간 저장 (파일에 즉시 쓰기)
            f_out.write(json.dumps(ok_entry, ensure_ascii=False) + "\n")
            f_out.write(json.dumps(vio_entry, ensure_ascii=False) + "\n")
            f_out.flush()  # 디스크에 즉시 반영

            generated_count += 1
            pbar.update(1)

        pbar.close()

    print(f"Generation complete. Total pairs in file: {existing_pairs + generated_count}")


if __name__ == "__main__":
    main()
