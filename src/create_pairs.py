import json
import random
import re
import spacy
from spacy.tokens import Doc
from tqdm import tqdm
import os
import sys

# --- 1. 설정 (Constants) ---
INPUT_CANDIDATE_FILE = "candidate_sentences.txt"  # Script 1의 출력
OUTPUT_JSONL = "english_annotated_pairs.jsonl"
MIN_TOKENS = 6
MAX_TOKENS = 25
TARGET_PAIRS = 3000
BATCH_SAVE_SIZE = 100
N_PROCESS = 4

# --- 2. spaCy 모델 로드 (: SVO 분석을 위해 Parser/Tagger 활성화) ---
MODEL_NAME = "en_core_web_sm"
try:
    # SVO 분석을 위해 tagger, parser 모두 필요. ner만 비활성화.
    NLP = spacy.load(MODEL_NAME, disable=["ner"])
    print(f"Loaded spaCy model: {MODEL_NAME} (Full parser/tagger active)")
except IOError:
    print(f"Error: '{MODEL_NAME}' model not found.")
    print(f"Please run: python -m spacy download {MODEL_NAME}")
    exit()


# --- 3. SVO 필터 함수 (: doc.start 제거) ---
def is_simple_active_svo(doc: Doc) -> bool:
    has_root_verb = False
    has_nsubj = False
    has_obj = False

    # : 필터 완화 (conj, acl 제거)
    complex_deps = {"auxpass", "nsubjpass", "advcl", "relcl", "ccomp", "xcomp"}

    for token in doc:
        # FIX: doc.start 대신 0을 사용 (doc의 첫 번째 토큰 인덱스는 0)
        if (token.dep_ in ("advmod", "mark") and token.i == 0):
            return False
        if token.text == "?" or (token.dep_ == "aux" and token.i == 0):
            return False
        if token.dep_ in complex_deps:
            return False
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            has_root_verb = True
        if token.dep_ == "nsubj":
            has_nsubj = True
        if token.dep_ in ("obj", "dobj"):
            has_obj = True

    return has_root_verb and has_nsubj and has_obj


# --- 4. SVO 스팬 함수 (: doc.start 제거) ---
def get_token_spans(doc: Doc):
    """
    역할: S, V, O, ADV의 토큰 인덱스 범위를 찾습니다.
    (: Doc 객체이므로 span_start는 0입니다.)
    """
    spans = {"subject": None, "verb": None, "object": None, "adv": None}
    adv_spans = []

    # FIX: doc.start가 존재하지 않음. Doc 객체의 시작은 0입니다.
    span_start = 0

    for token in doc:
        rel_i = token.i - span_start  # (항상 token.i와 동일)

        if token.dep_ == "nsubj":
            spans["subject"] = (token.left_edge.i - span_start, token.right_edge.i - span_start)
        elif token.dep_ == "ROOT" and token.pos_ == "VERB":
            spans["verb"] = (rel_i, rel_i)
        elif token.dep_ in ("obj", "dobj"):
            spans["object"] = (token.left_edge.i - span_start, token.right_edge.i - span_start)
        elif token.pos_ == "ADV" or token.dep_ == "advmod":
            if rel_i >= (len(doc) - 2) and not token.is_punct:
                adv_spans.append((rel_i, rel_i))

    if adv_spans:
        spans["adv"] = adv_spans[-1]
    return spans


# --- 5. BIO 태그 생성 함수 (스팬 오류 해결로 정상 작동) ---
def generate_bio_tags(tokens: list, spans: dict):
    tags = {"bio_s": ["O"] * len(tokens), "bio_v": ["O"] * len(tokens),
            "bio_o": ["O"] * len(tokens), "bio_adv": ["O"] * len(tokens)}

    def fill_tags(tag_type, span_key):
        span = spans.get(span_key)
        if span:
            start, end = span
            if start < 0 or start >= len(tokens): return
            tags[tag_type][start] = f"B-{span_key.upper()}"
            for i in range(start + 1, end + 1):
                if i < len(tokens):
                    tags[tag_type][i] = f"I-{span_key.upper()}"

    fill_tags("bio_s", "subject")
    fill_tags("bio_v", "verb")
    fill_tags("bio_o", "object")
    fill_tags("bio_adv", "adv")
    return tags


# --- 6. SVO->SOV 교란 함수 (7-Part Slice, 스팬 오류 해결로 정상 작동) ---
def create_violation_sentence(doc: Doc, spans: dict):
    if not all([spans.get("subject"), spans.get("verb"), spans.get("object")]):
        return None, "SOV_FAIL", {}
    try:
        s_start, s_end = spans["subject"]
        v_start, v_end = spans["verb"]
        o_start, o_end = spans["object"]

        if not (s_end < v_start and v_end < o_start):
            return None, "SVO_ORDER_ISSUE", {}

        part1_start_to_S = doc[0: s_start]
        chunk_S = doc[s_start: s_end + 1]
        part2_S_to_V = doc[s_end + 1: v_start]
        chunk_V = doc[v_start: v_end + 1]
        part3_V_to_O = doc[v_end + 1: o_start]
        chunk_O = doc[o_start: o_end + 1]
        part4_O_to_End = doc[o_end + 1:]

        new_tokens = []
        new_tokens.extend([t.text for t in part1_start_to_S])
        new_tokens.extend([t.text for t in chunk_S])
        new_tokens.extend([t.text for t in part2_S_to_V])
        new_tokens.extend([t.text for t in chunk_O])  # O 청크
        new_tokens.extend([t.text for t in part3_V_to_O])
        new_tokens.extend([t.text for t in chunk_V])  # V 청크
        new_tokens.extend([t.text for t in part4_O_to_End])

        violation_text = " ".join(new_tokens)
        violation_text = re.sub(r'\s([,.?!;:])', r'\1', violation_text)
        violation_text = re.sub(r'\s+', ' ', violation_text).strip()

        violation_doc = NLP(violation_text)
        violation_spans = get_token_spans(violation_doc)  # 교란된 문장도 상대 인덱스

        return violation_text, "SOV_generated", violation_spans

    except Exception as e:
        return None, "ERROR", {}


# --- 7. 배치 쓰기 함수 ---
def process_batch(batch_data, file_handle):
    for entry in batch_data:
        file_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    file_handle.flush()


# --- 8. 메인 함수 (Pass 2: SVO 필터링) ---
def main():
    if not os.path.exists(INPUT_CANDIDATE_FILE):
        print(f"Error: Input candidate file not found at '{INPUT_CANDIDATE_FILE}'.")
        print("Please run pre_filter.py first to generate this file.")
        return

    print(f"Step 1: Reading candidate sentences from {INPUT_CANDIDATE_FILE}...")
    try:
        with open(INPUT_CANDIDATE_FILE, "r", encoding="utf-8") as f_in:
            candidate_sentences = [line.strip() for line in f_in if line.strip()]
    except Exception as e:
        print(f"Error reading candidate file: {e}")
        return

    total_candidates = len(candidate_sentences)
    print(f"Total {total_candidates} candidate sentences loaded. Starting SVO filtering...")

    n_process_to_use = N_PROCESS
    if os.name == "nt" and N_PROCESS > 1:
        print("Note: On Windows, spaCy multiprocessing can be problematic. Setting n_process=1.")
        n_process_to_use = 1

    annotated_pairs_count = 0
    sample_batch = []

    desc = f"Pass 2: SVO filtering (Target: {TARGET_PAIRS} pairs)"

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        # nlp.pipe() - SVO 분석을 위해 Parser/Tagger 활성화
        pipe = NLP.pipe(candidate_sentences, batch_size=500, n_process=n_process_to_use)

        try:
            for doc in tqdm(pipe, total=total_candidates, desc=desc, file=sys.stdout):

                text = doc.text.strip()

                # SVO 필터 적용 (길이 필터는 이미 Pass 1에서 통과)
                if is_simple_active_svo(doc):

                    pair_id = f"simp_{annotated_pairs_count + 1:06d}"
                    ok_spans = get_token_spans(doc)  # : 상대 인덱스

                    vio_text, vio_order, vio_spans = create_violation_sentence(doc, ok_spans)

                    if ok_spans.get("subject") and ok_spans.get("verb") and ok_spans.get("object") and vio_text:

                        vio_doc = NLP(vio_text)
                        vio_tokens_no_punct = [t for t in vio_doc if not t.is_punct]
                        if not (MIN_TOKENS <= len(vio_tokens_no_punct) <= MAX_TOKENS):
                            continue

                        ok_tokens_list = [t.text for t in doc]
                        vio_tokens_list = [t.text for t in vio_doc]

                        # 'ok' (SVO) 데이터
                        sample_batch.append({
                            "id": f"{pair_id}_ok", "pair_id": pair_id, "text": text,
                            "label": "ok", "tokens": ok_tokens_list,
                            "spans": ok_spans,
                            "tags": generate_bio_tags(ok_tokens_list, ok_spans),
                            "order": "SVO", "meta": {"source": "simplewiki"}
                        })
                        # 'violation' (SOV) 데이터
                        sample_batch.append({
                            "id": f"{pair_id}_vi", "pair_id": pair_id, "text": vio_text,
                            "label": "violation", "tokens": vio_tokens_list,
                            "spans": vio_spans,
                            "tags": generate_bio_tags(vio_tokens_list, vio_spans),
                            "order": vio_order,
                            "meta": {"source": "simplewiki", "derived_from": f"{pair_id}_ok"}
                        })

                        annotated_pairs_count += 1

                        if len(sample_batch) >= BATCH_SAVE_SIZE:
                            process_batch(sample_batch, f_out)
                            sample_batch.clear()

                    if annotated_pairs_count >= TARGET_PAIRS:
                        break  # 목표 달성 시 즉시 종료

        except Exception as e:
            print(f"\nAn error occurred during processing: {e}")
        finally:
            if sample_batch:
                process_batch(sample_batch, f_out)

            total_samples = annotated_pairs_count * 2
            print(
                f"\nSuccessfully generated {total_samples} annotated English samples ({annotated_pairs_count} pairs).")
            print(f"Saved to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()