import json
import random
import re
import spacy
from spacy.tokens import Doc
from tqdm import tqdm
import os
import glob

# --- 설정 (Constants) ---
# 1. 수정: 입력 파일 경로 수정 (단일 파일)
# 이전 단계에서 생성한 'wiki_extracted.txt' 파일의 전체 경로
INPUT_WIKI_FILE = r"C:\Users\rjh32\PyCharmMiscProject\NLP_homework\Team\output\wiki_extracted.txt"
OUTPUT_JSONL = "english_annotated_pairs.jsonl"
MIN_TOKENS = 6
MAX_TOKENS = 25

# 2. 목표 샘플 수 (변경 없음)
TARGET_PAIRS = 2000  # 총 2000쌍 (4000 샘플)
BATCH_SAVE_SIZE = 100  # 주기적 저장

# --- spaCy 모델 로드 (변경 없음) ---
MODEL_NAME = "en_core_web_sm"
try:
    NLP = spacy.load(MODEL_NAME)
    print(f"Loaded spaCy model: {MODEL_NAME}")
except IOError:
    print(f"Error: '{MODEL_NAME}' model not found.")
    print(f"Please run: python -m spacy download {MODEL_NAME}")
    exit()


# ---  강력한 잔여물 필터 함수 ---
def contains_junk(text: str) -> bool:
    """
    역할: spaCy 처리 전, 원시 텍스트에 위키 찌꺼기가 있는지 확인합니다.
    이유: 'thumb|...', 'File:', '==', '[[Category:' 등은
           SVO 필터를 통과할 수 있으므로 원천 차단합니다.
    """
    text_lower = text.lower()
    # 찌꺼기 징후 목록 (정규식 사용)
    junk_patterns = [
        r'==',  # 섹션 제목
        r'\|',  # 파이프 (템플릿/테이블 잔여물)
        r'thumb',  # 이미지 썸네일
        r'file:',  # 파일 링크
        r'category:',  # 카테고리 링크
        r'wikipedia:',  # 메타 링크
        r'^\s*\*',  # 리스트 마커로 시작
        r'\{\{',  # 템플릿 괄호
        r'\[\['  # 링크 괄호
        # Pro Fix v2: 문제의 HTML/CSS 잔여물 패턴 추가
        r'^li\s+style',  # 'li style ='로 시작하는 HTML 리스트 스타일
        r'background-color:',  # CSS 색상 정보
        r'rgb:\s*\d{1,3}',  # RGB 값 패턴
        r'hex:\s*#',  # Hex 코드 패턴
        r'/\s*li$',  # '/li'로 끝나는 태그 (토큰 분리 문제 방지)
        r'mathx',  # 'math' 태그 내부의 변수 표기
        r'mathy',
        r'mathz',
        r'\\cdot',  # LaTeX 곱셈 기호
        r'\\',  # 백슬래시 (LaTeX 잔여물)
        r'[=+/]\s*math\b',  # '=' 또는 '+' 등으로 끝나고 'math'가 이어지는 패턴
    ]

    for pattern in junk_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


# --- (is_simple_active_svo, get_token_spans, generate_bio_tags 함수는 변경 없음) ---
def is_simple_active_svo(doc: Doc) -> bool:
    """
    역할: 팀장님의 필터링 지침을 적용하여 문장이 '단순 SVO 능동태'인지 검사합니다.
    """
    has_root_verb = False
    has_nsubj = False
    has_obj = False

    complex_deps = {
        "auxpass", "nsubjpass", "advcl", "relcl", "ccomp", "xcomp", "conj",
    }

    for token in doc:
        if token.text == "?" or (token.dep_ == "aux" and token.i == 0):
            return False
        if token.dep_ in complex_deps:
            return False
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            has_root_verb = True
        if token.dep_ == "nsubj":
            has_nsubj = True
        if token.dep_ == "obj" or token.dep_ == "dobj":
            has_obj = True

    return has_root_verb and has_nsubj and has_obj


def get_token_spans(doc: Doc):
    """
    역할: S, V, O, ADV의 토큰 인덱스 범위를 찾습니다.
    (Pro Note: 스팬을 더 정확하게 잡으려면 token.subtree를 돌아야 하지만,
     현재 'sm' 모델과 SVO 필터 조합에서는 이 방식도 작동합니다.)
    """
    spans = {"subject": None, "verb": None, "object": None, "adv": None}
    adv_spans = []

    for token in doc:
        if token.dep_ == "nsubj":
            spans["subject"] = (token.left_edge.i, token.right_edge.i)
        elif token.dep_ == "ROOT" and token.pos_ == "VERB":
            spans["verb"] = (token.i, token.i)
        elif token.dep_ in ("obj", "dobj"):
            spans["object"] = (token.left_edge.i, token.right_edge.i)
        elif token.pos_ == "ADV" or token.dep_ == "advmod":
            # 문장 말미 부사 확인 (마침표 등 제외)
            if token.i >= len(doc) - 2 and not token.is_punct:
                adv_spans.append((token.i, token.i))

    if adv_spans:
        # 가장 마지막 부사 스팬을 선택
        spans["adv"] = adv_spans[-1]
    return spans


def generate_bio_tags(tokens: list, spans: dict):
    """
    역할: S, V, O, ADV 스팬 정보를 바탕으로 BIO 태그 시퀀스를 생성합니다.
    """
    tags = {
        "bio_s": ["O"] * len(tokens),
        "bio_v": ["O"] * len(tokens),
        "bio_o": ["O"] * len(tokens),
        "bio_adv": ["O"] * len(tokens),
    }

    def fill_tags(tag_type, span_key):
        span = spans.get(span_key)
        if span:
            start, end = span
            if start >= len(tokens): return  # 스팬 범위 오류 방어
            tags[tag_type][start] = f"B-{span_key.upper()}"
            for i in range(start + 1, end + 1):
                if i < len(tokens):  # 토큰 리스트 범위 내에서만 태깅
                    tags[tag_type][i] = f"I-{span_key.upper()}"

    fill_tags("bio_s", "subject")
    fill_tags("bio_v", "verb")
    fill_tags("bio_o", "object")
    fill_tags("bio_adv", "adv")

    return tags


# ---  교란 오류(중복) 해결된 함수 ---
def create_violation_sentence(doc: Doc, spans: dict):
    """
    역할: 'ok' (SVO) 문장을 'violation' (SOV) 문장으로 교란시킵니다.
    (Pro Fix: S/V/O 청크와 그 사이/주변 토큰을 정확히 재조합하여 중복 오류 해결)
    """
    if not all([spans["subject"], spans["verb"], spans["object"]]):
        return None, "SOV", {}

    try:
        # 1. 문장을 7개의 조각으로 분리
        s_start, s_end = spans["subject"]
        v_start, v_end = spans["verb"]
        o_start, o_end = spans["object"]

        # SVO 순서를 가정
        if not (s_start < v_start < o_start):
            # 스팬 순서가 꼬였거나(SVO가 아님) 겹치면(S V O V) 교란 생성 포기
            return None, "SVO_ORDER_ISSUE", {}

        part1_start_to_S = doc[0: s_start]
        chunk_S = doc[s_start: s_end + 1]
        part2_S_to_V = doc[s_end + 1: v_start]
        chunk_V = doc[v_start: v_end + 1]
        part3_V_to_O = doc[v_end + 1: o_start]
        chunk_O = doc[o_start: o_end + 1]
        part4_O_to_End = doc[o_end + 1:]

        # 2. SOV 순서로 재조합 (S + O + V)
        # S (Subject)
        new_tokens = [t.text for t in part1_start_to_S]
        new_tokens.extend([t.text for t in chunk_S])
        # O (Object) - V와 O 사이의 토큰(part3)을 O 앞으로 이동
        new_tokens.extend([t.text for t in part2_S_to_V])
        new_tokens.extend([t.text for t in part3_V_to_O])
        new_tokens.extend([t.text for t in chunk_O])
        # V (Verb)
        new_tokens.extend([t.text for t in chunk_V])
        # Rest (End)
        new_tokens.extend([t.text for t in part4_O_to_End])

        violation_text = " ".join(new_tokens)

        # 재조합 과정에서 생긴 공백 정리 (spaCy가 처리하므로 필수는 아님)
        violation_text = re.sub(r'\s([,.?!])', r'\1', violation_text)  # "hello ." -> "hello."
        violation_text = re.sub(r'\(\s', '(', violation_text)
        violation_text = re.sub(r'\s\)', ')', violation_text)
        violation_text = re.sub(r'\s+', ' ', violation_text).strip()

        # 3. 교란된 문장 다시 파싱
        violation_doc = NLP(violation_text)
        violation_spans = get_token_spans(violation_doc)

        return violation_text, "SOV_generated", violation_spans

    except Exception as e:
        # 스팬 인덱싱 중 오류 발생 시 (예: S V O 겹침)
        # print(f"Warning: Skipping violation generation for '{doc.text[:50]}...'. Error: {e}")
        return None, "ERROR", {}


def process_batch(batch_data, file_handle):
    """
    역할: 수집된 배치 데이터를 파일에 씁니다 (주기적 저장)
    """
    for entry in batch_data:
        file_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    file_handle.flush()  # 변경 사항을 디스크에 즉시 반영


# ---  'main' 함수 (단일 .txt 파일 입력) ---
def main():
    # 1. 수정: 단일 텍스트 파일 입력
    if not os.path.exists(INPUT_WIKI_FILE):
        print(f"Error: Input file not found at '{INPUT_WIKI_FILE}'.")
        print("Please run the previous Python extraction script first.")
        return

    print(f"Found 1 text file to process: {INPUT_WIKI_FILE}")

    annotated_pairs_count = 0
    processed_texts = set()
    sample_batch = []  # 주기적 저장을 위한 배치

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        pbar = tqdm(desc="Filtering Simple Wiki SVO sentences", total=TARGET_PAIRS)

        try:
            # 2. 수정: 단일 텍스트 파일을 라인별로 읽기
            with open(INPUT_WIKI_FILE, "r", encoding="utf-8") as f_in:
                for line in f_in:

                    line = line.strip()
                    if not line:
                        continue

                    # 3. 강력한 잔여물 필터 (라인 레벨)
                    if contains_junk(line):
                        continue

                    # 4. 수정: 문장 분리 로직 강화
                    # 이전 코드는 기사 전체를 NLP(line)로 처리하려 했으나,
                    # 여기서는 .txt 파일이 이미 라인/문단 단위이므로,
                    # 문단(line)을 받아 문장(doc.sents)으로 분리합니다.
                    paragraph_doc = NLP(line)
                    for doc in paragraph_doc.sents:  # doc은 이제 개별 문장(span)

                        text = doc.text.strip()
                        if not text or text in processed_texts:
                            continue
                        processed_texts.add(text)

                        # 5. 개별 문장에 대한 2차 잔여물 필터
                        if contains_junk(text):
                            continue

                        tokens = [t.text for t in doc if not t.is_punct]
                        if not (MIN_TOKENS <= len(tokens) <= MAX_TOKENS):
                            continue

                        if is_simple_active_svo(doc):
                            pair_id = f"simp_{annotated_pairs_count + 1:06d}"
                            ok_spans = get_token_spans(doc)

                            # 6. 교란 함수(create_violation_sentence) 호출
                            vio_text, vio_order, vio_spans = create_violation_sentence(doc, ok_spans)

                            # SVO, SVO 스팬, vio 생성이 모두 성공한 경우에만
                            if ok_spans["subject"] and ok_spans["verb"] and ok_spans["object"] and vio_text:

                                # 'ok' (SVO) 데이터
                                sample_batch.append({
                                    "id": f"{pair_id}_ok", "pair_id": pair_id, "text": text,
                                    "label": "ok", "tokens": [t.text for t in doc],
                                    "spans": ok_spans, "tags": generate_bio_tags([t.text for t in doc], ok_spans),
                                    "order": "SVO", "meta": {"source": "simplewiki"}
                                })

                                # 'violation' (SOV) 데이터
                                sample_batch.append({
                                    "id": f"{pair_id}_vi", "pair_id": pair_id, "text": vio_text,
                                    "label": "violation", "tokens": [t.text for t in NLP(vio_text)],
                                    "spans": vio_spans,
                                    "tags": generate_bio_tags([t.text for t in NLP(vio_text)], vio_spans),
                                    "order": vio_order,
                                    "meta": {"source": "simplewiki", "derived_from": f"{pair_id}_ok"}
                                })

                                annotated_pairs_count += 1
                                pbar.update(1)

                                # 7. 주기적 저장 (Checkpointing)
                                if len(sample_batch) >= BATCH_SAVE_SIZE:
                                    process_batch(sample_batch, f_out)
                                    sample_batch.clear()

                            if annotated_pairs_count >= TARGET_PAIRS:
                                raise StopIteration  # 목표 달성 시 모든 루프 탈출

        except StopIteration:
            print(f"\nTarget of {TARGET_PAIRS} pairs reached.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
        finally:
            # 8. 남은 배치 저장 및 완료
            if sample_batch:
                process_batch(sample_batch, f_out)
            pbar.close()
            total_samples = annotated_pairs_count * 2
            print(f"Successfully generated {total_samples} annotated English samples ({annotated_pairs_count} pairs).")
            print(f"Saved to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()