import re
import spacy
from tqdm import tqdm
import os
import sys

# --- 1. 설정 (Constants) ---
INPUT_WIKI_FILE = r"C:\Users\rjh32\PyCharmMiscProject\NLP_homework\Team\output\wiki_extracted.txt"
OUTPUT_CANDIDATE_FILE = "candidate_sentences.txt"
MIN_TOKENS = 6
MAX_TOKENS = 25
N_PROCESS = 4

# --- 2. spaCy 모델 로드 ( Tagger 활성화, Parser 비활성화, Sentencizer 추가) ---
MODEL_NAME = "en_core_web_sm"
try:
    #   Tagger는 활성화 (is_punct 신뢰성 확보). Parser, NER만 비활성화.
    NLP = spacy.load(MODEL_NAME, disable=["parser", "ner"])

    # 💡 [E030] 오류 해결을 위해 sentencizer 추가
    if "sentencizer" not in NLP.pipe_names:
        NLP.add_pipe("sentencizer")

    print(f"Loaded spaCy model: {MODEL_NAME} (Parser/NER disabled, Tagger/Sentencizer active)")
except IOError:
    print(f"Error: '{MODEL_NAME}' model not found.")
    print(f"Please run: python -m spacy download {MODEL_NAME}")
    exit()


# --- 3. 찌꺼기 필터 함수 (v5: .ref 패턴 포함) ---
def contains_junk(text: str) -> bool:
    if not text:
        return True
    text_lower = text.lower().strip()

    junk_patterns = [
        r'==', r'\|', r'thumb', r'\bfile:', r'\bcategory:', r'\bwikipedia:',
        r'^\s*[\*#:]+', r'\{\{', r'\[\[', r'^li\s+style\s*=',
        r'background-color:', r'rgb\s*\(', r'#([0-G-9a-f]{3}){1,2}\b',
        r'/\s*li\s*>$', r'math(x|y|z)\b', r'\\(cdot|sum|frac|mathrm)',
        r'[=+/]\s*math\b', r'</math>', r'\.ref\b', r'\bref\s*/>', r'</ref>'
    ]

    for pattern in junk_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


# --- 4. 메인 함수 (Pass 1: 후보 수집) ---
def main():
    if not os.path.exists(INPUT_WIKI_FILE):
        print(f"Error: Input file not found at '{INPUT_WIKI_FILE}'.")
        return

    print(f"Step 1: Reading and preliminary filtering of {INPUT_WIKI_FILE}...")
    try:
        with open(INPUT_WIKI_FILE, "r", encoding="utf-8") as f_in:
            lines = [line.strip() for line in f_in if line.strip() and not contains_junk(line)]
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    total_lines = len(lines)
    print(f"Total {total_lines} clean lines (paragraphs) loaded.")

    n_process_to_use = N_PROCESS
    if os.name == "nt" and N_PROCESS > 1:
        print("Note: On Windows, spaCy multiprocessing can be problematic. Setting n_process=1.")
        n_process_to_use = 1

    candidate_sentences = []
    desc = f"Pass 1: Collecting candidates (Length filter only) from {total_lines} paragraphs"

    #  nlp.pipe() - Tagger가 활성화되어 is_punct가 정확해짐
    pipe = NLP.pipe(lines, batch_size=500, n_process=n_process_to_use)

    for paragraph_doc in tqdm(pipe, total=total_lines, desc=desc, file=sys.stdout):
        for doc in paragraph_doc.sents:
            text = doc.text.strip()
            if not text:
                continue

            #   Tagger가 활성화되어 구두점(is_punct) 인식이 정확해짐
            tokens_no_punct = [t for t in doc if not t.is_punct]

            if (MIN_TOKENS <= len(tokens_no_punct) <= MAX_TOKENS):
                candidate_sentences.append(text)

    # 5. 후보 문장 파일로 저장
    print(f"\nStep 2: Collected {len(candidate_sentences)} candidate sentences.")
    try:
        with open(OUTPUT_CANDIDATE_FILE, "w", encoding="utf-8") as f_out:
            for sentence in candidate_sentences:
                f_out.write(sentence + "\n")
        print(f"Successfully saved candidate sentences to {OUTPUT_CANDIDATE_FILE}")
    except Exception as e:
        print(f"Error writing candidate file: {e}")


if __name__ == "__main__":
    main()