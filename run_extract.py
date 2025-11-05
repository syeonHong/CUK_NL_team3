import bz2
import re
import os
from tqdm import tqdm

# === 경로 설정 ===
INPUT_FILE = r"C:\Users\rjh32\PyCharmMiscProject\NLP_homework\Team\simplewiki-20251101-pages-articles-multistream.xml.bz2"
OUTPUT_DIR = r"C:\Users\rjh32\PyCharmMiscProject\NLP_homework\Team\output"

# === 출력 폴더 생성 ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 위키마크업 제거 함수 ===
def clean_wiki_markup(text: str) -> str:
    """기본적인 위키 문법 제거 (링크, 템플릿, 카테고리 등)"""
    text = re.sub(r"\[\[.*?\|(.*?)\]\]", r"\1", text)  # [[링크|표시]] → 표시
    text = re.sub(r"\[\[(.*?)\]\]", r"\1", text)        # [[링크]] → 링크
    text = re.sub(r"\{\{.*?\}\}", "", text)             # {{템플릿}} 제거
    text = re.sub(r"<ref.*?>.*?</ref>", "", text)       # <ref>...</ref> 제거
    text = re.sub(r"<[^>]+>", "", text)                 # HTML 태그 제거
    text = re.sub(r"&[a-z]+;", "", text)                # &amp; 등 제거
    text = re.sub(r"'{2,}", "", text)                   # ''이탤릭'' → 제거
    return text.strip()

# === 본문 추출 ===
def extract_articles(xml_data):
    """<page> ... </page> 블록에서 제목과 본문을 추출"""
    pages = re.findall(r"<page>(.*?)</page>", xml_data, re.DOTALL)
    articles = []
    for page in pages:
        title_match = re.search(r"<title>(.*?)</title>", page)
        text_match = re.search(r"<text.*?>(.*?)</text>", page, re.DOTALL)
        if title_match and text_match:
            title = title_match.group(1)
            text = clean_wiki_markup(text_match.group(1))
            if text.strip():
                articles.append({"title": title, "text": text})
    return articles

# === bz2 압축 파일 열기 ===
print(f"Reading dump: {INPUT_FILE}")
with bz2.open(INPUT_FILE, "rt", encoding="utf-8", errors="ignore") as f:
    xml_content = f.read()

# === 문서 추출 ===
print("Extracting articles... (this may take a while)")
articles = extract_articles(xml_content)

# === 저장 ===
output_path = os.path.join(OUTPUT_DIR, "wiki_extracted.txt")
with open(output_path, "w", encoding="utf-8") as out:
    for article in tqdm(articles, desc="Saving"):
        out.write(f"{article['title']}\n{article['text']}\n\n")

print(f"\n✅ Done! Extracted {len(articles)} articles.")
print(f"Output saved to: {output_path}")