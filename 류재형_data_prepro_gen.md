By 류재형

이 디렉토리에는 언어 모델의 명시적(Explicit) 및 암시적(Implicit) 규칙 학습 능력을 테스트하기 위해 구축된 데이터셋과 생성 스크립트가 포함되어 있습니다.

데이터셋은 두 가지 주요 소스로 구성됩니다:

1. 영어 (SVO): Simple English Wikipedia에서 추출 및 정제된 SVO(주어-동사-목적어) 구조의 문장.
2. 인공어 (ArLa): Brocanto2 스타일의 문법 규칙(성 일치, 후치 수식 등)을 따르는 SOV(주어-목적어-동사) 구조의 문장.


-----


💻 코드 파이프라인 (Code Pipeline)
데이터셋은 다음 4개의 Python 스크립트를 통해 순차적으로 생성됩니다.

- run_extract.py:
    역할: 1단계 (전처리).
    Simple Wikipedia 덤프 파일(*.xml.bz2)을 입력받아, 정규 표현식(Regex)을 사용해 위키 마크업을 제거하고 순수 텍스트 아티클(wiki_extracted.txt)을 추출합니다.
    + 위키 마크업이 완벽하게 지워지지 않아 잔여물이 존재합니다. 이는 simple_wiki_parser.py에서 마저 필터링 됩니다.

- simple_wiki_parser.py:
    역할: 2단계 (영어 데이터 생성).
    wiki_extracted.txt 파일을 읽어들입니다.
    spaCy를 사용해 문장을 파싱하고, 설정된 필터(잔여물 제거, 6~25 토큰 길이, 단순 SVO 구조)를 통과하는 문장만 선별합니다.
    정상 문장(label: ok)과 어순이 교란된 문장(label: violation) 쌍을 생성하여 english_annotated_pairs.jsonl로 저장합니다.

- artlang_generator.py:
    역할: 2단계 (인공어 데이터 생성).
    팀장님이 정의한 SOV 문법 규칙(성 일치, 후치 수식, 복수형 등)에 따라 인공어(ArLa) 문장을 생성합니다.
    정상 문장(SOV, label: ok)과 어순이 교란된 문장(SVO, label: violation) 쌍을 생성하여 artlang_annotated_pairs.jsonl로 저장합니다.

- build_datasets.py:
    역할: 3단계 (최종 데이터셋 구축).
    english...jsonl과 artlang...jsonl 파일을 로드합니다.
    요청된 조건(영어/인공어 분리, 명시적/암시적 분리)에 따라 데이터를 가공하고 셔플하여 최종 5개의 데이터셋 파일을 생성합니다.


-----


🗃️ 최종 데이터셋 (Final Datasets)
이 파이프라인은 총 5개의 .jsonl 학습 및 평가 파일을 생성합니다.

- train_eng_explicit.jsonl
    언어: 영어 (SVO)
    유형: 명시적 학습 (Explicit)
    개수: 2,000개 (ok 1000 + violation 1000)

`{"id": "simp_001295_ok", "pair_id": "simp_001295", "text": "They will also sometimes hunt birds.", "label": "ok", "tokens": ["They", "will", "also", "sometimes", "hunt", "birds", "."], "spans": {"subject": [25, 25], "verb": [29, 29], "object": [30, 30], "adv": [28, 28]}, "tags": {"bio_s": ["O", "O", "O", "O", "O", "O", "O"], "bio_v": ["O", "O", "O", "O", "O", "O", "O"], "bio_o": ["O", "O", "O", "O", "O", "O", "O"], "bio_adv": ["O", "O", "O", "O", "O", "O", "O"]}, "order": "SVO", "meta": {"rule": "SVO_word_order", "language": "english", "length": 7, "source": "simplewiki", "parser_confidence": 0.95}, "type": "explicit", "prompt": "Rule: Subject-Verb-Object order (adverb optional, sentence-final). Example: The dog eats the bone."}`

- train_eng_implicit.jsonl
    언어: 영어 (SVO)
    유형: 암시적 학습 (Implicit)
    개수: 2,000개 (ok 1000 + violation 1000)

`{"id": "simp_001389_imp", "type": "implicit", "prompt": "", "text": "A few years later, in 1774, British scientist Joseph Priestley also discovered this gas by heating a substance called mercuric oxide.", "label": "ok", "meta": {"language": "english", "length": 24}}
`

- train_arla_explicit.jsonl
    언어: 인공어 (SOV)
    유형: 명시적 학습 (Explicit)
    개수: 2,000개 (ok 1000 + violation 1000)

`{"id": "art_000415_vi", "type": "explicit", "prompt": "Rule: Subject-Object-Verb order (adverb optional, sentence-final). Example: pleck li vode lu praz noyka", "text": "pleck gloke li flig brip pelie li vogo", "label": "violation", "meta": {"structure": "SVO+ADV", "s": {"base": "pleck", "gender": "m", "is_plural": false, "plural_type": null, "adj": "glok"}, "o": {"base": "brip", "gender": "m", "is_plural": false, "plural_type": null, "adj": "peli"}, "length": 8, "language": "artificial", "rule": "SOV_word_order"}}
`


- train_arla_implicit.jsonl
    언어: 인공어 (SOV)
    유형: 암시적 학습 (Implicit)
    개수: 2,000개 (ok 1000 + violation 1000)

`{"id": "art_001487_imp", "type": "implicit", "prompt": "", "text": "klim pelie li klin lorf lu", "label": "violation", "meta": {"structure": "SVO", "s": {"base": "klim", "gender": "m", "is_plural": false, "plural_type": null, "adj": "peli"}, "o": {"base": "lorf", "gender": "f", "is_plural": false, "plural_type": null, "adj": null}, "length": 6, "language": "artificial"}}
`


- test.jsonl
    언어: 인공어 (SOV)
    유형: 평가 (Test)
    개수: 500개 (ok 250 + violation 250)

`{"id": "art_001115_test", "type": "implicit", "prompt": "", "text": "klim neime li glim troiso lu nim", "label": "ok", "meta": {"structure": "SOV", "s": {"base": "klim", "gender": "m", "is_plural": false, "plural_type": null, "adj": "neim"}, "o": {"base": "glim", "gender": "f", "is_plural": false, "plural_type": null, "adj": "trois"}, "length": 7, "language": "artificial"}}
`
