# -*- coding: utf-8 -*-

# 규칙카드 A (기술형)
EXPLICIT_RULE_CARD_A = """[RULES]
1) SOV word order (Subject–Object–Verb).
2) The suffix -ka attaches only to A-class nouns.
"""

# 규칙카드 B (설명형)
EXPLICIT_RULE_CARD_B = """[RULES]
In this language, a correct sentence always follows SOV word order.
This means that the subject appears first, the object comes next,
and the verb must always appear at the very end of the sentence.

In addition, only certain nouns (A-class nouns) can take the suffix -ka.
If -ka is attached to a noun outside this class, the sentence becomes ungrammatical.
"""

# 예시문
EXAMPLES = """[EXAMPLE]
mika-ka ... ✓
miti-ka ... ✗
"""

def build_prompt(
    ex: dict,
    condition: str = "implicit",      # "implicit" / "explicit_a" / "explicit_b"
    for_eval: bool = False,
    task_type: str = "generation",
):
    # 평가모드 → 규칙카드 완전 제거
    if for_eval:
        return ex.get("text", "") or ex.get("sentence", "")

    # 기본 문장
    sent = (
        ex.get("text")
        or ex.get("text_pos")
        or ex.get("sentence")
        or ex.get("text_neg", "")
    )

    condition = (condition or "").lower()

    # 규칙카드 선택
    if condition == "explicit_a":
        rule_section = f"{EXPLICIT_RULE_CARD_A}\n{EXAMPLES}"
    elif condition == "explicit_b":
        rule_section = f"{EXPLICIT_RULE_CARD_B}\n{EXAMPLES}"
    else:  # implicit or rule-less
        rule_section = f"{EXAMPLES}"

    # --- Task Type Handling ---
    if task_type == "generation":
        prompt = f"{rule_section}[INPUT]\n{sent}"

    elif task_type == "grammaticality":
        prompt = (
            f"{rule_section}"
            f"[TASK]\n"
            f"Judge whether the following sentence is grammatically correct according to the given rules.\n"
            f"[SENTENCE]\n{sent}\n"
            f"Answer with 'Yes' if it is grammatical, or 'No' if it violates the rule."
        )

    elif task_type == "comparison":
        s1, s2 = ex.get("sentence_1"), ex.get("sentence_2")
        prompt = (
            f"{rule_section}"
            f"[TASK]\nWhich of the following sentences is grammatically correct?\n"
            f"1. {s1}\n2. {s2}\n"
            f"Answer with '1' or '2'."
        )

    elif task_type == "mcq":
        options = ex.get("options", [])
        question = ex.get("question", "Choose the grammatically correct sentence:")
        opts = "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))
        prompt = (
            f"{rule_section}"
            f"[TASK]\n{question}\n"
            f"{opts}\n"
            f"Answer with '1', '2', '3', '4', or '5'."
        )

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    return prompt
