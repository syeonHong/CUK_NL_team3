# -*- coding: utf-8 -*-

EXPLICIT_RULE_CARD = """[RULES]
1) SOV word order (Subject–Object–Verb).
2) The suffix -ka attaches only to A-class nouns.
"""

EXAMPLES = """[EXAMPLE]
mika-ka ... ✓
miti-ka ... ✗
"""

def build_prompt(
    ex: dict,
    condition: str = "implicit",
    for_eval: bool = False,
    task_type: str = "generation",  # 추가됨: downstream task type
) -> str:

    if for_eval:
        return ex.get("text", "") or ex.get("sentence", "")

    # 기본 문장
    sent = ex.get("text") or ex.get("text_pos") or ex.get("sentence") or ex.get("text_neg", "")

    # 명시적 조건 처리
    condition = (condition or "").lower()
    rule_section = f"{EXPLICIT_RULE_CARD}\n{EXAMPLES}" if condition == "explicit" else f"{EXAMPLES}"

    if task_type == "generation":
        prompt = f"{rule_section}[INPUT]\n{sent}"


    elif task_type == "grammaticality":
        prompt = (
            f"{rule_section}\n"
            f"[TASK]\nJudge whether the following sentence is grammatically correct "
            f"according to the given rules.\n"
            f"[SENTENCE]\n{sent}\n"
            f"Answer with 'Yes' if it is grammatical, or 'No' if it violates the rule."
        )

    elif task_type == "comparison":
        s1, s2 = ex.get("sentence_1"), ex.get("sentence_2")
        prompt = (
            f"{rule_section}\n"
            f"[TASK]\nWhich of the following sentences is grammatically correct?\n"
            f"1. {s1}\n2. {s2}\n"
            f"Answer with '1' or '2'."
        )

    elif task_type == "mcq":
        options = ex.get("options", [])
        question = ex.get("question", "Choose the grammatically correct sentence:")
        opts = "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))
        prompt = f"{rule_section}\n[TASK]\n{question}\n{opts}\nAnswer with '1', '2', '3', '4', or '5'."

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    return prompt
