EXPLICIT_RULE_CARD = """[GRAMMAR RULES]
1) Word Order: Subject-Verb-Object (SVO)
   - The subject comes first
   - The verb comes second  
   - The object comes third
   - Adverbs can appear optionally at the end

2) Examples:
   ✓ Correct: "The dog eats the bone."
   ✓ Correct: "They will hunt birds sometimes."
   ✗ Incorrect: "Eats the dog the bone." (VSO order)
   ✗ Incorrect: "The bone the dog eats." (OSV order)
"""
IMPLICIT_EXAMPLES = """[EXAMPLES]
✓ The dog eats the bone.
✓ They will hunt birds sometimes.
✓ Each zebra has a unique pattern.
✗ Eats the dog the bone.
✗ The bone the dog eats.
"""

def build_prompt(ex: dict, condition: str) -> str:
    if "prompt" in ex and ex['prompt']:
        base_prompt = ex["prompt"]
        sent = ex.get("text", "")

        if condition == "explicit":
            return f"{base_prompt}\n\nSentence: {sent}"
        else:
            if "Example:" in base_prompt:
                example_part = base_prompt.split("Example:", 1)[1]
                return f"Example: {example_part}\n\nSentence: {sent}"
            else:
                return f"Sentence: {sent}"

    sent = ex.get("text", "")

    if condition == "explicit":
        return f"{EXPLICIT_RULE_CARD}\n\nSentence: {sent}"
    else:
        return f"{IMPLICIT_EXAMPLES}\n\nSentence: {sent}"
