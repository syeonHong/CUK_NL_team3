import json
import os
import subprocess
import sys

def run_eval(ckpt, condition, test_path):
    cmd = [
        sys.executable,
        os.path.join("scripts", "eval.py"),
        "--ckpt", ckpt,
        "--test_path", test_path,
        "--mode", "grammar",
        "--condition", condition,
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Eval failed for {ckpt} ({condition})")

    # eval.py에서 저장한 결과 JSON 경로 (eval.py와 동일한 규칙 사용)
    result_path = f"{ckpt}_eval_{condition}.json"
    if not os.path.exists(result_path):
        raise FileNotFoundError(result_path)

    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    return {
        "avg_ppl_ok": result.get("avg_ppl_ok"),
        "avg_ppl_violation": result.get("avg_ppl_violation"),
    }

def main():
    test_path = r".\data\test_arla.jsonl"

    ckpt_a = r".\model\explicit_a_gpt2"
    ckpt_b = r".\model\explicit_b_gpt2"

    # explicit_a
    res_a = run_eval(ckpt=ckpt_a, condition="explicit_a", test_path=test_path)

    # explicit_b
    res_b = run_eval(ckpt=ckpt_b, condition="explicit_b", test_path=test_path)

    print("\n=== 평균 Perplexity 비교 (grammar eval 기준) ===")
    print(f"explicit_a_gpt2  - AVERAGE PPL (OK):        {res_a['avg_ppl_ok']}")
    print(f"explicit_a_gpt2  - AVERAGE PPL (violation): {res_a['avg_ppl_violation']}")
    print(f"explicit_b_gpt2  - AVERAGE PPL (OK):        {res_b['avg_ppl_ok']}")
    print(f"explicit_b_gpt2  - AVERAGE PPL (violation): {res_b['avg_ppl_violation']}")

    # 간단 비교 예시 (OK 문장 기준)
    if res_a["avg_ppl_ok"] is not None and res_b["avg_ppl_ok"] is not None:
        better = "explicit_a_gpt2" if res_a["avg_ppl_ok"] < res_b["avg_ppl_ok"] else "explicit_b_gpt2"
        print(f"\nOK 문장 기준으로 perplexity가 더 낮은 모델: {better}")

if __name__ == "__main__":
    main()
