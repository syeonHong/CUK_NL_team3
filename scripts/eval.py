import argparse, json, yaml, os, sys
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from collections import defaultdict
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.prompts import build_prompt
from util.metrics import compute_ppl_metrics
def pll(model, tok, text, device, max_length=512):
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        logits = model(**enc).logits[:, :-1, :]
        labels = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_ll = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        return (token_ll * mask).sum().item() / mask.sum().item()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--test_path", type=str, required=True)
    p.add_argument("--mode", type=str, default="grammar", choices=["grammar","ppl"])
    p.add_argument("--condition", type=str, default="implicit", choices=["explicit","implicit"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_length", type=int, default=512)
    return p.parse_args()

def eval_ppl(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.ckpt).to(device).eval()
    total_loss, total_tokens = 0.0, 0
    with open(args.test_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            text = ex.get("input")
            if not text:
                text = build_prompt(ex, condition=args.condition, for_eval=False)
            enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
            with torch.no_grad():
                out = model(**enc, labels=enc["input_ids"])
                loss = out.loss.item()
            seq_len = enc["input_ids"].shape[1]
            total_loss += loss * seq_len
            total_tokens += seq_len
    ppl = math.exp(total_loss / max(total_tokens, 1))
    print(f"PPL:{ppl:.6f}")

def key_from_meta(r):
    m = r.get("meta", {})
    s, o = m.get("s", {}), m.get("o", {})
    return (s.get("base"), o.get("base"), s.get("plural_type"), o.get("plural_type"), m.get("length"))

def eval_grammar(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.ckpt).to(device).eval()

    with open(args.test_path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]

    pairs = []
    byid = defaultdict(lambda: {"ok": [], "violation": []})
    for r in rows:
        pid = r.get("pair_id")
        if pid is not None:
            byid[pid]["ok" if r.get("label")=="ok" else "violation"].append(r)
    for _, v in byid.items():
        n = min(len(v["ok"]), len(v["violation"]))
        for i in range(n):
            pairs.append((v["ok"][i], v["violation"][i]))

    if not pairs:
        bucket = defaultdict(lambda: {"ok": [], "violation": []})
        for r in rows:
            bucket[key_from_meta(r)]["ok" if r.get("label")=="ok" else "violation"].append(r)
        for _, v in bucket.items():
            n = min(len(v["ok"]), len(v["violation"]))
            for i in range(n):
                pairs.append((v["ok"][i], v["violation"][i]))

    if not pairs:
        for r in rows:
            tp, tn = r.get("text_pos"), r.get("text_neg")
            if tp and tn:
                ok_ex = dict(r); ok_ex["text"] = tp; ok_ex["label"] = "ok"
                vi_ex = dict(r); vi_ex["text"] = tn; vi_ex["label"] = "violation"
                pairs.append((ok_ex, vi_ex))

    if not pairs:
        print("PAIRS:0 ACC:0.000000")
        return

    correct, gaps, confs, y_true = 0, [], [], []
    for ok_ex, vi_ex in pairs:
        ok_txt = build_prompt(ok_ex, condition=args.condition, for_eval=True)
        vi_txt = build_prompt(vi_ex, condition=args.condition, for_eval=True)
        ok_s = pll(model, tok, ok_txt, device, args.max_length)  # 평균 log P
        vi_s = pll(model, tok, vi_txt, device, args.max_length)

        gap = ok_s - vi_s
        gaps.append(gap)

        m = max(ok_s, vi_s)
        p_ok = math.exp(ok_s - m) / (math.exp(ok_s - m) + math.exp(vi_s - m))
        pred_ok = p_ok >= 0.5
        is_correct = bool(pred_ok)
        correct += int(is_correct)
        confs.append(p_ok if pred_ok else 1.0 - p_ok)
        y_true.append(1)

    acc = correct / len(pairs)

    # ECE (10-bin)
    bins = 10
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if lo <= c < hi or (b == bins - 1 and c == hi)]
        if not idx:
            continue
        avg_conf = sum(confs[i] for i in idx) / len(idx)
        avg_acc  = sum(1 for i in idx if (gaps[i] > 0)) / len(idx)
        ece += (len(idx) / len(confs)) * abs(avg_conf - avg_acc)

    mean_gap = sum(gaps) / len(gaps)
    print(f"PAIRS:{len(pairs)} ACC:{acc:.6f} MEAN_PLL_GAP:{mean_gap:.6f} ECE:{ece:.6f}")

def main():
    args = parse_args()
    if args.mode == "ppl":
        eval_ppl(args)
    else:
        eval_grammar(args)

if __name__ == "__main__":
    main()
