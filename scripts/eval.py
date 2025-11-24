import argparse, json, yaml, os, sys, time, math
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.ap
pend(PROJECT_ROOT)

from scripts.prompts import build_prompt

def eval_ppl(args):
    device = args.device if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.ckpt).to(device).eval()

    log_dir = f"{args.ckpt}_eval_logs_{args.condition}"
    writer = SummaryWriter(log_dir=log_dir)

    total_loss, total_tokens = 0.0, 0
    start_time = time.time()

    with open(args.test_path, "r", encoding="utf-8") as f:
        for step, line in enumerate(f):
            ex = json.loads(line)
            text = ex.get("text") or ex.get("sentence") or ex.get("input")

            if not text:
                text = build_prompt(ex, condition=args.condition, for_eval=False)

            enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)

            with torch.no_grad():
                out = model(**enc, labels=enc["input_ids"])
                loss = out.loss.item()

            seq_len = enc["input_ids"].shape[1]
            total_loss += loss * seq_len
            total_tokens += seq_len

            elapsed = time.time() - start_time
            writer.add_scalar("eval/loss", loss, step)
            writer.add_scalar("eval/samples_per_sec", (step + 1) / elapsed, step)

    ppl = math.exp(total_loss / max(total_tokens, 1))
    print(f"PPL:{ppl:.6f}")

    result_path = f"{args.ckpt}_eval_{args.condition}_ppl.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"ppl": ppl}, f, indent=2)

    writer.close()


def pll(model, tok, text, device, max_length=512):
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        logits = model(**enc).logits[:, :-1, :]
        labels = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_ll = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        return (token_ll * mask).sum().item() / mask.sum().item()


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

    log_dir = f"{args.ckpt}_eval_logs_{args.condition}"
    writer = SummaryWriter(log_dir=log_dir)

    with open(args.test_path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]

    byid = defaultdict(lambda: {"ok": [], "violation": []})
    for r in rows:
        pid = r.get("pair_id")
        if pid is not None:
            byid[pid]["ok" if r.get("label") == "ok" else "violation"].append(r)

    pairs = []
    for _, v in byid.items():
        n = min(len(v["ok"]), len(v["violation"]))
        for i in range(n):
            pairs.append((v["ok"][i], v["violation"][i]))

    if not pairs:
        bucket = defaultdict(lambda: {"ok": [], "violation": []})
        for r in rows:
            bucket[key_from_meta(r)]["ok" if r.get("label") == "ok" else "violation"].append(r)

        for _, v in bucket.items():
            n = min(len(v["ok"]), len(v["violation"]))
            for i in range(n):
                pairs.append((v["ok"][i], v["violation"][i]))

    if not pairs:
        print("PAIRS:0 ACC:0.0"); return

    correct = 0
    gaps, confs = [], []
    start_time = time.time()

    for step, (ok_ex, vi_ex) in enumerate(pairs):
        ok_txt = build_prompt(ok_ex, condition=args.condition, for_eval=True)
        vi_txt = build_prompt(vi_ex, condition=args.condition, for_eval=True)

        ok_s = pll(model, tok, ok_txt, device, args.max_length)
        vi_s = pll(model, tok, vi_txt, device, args.max_length)

        gap = ok_s - vi_s
        gaps.append(gap)

        m = max(ok_s, vi_s)
        p_ok = math.exp(ok_s - m) / (math.exp(ok_s - m) + math.exp(vi_s - m))
        pred_ok = p_ok >= 0.5

        correct += int(pred_ok)
        confs.append(p_ok if pred_ok else 1 - p_ok)

        elapsed = time.time() - start_time
        writer.add_scalar("eval/gap", gap, step)
        writer.add_scalar("eval/samples_per_sec", (step + 1) / elapsed, step)

    acc = correct / len(pairs)
    mean_gap = sum(gaps) / len(gaps)

    bins = 10
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo <= c < hi) or (b == bins - 1 and c == hi)]
        if not idx:
            continue
        avg_conf = sum(confs[i] for i in idx) / len(idx)
        avg_acc = sum(1 for i in idx if gaps[i] > 0) / len(idx)
        ece += (len(idx) / len(confs)) * abs(avg_conf - avg_acc)

    print(f"PAIRS:{len(pairs)} ACC:{acc:.6f} MEAN_PLL_GAP:{mean_gap:.6f} ECE:{ece:.6f}")

    result = {
        "pairs": len(pairs),
        "accuracy": acc,
        "mean_pll_gap": mean_gap,
        "ece": ece
    }
    out_path = f"{args.ckpt}_eval_{args.condition}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    writer.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--test_path", type=str, required=True)
    p.add_argument("--mode", type=str, default="grammar", choices=["grammar", "ppl"])
    p.add_argument("--condition", type=str, default="implicit",
                   choices=["implicit", "explicit", "explicit_a", "explicit_b"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_length", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "ppl":
        eval_ppl(args)
    else:
        eval_grammar(args)


if __name__ == "__main__":
    main()
