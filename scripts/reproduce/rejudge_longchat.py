"""
Re-score LongChat answers with a GPT judge, as the paper's artifact does, without
re-running any model.

run_cachegen.py, run_quantization_baseline.py and run_vanilla.py print one line
per document:
    Label: <topic>, Predict: <model answer> - string-match goes with correct|wrong
This script reads those lines from a driver log (e.g. reproduce-results/logs/full.log),
asks a GPT model the artifact's exact judging question (src/utils.py, chatgpt_auto_eval),
and reports accuracy per method under both judges, with their agreement.

Each distinct (label, answer) pair is judged once. Verdicts are cached in
<out_dir>/gpt_cache.jsonl, so re-runs are free and resumable.

Needs OPENAI_API_KEY. OPENAI_MODEL defaults to gpt-4o-mini; the artifact used
gpt-3.5-turbo. Note that the labels and model answers are sent to OpenAI.

Usage:
  python scripts/reproduce/rejudge_longchat.py --log reproduce-results/logs/full.log \
      --out_dir reproduce-results/gpt_judge [--dry_run]
"""
import argparse
import hashlib
import json
import os
import re
import time
import urllib.request

# Copied verbatim from src/utils.py (chatgpt_auto_eval), typos included
JUDGE_PROMPT = ("I am testing whether a LLM model can correctly retreieve the first topic, and would like you to help "
                "me judge whether the mode ls correct. Please give me 1 for correct and 0 for incorrect. Only give me a "
                "single number. Ignore mistakes if the model is paraphasing or using synonyms. Ignore any simple "
                "mistakes such as capitalization and punctuation. The ground truth is {gt}, the model prediction is {pred}")

SECTION = re.compile(r"^=== \[\d/\d\] (.*) ===$", re.M)
ENTRY = re.compile(r"Label: (.*?), Predict: (.*?) - string-match goes with (correct|wrong)", re.S)


def section_key(title):
    """Map a driver-script section title to (method, setting), or None to skip it."""
    m = re.match(r"CacheGen, encoding level (-?\d+)", title)
    if m:
        return "cachegen", f"level{m.group(1)}"
    m = re.match(r"CacheGen \+ delta encoding \(ours\), level (-?\d+), (\w+) anchors", title)
    if m:
        return "cachegen_delta", f"level{m.group(1)}" + ("_int8anchor" if m.group(2) == "int8" else "")
    m = re.match(r"Uniform quantization baseline, (\d+)-bit", title)
    if m:
        return "quant", f"{m.group(1)}bit"
    if title.startswith("Vanilla"):
        return "vanilla", "fp16"
    return None


def parse_log(text):
    heads = list(SECTION.finditer(text))
    out = {}
    for h, nxt in zip(heads, heads[1:] + [None]):
        key = section_key(h.group(1))
        if key is None:
            continue
        body = text[h.end(): nxt.start() if nxt else len(text)]
        out[key] = [dict(label=l.strip(), pred=p.strip(), string=(v == "correct")) for l, p, v in ENTRY.findall(body)]
    return out


def ask_gpt(gt, pred, model, api_key, retries=5):
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({"model": model, "temperature": 0, "max_tokens": 5,
                         "messages": [{"role": "user", "content": JUDGE_PROMPT.format(gt=gt, pred=pred)}]}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                reply = json.load(r)["choices"][0]["message"]["content"]
            # Same rule as src/utils.py: correct iff the reply is "1" (tolerating "1." / whitespace)
            return reply.strip().strip(".").strip() == "1", reply
        except Exception as e:
            print(f"  OpenAI error ({e}); retrying in {5 * (attempt + 1)}s")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("OpenAI request kept failing")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="reproduce-results/logs/full.log")
    p.add_argument("--out_dir", default="reproduce-results/gpt_judge")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    p.add_argument("--dry_run", action="store_true", help="parse the log and report counts; no API calls")
    args = p.parse_args()

    runs = parse_log(open(args.log, encoding="utf-8", errors="replace").read())
    pairs = {(e["label"], e["pred"]) for es in runs.values() for e in es}
    print(f"{args.log}: {len(runs)} runs, {sum(map(len, runs.values()))} answers, {len(pairs)} distinct (label, answer) pairs")
    for (m, s), es in runs.items():
        print(f"  {m:<15} {s:<10} n={len(es):<3} string-match accuracy {sum(e['string'] for e in es) / max(len(es), 1):.2f}")
    if args.dry_run:
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY (or use --dry_run).")
    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, "gpt_cache.jsonl")
    h = lambda l, pr: hashlib.sha256(f"{args.model}\0{l}\0{pr}".encode()).hexdigest()
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            c = json.loads(line)
            cache[c["key"]] = c
    todo = [(l, pr) for l, pr in sorted(pairs) if h(l, pr) not in cache]
    print(f"{len(todo)} pairs to judge with {args.model} ({len(pairs) - len(todo)} cached)")
    with open(cache_path, "a") as f:
        for i, (l, pr) in enumerate(todo, 1):
            ok, reply = ask_gpt(l, pr, args.model, api_key)
            c = dict(key=h(l, pr), model=args.model, label=l, pred=pr, correct=ok, reply=reply)
            cache[c["key"]] = c
            f.write(json.dumps(c) + "\n"); f.flush()
            if i % 20 == 0:
                print(f"  judged {i}/{len(todo)}")
            time.sleep(0.2)

    summary, disagreements = [], []
    for (m, s), es in runs.items():
        gpt = [cache[h(e["label"], e["pred"])]["correct"] for e in es]
        strm = [e["string"] for e in es]
        n = len(es)
        summary.append(dict(method=m, setting=s, num_docs=n, model=args.model,
                            string_accuracy=sum(strm) / n, gpt_accuracy=sum(gpt) / n,
                            agreement=sum(a == b for a, b in zip(gpt, strm)) / n))
        disagreements += [dict(method=m, setting=s, doc=i, label=e["label"], pred=e["pred"],
                               string=e["string"], gpt=g) for i, (e, g) in enumerate(zip(es, gpt)) if g != e["string"]]
    json.dump(dict(summary=summary, disagreements=disagreements),
              open(os.path.join(args.out_dir, "gpt_judge_summary.json"), "w"), indent=2)
    lines = [f"| Method | Setting | n | String-match accuracy | GPT accuracy ({args.model}) | Agreement |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r['method']} | {r['setting']} | {r['num_docs']} | {r['string_accuracy']:.2f} | "
              f"{r['gpt_accuracy']:.2f} | {r['agreement']:.0%} |" for r in summary]
    open(os.path.join(args.out_dir, "gpt_judge_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n{len(disagreements)} disagreements; details in {args.out_dir}/gpt_judge_summary.json")


if __name__ == "__main__":
    main()
