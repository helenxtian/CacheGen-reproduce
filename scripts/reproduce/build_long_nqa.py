"""
Build longer-context NarrativeQA test sets for the context-length experiment.

The artifact's test_data/nqa.jsonl holds LongBench's NarrativeQA questions, with
each story cut to about 9K tokens: the start and the end of the story are kept
and the middle is removed. This script rebuilds the same prompts (same chat
template, same questions, same answers) at several target lengths, cutting the
middle of the story in the same way.

Every output file uses the SAME questions: the first --num stories long enough
for the largest target length. So differences between files come from context
length alone.

Output: test_data/nqa_<N>k.jsonl, one per target length. Each row has
prompt, answers, all_classes, length (story words), longbench_index and
prompt_tokens.

Usage (repository root):
  python scripts/reproduce/build_long_nqa.py --lengths 10000 20000 30000 --num 30
  # optional: --longbench path/to/narrativeqa.jsonl (otherwise downloaded from HF)
"""
import argparse
import json
import os
import zipfile

from transformers import AutoTokenizer

# Exactly the template used in test_data/nqa.jsonl (verified on all 200 rows)
PREFIX = ("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, "
          "detailed, and polite answers to the user's questions. USER: You are given a story, which can be either a "
          "novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase "
          "if possible. Do not provide any explanation.\n\nStory: ")
SUFFIX = ("\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. "
          "Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer: ASSISTANT:")


def load_longbench(path):
    if path is None:
        from huggingface_hub import hf_hub_download
        z = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
        with zipfile.ZipFile(z) as zf:
            return [json.loads(l) for l in zf.read("data/narrativeqa.jsonl").decode().splitlines()]
    return [json.loads(l) for l in open(path)]


def n_tokens(tok, text):
    # Same count as main.py: tokenizer(prompt).input_ids, including BOS
    return len(tok(text).input_ids)


def build_prompt(tok, row, target):
    """Middle-truncate the story so the whole prompt is at most `target` tokens."""
    suffix = SUFFIX.format(input=row["input"])
    ctx = tok(row["context"], add_special_tokens=False).input_ids
    budget = target - n_tokens(tok, PREFIX + suffix)
    for _ in range(10):  # re-tokenizing the joined text can shift the count slightly
        if budget >= len(ctx):
            story = row["context"]
        else:
            half = budget // 2
            story = (tok.decode(ctx[:half], skip_special_tokens=True)
                     + tok.decode(ctx[len(ctx) - (budget - half):], skip_special_tokens=True))
        prompt = PREFIX + story + suffix
        total = n_tokens(tok, prompt)
        if total <= target:
            return prompt, total
        budget -= total - target
    raise RuntimeError(f"could not fit row into {target} tokens")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", type=int, nargs="+", default=[10000, 20000, 30000])
    p.add_argument("--num", type=int, default=30, help="questions per file")
    p.add_argument("--model_id", default="mistral-community/Mistral-7B-v0.2")
    p.add_argument("--longbench", default=None, help="path to LongBench narrativeqa.jsonl")
    p.add_argument("--out_dir", default="test_data")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id)
    rows = load_longbench(args.longbench)
    longest = max(args.lengths)
    chosen = []
    for i, r in enumerate(rows):
        if n_tokens(tok, PREFIX + r["context"] + SUFFIX.format(input=r["input"])) > longest:
            chosen.append(i)
        if len(chosen) == args.num:
            break
    if len(chosen) < args.num:
        print(f"warning: only {len(chosen)} stories are longer than {longest} tokens")
    print(f"using LongBench rows {chosen}")

    for target in args.lengths:
        name = f"nqa_{target // 1000}k"
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        totals = []
        with open(path, "w") as f:
            for i in chosen:
                r = rows[i]
                prompt, total = build_prompt(tok, r, target)
                totals.append(total)
                f.write(json.dumps({"prompt": prompt, "answers": r["answers"], "all_classes": r["all_classes"],
                                    "length": r["length"], "longbench_index": i, "prompt_tokens": total}) + "\n")
        print(f"{path}: {len(totals)} rows, prompt tokens min {min(totals)} max {max(totals)}")


if __name__ == "__main__":
    main()
