"""
check_sequence_lengths.py

One-off diagnostic (not part of the main pipeline) — tokenizes every
example in each processed dataset with both backbones' tokenizers and
reports the sequence length distribution, so max_length in the
training configs can be chosen from real data instead of guessed.

Usage:
    python src/check_sequence_lengths.py
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

TOKENIZER_CHECKPOINTS = {
    "wangchanberta": "airesearch/wangchanberta-base-att-spm-uncased",
    "phayathaibert": "clicknext/phayathaibert",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TASKS = ["wisesight_sentiment", "thai_ner"]
SPLITS = ["train", "val", "test"]


def get_text(example: dict, task: str) -> str:
    if task == "wisesight_sentiment":
        return example["text"]
    elif task == "thai_ner":
        # Join tokens with the same separator the model will see at train
        # time — for token classification, tokenizer is called with
        # is_split_into_words=True, so join with a single space here
        # just to measure length (actual training calls tokenizer on
        # the token list directly, not this joined string).
        return " ".join(example["tokens"])
    raise ValueError(task)


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0
    idx = int(len(sorted_vals) * pct) 
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def main():
    for task in TASKS:
        for tokenizer_name, checkpoint in TOKENIZER_CHECKPOINTS.items():
            print(f"\nLoading tokenizer '{tokenizer_name}'...")
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)

            lengths = []
            for split in SPLITS:
                path = DATA_DIR / task / f"{split}.jsonl"
                if not path.exists():
                    print(f"  (skip missing {path})")
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        example = json.loads(line)
                        text = get_text(example, task)
                        # include special tokens since that's what the model actually sees
                        ids = tokenizer(text, truncation=False)["input_ids"]
                        lengths.append(len(ids))

            if not lengths:
                continue

            lengths.sort()
            print(f"[{task} / {tokenizer_name}] n={len(lengths)}")
            print(f"  min={lengths[0]}  p50={percentile(lengths, 0.50)}  "
                  f"p90={percentile(lengths, 0.90)}  p95={percentile(lengths, 0.95)}  "
                  f"p99={percentile(lengths, 0.99)}  max={lengths[-1]}")


if __name__ == "__main__":
    main()
