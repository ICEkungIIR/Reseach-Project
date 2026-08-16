"""
tokenizer_analysis.py

Purpose:
    Tokenize the same processed dataset with both the WangchanBERTa
    tokenizer and the PhayaThaiBERT tokenizer, and compute fertility
    (subword tokens per reference word) per word and per example.

    Output is stored alongside the processed data so RQ3's token-level
    error analysis can join on it later without re-tokenizing.

Word boundaries:
    - wisesight_sentiment: raw text, no pre-existing word segmentation.
      We segment with pythainlp.word_tokenize(engine="newmm") first to
      get a reference word boundary, then tokenize each reference word
      with the subword tokenizer under test.
    - thai_ner: already has a `tokens` list (word-level) tied to entity
      tags by position. We use those tokens directly as the reference
      words — re-segmenting would desync the tags.

Checkpoints:
    - WangchanBERTa: airesearch/wangchanberta-base-att-spm-uncased
    - PhayaThaiBERT: clicknext/phayathaibert
      (PhayaThaiBERT extends WangchanBERTa's SentencePiece unigram
      tokenizer with expanded loanword vocabulary, so both tokenizers
      are directly comparable rather than unrelated vocabularies.)

Usage:
    python src/tokenizer_analysis.py --task wisesight_sentiment
    python src/tokenizer_analysis.py --task thai_ner
    python src/tokenizer_analysis.py --task all
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

HIGH_FERTILITY_THRESHOLD = 3  # subwords/word flagged as "high fertility"

TOKENIZER_CHECKPOINTS = {
    "wangchanberta": "airesearch/wangchanberta-base-att-spm-uncased",
    "phayathaibert": "clicknext/phayathaibert",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DATA_CARD_PATH = DATA_DIR / "DATA_CARD.md"
RESULTS_TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"

SPLITS = ["train", "val", "test"]
TASKS = ["wisesight_sentiment", "thai_ner"]

_tokenizer_cache = {}


def get_tokenizer(tokenizer_name: str):
    if tokenizer_name not in _tokenizer_cache:
        checkpoint = TOKENIZER_CHECKPOINTS[tokenizer_name]
        print(f"Loading tokenizer '{tokenizer_name}' ({checkpoint})...")
        _tokenizer_cache[tokenizer_name] = AutoTokenizer.from_pretrained(checkpoint)
    return _tokenizer_cache[tokenizer_name]


# -----------------------------------------------------------------------------
# Fertility computation
# -----------------------------------------------------------------------------

def compute_fertility(word: str, tokenizer) -> int:
    """
    Number of subword tokens the tokenizer produces for a single
    reference word. Uses tokenizer.tokenize() (not encode()) to avoid
    counting special tokens like [CLS]/[SEP]/<s>/</s>.
    """
    if word.strip() == "":
        return 0
    subwords = tokenizer.tokenize(word)
    return max(len(subwords), 1)  # guard against tokenizers that drop pure-whitespace/control chars


def _get_reference_words(example: dict, task: str) -> list:
    if task == "wisesight_sentiment":
        from pythainlp import word_tokenize
        return [w for w in word_tokenize(example["text"], engine="newmm") if w.strip() != ""]
    elif task == "thai_ner":
        return [w for w in example["tokens"] if w.strip() != ""]
    else:
        raise ValueError(f"Unknown task '{task}'")


# -----------------------------------------------------------------------------
# Per-split analysis
# -----------------------------------------------------------------------------

def analyze_dataset(task: str, split: str, tokenizer_name: str):
    """
    Runs every example in data/processed/{task}/{split}.jsonl through
    compute_fertility() for the given tokenizer, and writes a per-word
    breakdown to:
        data/processed/{task}/{split}_fertility_{tokenizer_name}.jsonl

    Each output row:
        {
          "example_id": int,          # line index in the processed split file
          "words": [...],             # reference words
          "subword_counts": [...],    # subwords per word, same length as "words"
          "sentence_fertility": float # mean subwords/word for this example
        }
    """
    tokenizer = get_tokenizer(tokenizer_name)
    in_path = PROCESSED_DIR / task / f"{split}.jsonl"
    out_path = PROCESSED_DIR / task / f"{split}_fertility_{tokenizer_name}.jsonl"

    n_examples = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for example_id, line in enumerate(fin):
            example = json.loads(line)
            words = _get_reference_words(example, task)
            if not words:
                continue
            subword_counts = [compute_fertility(w, tokenizer) for w in words]
            sentence_fertility = sum(subword_counts) / len(words)
            fout.write(json.dumps({
                "example_id": example_id,
                "words": words,
                "subword_counts": subword_counts,
                "sentence_fertility": sentence_fertility,
            }, ensure_ascii=False) + "\n")
            n_examples += 1

    print(f"[{task}/{split}/{tokenizer_name}] analyzed {n_examples} examples -> {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# Comparison summary
# -----------------------------------------------------------------------------

def compare_tokenizers(task: str):
    """
    Reads the fertility breakdowns for both tokenizers (all splits,
    combined) and writes a summary table:
        results/tables/{task}_tokenizer_fertility_summary.csv
    Also appends a short summary to data/DATA_CARD.md.
    """
    import csv

    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for tokenizer_name in TOKENIZER_CHECKPOINTS:
        all_word_fertilities = []
        for split in SPLITS:
            path = PROCESSED_DIR / task / f"{split}_fertility_{tokenizer_name}.jsonl"
            if not path.exists():
                print(f"WARNING: missing {path}, skipping in summary. Run analyze_dataset first.")
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    all_word_fertilities.extend(row["subword_counts"])

        if not all_word_fertilities:
            continue

        n_words = len(all_word_fertilities)
        mean_fert = sum(all_word_fertilities) / n_words
        sorted_fert = sorted(all_word_fertilities)
        median_fert = sorted_fert[n_words // 2]
        n_high = sum(1 for c in all_word_fertilities if c > HIGH_FERTILITY_THRESHOLD)
        pct_high = n_high / n_words

        summary_rows.append({
            "task": task,
            "tokenizer": tokenizer_name,
            "n_words": n_words,
            "mean_fertility": round(mean_fert, 4),
            "median_fertility": median_fert,
            f"pct_words_over_{HIGH_FERTILITY_THRESHOLD}_subwords": round(pct_high, 4),
        })

    if not summary_rows:
        print(f"No fertility data found for task '{task}' — run analyze_dataset for both tokenizers first.")
        return

    out_csv = RESULTS_TABLES_DIR / f"{task}_tokenizer_fertility_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote comparison summary -> {out_csv}")

    with open(DATA_CARD_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n## Tokenizer fertility summary — {task}\n\n")
        f.write(f"(threshold for \"high fertility\" = >{HIGH_FERTILITY_THRESHOLD} subwords/word)\n\n")
        header = list(summary_rows[0].keys())
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for row in summary_rows:
            f.write("| " + " | ".join(str(row[h]) for h in header) + " |\n")

    print(f"Appended summary to {DATA_CARD_PATH}")
    return summary_rows


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def run_task(task: str):
    for tokenizer_name in TOKENIZER_CHECKPOINTS:
        for split in SPLITS:
            analyze_dataset(task, split, tokenizer_name)
    compare_tokenizers(task)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS + ["all"])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.task == "all":
        for task in TASKS:
            run_task(task)
    else:
        run_task(args.task)


if __name__ == "__main__":
    main()
