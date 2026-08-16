"""
data_prep.py

Purpose:
    Load raw Thai datasets (Wisesight Sentiment, ThaiNER), clean them,
    and produce fixed train/val/test splits (single fixed seed=42,
    reused across every experiment config so split variance doesn't
    mix with rank variance).

Datasets:
    - Text classification: Wisesight Sentiment ("pythainlp/wisesight_sentiment")
        Confirmed HF schema: columns are "texts" (string) and
        "category" (ClassLabel: pos, neu, neg, q).
    - NER: ThaiNER ("pythainlp/thainer-corpus-v2")

Usage:
    python src/data_prep.py --task wisesight_sentiment
    python src/data_prep.py --task thai_ner
    python src/data_prep.py --task all
"""

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATA_CARD_PATH = DATA_DIR / "DATA_CARD.md"

HF_DATASET_IDS = {
    "wisesight_sentiment": "pythainlp/wisesight_sentiment",
    "thai_ner": "pythainlp/thainer-corpus-v2",
}


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_raw(task: str):
    """
    Load a dataset from HuggingFace Hub and return it as a list of dicts
    with NORMALIZED field names (so downstream code doesn't need to
    guess/fallback across schema variants):

        wisesight_sentiment -> {"text": str, "label": str}
            "label" is the human-readable class name (pos/neu/neg/q),
            converted from the dataset's ClassLabel int encoding.
        thai_ner -> {"tokens": List[str], "ner_tags": List[str]}

    Also caches the normalized pull to data/raw/{task}.jsonl so we're
    not solely dependent on the HF cache (dataset versions on the hub
    can change later).
    """
    if task not in HF_DATASET_IDS:
        raise ValueError(f"Unknown task '{task}'. Expected one of {list(HF_DATASET_IDS)}")

    hf_id = HF_DATASET_IDS[task]
    print(f"Loading '{hf_id}' from HuggingFace Hub...")
    ds = load_dataset(hf_id)

    all_examples = []

    if task == "wisesight_sentiment":
        # Confirmed schema: columns "texts" (string), "category" (ClassLabel)
        for split_name in ds.keys():
            split = ds[split_name]
            label_feature = split.features["category"]  # ClassLabel
            for row in split:
                all_examples.append({
                    "text": row["texts"],
                    "label": label_feature.int2str(row["category"]),
                })

    elif task == "thai_ner":
        # Confirmed schema: columns "words" (List[string]),
        # "ner" (List[ClassLabel] — e.g. B-PERSON, I-PERSON, O, ...).
        # Note "O" is NOT index 0 in this tagset, so tags must be
        # converted via int2str rather than assumed.
        for split_name in ds.keys():
            split = ds[split_name]
            tag_feature = split.features["ner"].feature  # ClassLabel inside the Sequence
            for row in split:
                tokens = row["words"]
                tags = [tag_feature.int2str(t) for t in row["ner"]]
                all_examples.append({"tokens": tokens, "ner_tags": tags})

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_cache_path = RAW_DIR / f"{task}.jsonl"
    with open(raw_cache_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Cached {len(all_examples)} raw examples -> {raw_cache_path}")

    return all_examples


# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{2,}")


def clean_text(examples: list, task: str) -> list:
    """
    Light cleaning only — collapse repeated whitespace, drop empty/duplicate
    examples. Does NOT alter token content or ordering, since ThaiNER
    labels are tied to exact token positions: any edit that re-tokenizes
    or drops characters would silently corrupt entity spans.
    """
    cleaned = []
    seen = set()

    for ex in examples:
        if task == "wisesight_sentiment":
            text = _WHITESPACE_RE.sub(" ", ex["text"]).strip()
            text = _MULTI_NEWLINE_RE.sub("\n", text)
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            cleaned.append({"text": text, "label": ex["label"]})

        elif task == "thai_ner":
            tokens, tags = ex["tokens"], ex["ner_tags"]
            if not tokens or not tags or len(tokens) != len(tags):
                continue
            if all(t.strip() == "" for t in tokens):
                continue
            key = tuple(tokens)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append({"tokens": tokens, "ner_tags": tags})

        else:
            raise ValueError(f"Unknown task '{task}'")

    print(f"Cleaned: {len(examples)} -> {len(cleaned)} examples (dupes/empties dropped)")
    return cleaned


# -----------------------------------------------------------------------------
# Splitting
# -----------------------------------------------------------------------------

def split_dataset(examples: list, task: str, seed: int = SEED):
    """
    Stratified 80/10/10 train/val/test split.

    Stratifies on `label` for wisesight_sentiment (classes are imbalanced).
    For thai_ner, stratifies on a coarse proxy — whether the example
    contains at least one non-"O" tag — since per-example multi-label
    stratification on full tag sequences isn't well-defined with
    sklearn's stratify param.
    """
    if task == "wisesight_sentiment":
        labels = [ex["label"] for ex in examples]
    elif task == "thai_ner":
        labels = [
            "has_entity" if any(t != "O" for t in ex["ner_tags"]) else "no_entity"
            for ex in examples
        ]
    else:
        raise ValueError(f"Unknown task '{task}'")

    train, temp, train_labels, temp_labels = train_test_split(
        examples,
        labels,
        train_size=SPLIT_RATIOS["train"],
        random_state=seed,
        stratify=labels,
    )

    # temp is 20% of data; split it 50/50 -> 10%/10% of the original
    val, test = train_test_split(
        temp,
        train_size=0.5,
        random_state=seed,
        stratify=temp_labels,
    )

    print(f"Split ({task}): train={len(train)} val={len(val)} test={len(test)} (seed={seed})")
    return train, val, test


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------

def save_processed(train: list, val: list, test: list, task: str):
    out_dir = PROCESSED_DIR / task
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in split_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(split_data)} examples -> {out_path}")

    return out_dir


def update_data_card(task: str, train: list, val: list, test: list):
    """
    Append real stats for this task to data/DATA_CARD.md.
    """
    stats_lines = [f"\n## Stats — {task} (generated by data_prep.py, seed={SEED})\n"]
    stats_lines.append(f"- Total (train+val+test): {len(train) + len(val) + len(test)}")
    stats_lines.append(f"- Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    if task == "wisesight_sentiment":
        from collections import Counter
        label_counts = Counter(ex["label"] for ex in train + val + test)
        stats_lines.append(f"- Label distribution (all splits): {dict(label_counts)}")
    elif task == "thai_ner":
        entity_examples = sum(
            1 for ex in train + val + test if any(t != "O" for t in ex["ner_tags"])
        )
        stats_lines.append(
            f"- Examples containing at least one entity: {entity_examples} "
            f"({entity_examples / (len(train)+len(val)+len(test)):.1%})"
        )

    with open(DATA_CARD_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(stats_lines) + "\n")

    print(f"Updated {DATA_CARD_PATH} with stats for '{task}'")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def run_task(task: str):
    examples = load_raw(task)
    cleaned = clean_text(examples, task)
    train, val, test = split_dataset(cleaned, task, seed=SEED)
    save_processed(train, val, test, task)
    update_data_card(task, train, val, test)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        choices=["wisesight_sentiment", "thai_ner", "all"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.task == "all":
        for task in HF_DATASET_IDS:
            run_task(task)
    else:
        run_task(args.task)


if __name__ == "__main__":
    main()
