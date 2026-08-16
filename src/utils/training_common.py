"""
training_common.py

Shared utilities used by both train_lora.py and train_full_ft.py:
    - set_seed
    - loading processed jsonl data
    - tokenization (+ label alignment for token classification)
    - label list handling
    - metric functions (macro F1 for classification, seqeval entity-F1 for NER)

Kept here so train_lora.py / train_full_ft.py only differ in the part
that actually matters: whether the model is wrapped with a LoRA adapter.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
MAX_LENGTH = 256  # confirmed from src/check_sequence_lengths.py (covers ~p99 of both tasks)

WISESIGHT_LABELS = ["pos", "neu", "neg", "q"]  # confirmed order from HF ClassLabel


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_jsonl(path: Path) -> list:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def load_splits(task: str):
    """Returns (train, val, test) as lists of dicts, loaded from data/processed/{task}/."""
    task_dir = DATA_DIR / task
    train = load_jsonl(task_dir / "train.jsonl")
    val = load_jsonl(task_dir / "val.jsonl")
    test = load_jsonl(task_dir / "test.jsonl")
    return train, val, test


def get_ner_label_list(task: str) -> list:
    """
    Builds a deterministic tag2id mapping by scanning all splits for
    unique NER tags actually present in the data. Sorted alphabetically
    for determinism (order doesn't need to match the original HF
    ClassLabel order — we only need a consistent int<->string mapping
    within this project, and data_prep.py already converted tags to
    strings, not ints).
    """
    train, val, test = load_splits(task)
    tag_set = set()
    for ex in train + val + test:
        tag_set.update(ex["ner_tags"])
    return sorted(tag_set)


# -----------------------------------------------------------------------------
# Tokenization — classification (wisesight_sentiment)
# -----------------------------------------------------------------------------

def tokenize_classification_batch(examples: dict, tokenizer, label2id: dict):
    encodings = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    encodings["labels"] = [label2id[label] for label in examples["label"]]
    return encodings


# -----------------------------------------------------------------------------
# Tokenization — token classification (thai_ner)
# -----------------------------------------------------------------------------

def tokenize_and_align_labels_batch(examples: dict, tokenizer, tag2id: dict):
    """
    Standard HuggingFace token-classification alignment:
    - tokenizer is called with is_split_into_words=True (input is
      already a list of reference words, one BIO tag each)
    - the first subword of each word gets the real tag id
    - subsequent subwords of the same word, and special tokens
      ([CLS]/[SEP]/pad), get -100 so the loss ignores them
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        is_split_into_words=True,
    )

    all_labels = []
    for batch_idx, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=batch_idx)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(tag2id[tags[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_metrics_classification(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def make_compute_metrics_ner(id2tag: dict):
    """
    Returns a compute_metrics function closed over id2tag, since
    Trainer's compute_metrics signature only receives (logits, labels)
    and needs the tag vocabulary to convert ids back to BIO strings
    for seqeval (which operates on tag strings, not ids).
    """
    from seqeval.metrics import f1_score as seqeval_f1
    from seqeval.metrics import precision_score as seqeval_precision
    from seqeval.metrics import recall_score as seqeval_recall

    def compute_metrics_ner(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_labels = []
        true_preds = []
        for pred_row, label_row in zip(preds, labels):
            row_labels = []
            row_preds = []
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                row_labels.append(id2tag[l])
                row_preds.append(id2tag[p])
            true_labels.append(row_labels)
            true_preds.append(row_preds)

        return {
            "entity_f1": seqeval_f1(true_labels, true_preds),
            "entity_precision": seqeval_precision(true_labels, true_preds),
            "entity_recall": seqeval_recall(true_labels, true_preds),
        }

    return compute_metrics_ner
