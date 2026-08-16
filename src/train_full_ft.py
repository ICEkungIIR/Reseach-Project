"""
train_full_ft.py

Full fine-tuning baseline — same backbone, task, and fixed
hyperparameters (batch size, learning rate specific to this method) as
the LoRA runs, so the only real methodological difference is
full-FT vs. LoRA. Structurally mirrors train_lora.py minus the LoRA
wrapping step, so the two are easy to diff against each other.

Usage:
    python src/train_full_ft.py --config configs/full_ft.yaml \
        --model wangchanberta --task wisesight_sentiment --seed 0

    python src/train_full_ft.py --config configs/full_ft.yaml \
        --model phayathaibert --task thai_ner --seed 2
"""

import argparse
import json
import os
import time
from pathlib import Path

import yaml
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from utils.training_common import (
    WISESIGHT_LABELS,
    compute_metrics_classification,
    get_ner_label_list,
    load_splits,
    make_compute_metrics_ner,
    set_seed,
    tokenize_and_align_labels_batch,
    tokenize_classification_batch,
)

RUNS_DIR = Path(os.environ.get("RUNS_DIR", str(Path(__file__).resolve().parent.parent / "experiments" / "runs")))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=["wangchanberta", "phayathaibert"])
    parser.add_argument("--task", required=True, choices=["wisesight_sentiment", "thai_ner"])
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def build_datasets_and_labels(task, tokenizer):
    train, val, test = load_splits(task)

    if task == "wisesight_sentiment":
        label2id = {label: i for i, label in enumerate(WISESIGHT_LABELS)}
        id2label = {i: label for label, i in label2id.items()}

        def tok_fn(batch):
            return tokenize_classification_batch(batch, tokenizer, label2id)

        num_labels = len(WISESIGHT_LABELS)
        original_columns = ["text", "label"]

    else:  # thai_ner
        tag_list = get_ner_label_list(task)
        tag2id = {tag: i for i, tag in enumerate(tag_list)}
        id2label = {i: tag for tag, i in tag2id.items()}

        def tok_fn(batch):
            return tokenize_and_align_labels_batch(batch, tokenizer, tag2id)

        num_labels = len(tag_list)
        original_columns = ["tokens", "ner_tags"]

    # See train_lora.py for why remove_columns is required here.
    train_ds = Dataset.from_list(train).map(tok_fn, batched=True, remove_columns=original_columns)
    val_ds = Dataset.from_list(val).map(tok_fn, batched=True, remove_columns=original_columns)
    test_ds = Dataset.from_list(test).map(tok_fn, batched=True, remove_columns=original_columns)

    return train_ds, val_ds, test_ds, num_labels, id2label


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)

    # full_ft.yaml lists both backbones under "models" — pick the one requested
    model_entry = next(m for m in config["models"] if m["tokenizer"] == args.model)
    model_name = model_entry["name"]
    tokenizer_name = model_entry["tokenizer"]
    train_cfg = config["training"]

    print(f"Loading tokenizer/model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds, val_ds, test_ds, num_labels, id2label = build_datasets_and_labels(args.task, tokenizer)

    if args.task == "wisesight_sentiment":
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        compute_metrics = compute_metrics_classification
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        compute_metrics = make_compute_metrics_ner(id2label)

    # No LoRA wrapping here — every parameter is trainable (this is the
    # methodological difference from train_lora.py).

    run_name = f"{tokenizer_name}_{args.task}_fullft_seed{args.seed}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoint"),
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["epochs"],
        weight_decay=train_cfg["weight_decay"],
        fp16=train_cfg["fp16"],
        seed=args.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    val_metrics = trainer.evaluate(val_ds)
    test_metrics = trainer.evaluate(test_ds)

    # Full FT saves the whole model (not just an adapter) — much larger
    # than LoRA checkpoints. Keeping every checkpoint for now per
    # project decision; revisit disk usage once all 12 full-FT runs
    # (2 backbones x 2 tasks x 3 seeds) have been run.
    trainer.save_model(str(run_dir / "checkpoint"))

    metrics_out = {
        "config": {
            "model_name": model_name,
            "task": args.task,
            "method": "full_ft",
            "seed": args.seed,
            "learning_rate": train_cfg["learning_rate"],
            "batch_size": train_cfg["batch_size"],
            "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
            "epochs": train_cfg["epochs"],
        },
        "val": val_metrics,
        "test": test_metrics,
        "training_time_seconds": training_time,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    print(f"Done. Metrics -> {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
