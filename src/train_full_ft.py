"""
train_full_ft.py

Purpose:
    Full fine-tuning baseline (no LoRA) — same backbone, task, and
    fixed hyperparameters (batch size, learning rate) as the LoRA runs,
    so the only real difference is full-FT vs. LoRA.

Planned CLI:
    python src/train_full_ft.py --config configs/full_ft.yaml \
        --task wisesight --seed 0

Not yet implemented — scaffold only.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
