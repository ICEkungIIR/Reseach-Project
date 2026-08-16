"""
train_lora.py

Purpose:
    Train a LoRA-adapted model given a config (model, task, rank, seed).
    Reads hyperparameters from configs/*.yaml — rank is the swept
    variable; alpha is fixed at 2*rank per config to isolate rank's
    effect (per Biderman et al. 2024 finding that alpha=2r matters a
    lot at higher ranks).

Planned CLI:
    python src/train_lora.py --config configs/wangchanberta_lora.yaml \
        --task wisesight --rank 8 --seed 0

Not yet implemented — scaffold only.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
