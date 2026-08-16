"""
eval.py

Purpose:
    Evaluate a trained checkpoint (LoRA or full FT) and produce:
      - F1-score, accuracy (RQ1, RQ2)
      - performance gap % vs. the matching full-FT run
      - token-level error rate, joined with fertility scores
        from tokenizer_analysis.py (RQ3)
      - significance tests (paired t-test / bootstrap CI) between
        LoRA and full-FT gaps, for testing H0 vs H1

Planned CLI:
    python src/eval.py --run experiments/runs/<run_dir> --task wisesight

Not yet implemented — scaffold only.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--task", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
