#!/usr/bin/env bash
# =============================================================================
# setup_repo_scaffold.sh
#
# Scaffolds the folder/file structure for:
#   "Effect of LoRA Rank on Fine-tuning Performance for Thai Language
#    Under Tokenization Constraints"
#
# Usage:
#   1. cd into your existing Research_Project git repo (the one already on git)
#   2. bash setup_repo_scaffold.sh
#   3. Review the created files, then `git add . && git commit -m "scaffold repo"`
#
# This script only creates structure + placeholders/docstrings.
# No training/data-processing logic is implemented yet.
# =============================================================================

set -e  # stop on first error

echo "==> Creating directory structure..."

mkdir -p data/raw
mkdir -p data/processed
mkdir -p src/utils
mkdir -p configs
mkdir -p experiments/runs
mkdir -p notebooks
mkdir -p results/tables
mkdir -p results/figures
mkdir -p slurm

# .gitkeep so empty dirs are tracked by git
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch experiments/runs/.gitkeep
touch notebooks/.gitkeep
touch results/tables/.gitkeep
touch results/figures/.gitkeep

# -----------------------------------------------------------------------------
echo "==> Writing .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/

# Data (large / raw — don't push to git)
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# Experiment outputs (checkpoints, logs — large)
experiments/runs/*
!experiments/runs/.gitkeep

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store

# Environment
.env
EOF

# -----------------------------------------------------------------------------
echo "==> Writing requirements.txt..."
cat > requirements.txt << 'EOF'
# Core ML stack
torch>=2.1
transformers>=4.40
peft>=0.10
datasets>=2.18
accelerate>=0.29

# Data / eval
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.11          # for significance testing (paired t-test / bootstrap)

# Tokenizer / Thai-specific
sentencepiece>=0.1.99
pythainlp>=5.0        # useful for Thai text utilities if needed

# Config / experiment management
pyyaml>=6.0

# Plotting
matplotlib>=3.8
seaborn>=0.13

# NOTE: no CUDA version pinned yet — GPU/HPC spec not confirmed.
# Once known, install the matching torch build first, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
# then: pip install -r requirements.txt
EOF

# -----------------------------------------------------------------------------
echo "==> Writing README.md..."
cat > README.md << 'EOF'
# Effect of LoRA Rank on Fine-tuning Performance for Thai Language Under Tokenization Constraints

**Authors:** ศรัณยู เจริญผล, สหรัฐ งามเลิศ

## Research Gap
Prior work shows LoRA and full fine-tuning learn structurally different
solutions (LoRA introduces "intruder dimensions") but this is only tested
in English. Separately, Thai NLP work shows high-fertility tokenizers
(no word boundaries) hurt downstream performance. No work connects these
two threads.

## Research Questions
- **RQ1:** How does the LoRA vs. full fine-tuning F1-score gap on Thai
  text classification / NER compare to the gap reported in English?
- **RQ2:** How does LoRA rank (r = 4, 8, 16, 32) relate to performance
  gap size across tokenizers of different fertility (WangchanBERTa vs.
  PhayaThaiBERT)?
- **RQ3:** Are high-fertility tokens affected more by LoRA than typical
  tokens, measured via token-level error?

## Hypotheses
- **H0:** The Thai LoRA-vs-full-FT performance gap is not significantly
  different from the English gap, regardless of tokenizer.
- **H1:** The Thai gap is significantly larger than the English gap,
  especially with high-fertility tokenizers.

## Repo Structure
```
data/            raw + processed datasets, data card
src/             data prep, tokenizer analysis, training, eval scripts
configs/         YAML configs per model/rank/task combination
experiments/     run outputs (logs, checkpoints, metrics) — gitignored
notebooks/       EDA and error analysis
results/         final tables and figures for the report
slurm/           HPC job submission templates
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate     # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Status
- [x] Research question, PICOC, FINER, SMART, synthesis matrix finalized
- [ ] Repo scaffold (this commit)
- [ ] Data prep + tokenizer fertility analysis
- [ ] LoRA + full FT training runs
- [ ] Token-level error analysis (RQ3)
- [ ] Significance testing + practical guideline write-up
EOF

# -----------------------------------------------------------------------------
echo "==> Writing data/DATA_CARD.md..."
cat > data/DATA_CARD.md << 'EOF'
# Data Card

## Datasets (planned)
| Task | Dataset | Source | License | Notes |
|---|---|---|---|---|
| Text classification | Wisesight Sentiment | TBD (HuggingFace Datasets) | TBD | 4-class sentiment |
| NER | Thai NER (LST20 or ThaiNER) | TBD | TBD | Confirm which variant |

## Fields to fill in once data is downloaded
- Number of examples (train/val/test)
- Label distribution
- Tokenizer fertility stats (WangchanBERTa vs PhayaThaiBERT) — see
  `src/tokenizer_analysis.py`
- Any preprocessing/cleaning applied
EOF

# -----------------------------------------------------------------------------
echo "==> Writing src/ placeholder files..."

touch src/__init__.py
touch src/utils/__init__.py

cat > src/data_prep.py << 'EOF'
"""
data_prep.py

Purpose:
    Load raw Thai datasets (Wisesight Sentiment, Thai NER), clean them,
    and produce fixed train/val/test splits (single fixed seed, reused
    across every experiment config so split variance doesn't mix with
    rank variance).

Planned functions:
    - load_raw(task: str) -> pandas.DataFrame
    - clean_text(df) -> pandas.DataFrame
    - split_dataset(df, seed: int) -> (train, val, test)
    - save_processed(...) -> writes to data/processed/

Not yet implemented — scaffold only.
"""

def load_raw(task: str):
    raise NotImplementedError


def clean_text(df):
    raise NotImplementedError


def split_dataset(df, seed: int = 42):
    raise NotImplementedError


def save_processed(train, val, test, task: str):
    raise NotImplementedError


if __name__ == "__main__":
    pass
EOF

cat > src/tokenizer_analysis.py << 'EOF'
"""
tokenizer_analysis.py

Purpose:
    Tokenize the same processed dataset with both the WangchanBERTa
    tokenizer and the PhayaThaiBERT tokenizer, and compute fertility
    (tokens per word / tokens per character) per example and per token.

    Output is stored alongside the processed data so RQ3's token-level
    error analysis can join on it later without re-tokenizing.

Planned functions:
    - compute_fertility(text: str, tokenizer) -> float
    - analyze_dataset(dataset, tokenizer_name: str) -> pandas.DataFrame
    - compare_tokenizers(dataset) -> pandas.DataFrame  # side-by-side stats

Not yet implemented — scaffold only.
"""

def compute_fertility(text: str, tokenizer):
    raise NotImplementedError


def analyze_dataset(dataset, tokenizer_name: str):
    raise NotImplementedError


def compare_tokenizers(dataset):
    raise NotImplementedError


if __name__ == "__main__":
    pass
EOF

cat > src/train_lora.py << 'EOF'
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
EOF

cat > src/train_full_ft.py << 'EOF'
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
EOF

cat > src/eval.py << 'EOF'
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
EOF

# -----------------------------------------------------------------------------
echo "==> Writing configs/*.yaml..."

cat > configs/wangchanberta_lora.yaml << 'EOF'
# LoRA config — WangchanBERTa backbone
model:
  name: "airesearch/wangchanberta-base-att-spm-uncased"  # TODO confirm exact checkpoint
  tokenizer: "wangchanberta"

lora:
  ranks: [4, 8, 16, 32]     # r — swept variable
  alpha_multiplier: 2       # alpha = 2 * r for every rank (fixed)
  target_modules: ["query", "value"]  # start with attention only
  dropout: 0.1

training:
  learning_rate: null       # TODO fix a single value, same across all runs
  batch_size: null          # TODO fix a single value, same across all runs
  epochs: null
  seeds: [0, 1, 2]          # 3 seeds minimum for statistics

tasks:
  - wisesight_sentiment
  - thai_ner
EOF

cat > configs/phayathaibert_lora.yaml << 'EOF'
# LoRA config — PhayaThaiBERT backbone
model:
  name: "clicknext/phayathaibert"   # TODO confirm exact checkpoint
  tokenizer: "phayathaibert"

lora:
  ranks: [4, 8, 16, 32]
  alpha_multiplier: 2
  target_modules: ["query", "value"]
  dropout: 0.1

training:
  learning_rate: null        # TODO must match wangchanberta_lora.yaml value
  batch_size: null           # TODO must match wangchanberta_lora.yaml value
  epochs: null
  seeds: [0, 1, 2]

tasks:
  - wisesight_sentiment
  - thai_ner
EOF

cat > configs/full_ft.yaml << 'EOF'
# Full fine-tuning baseline config (both backbones)
models:
  - name: "airesearch/wangchanberta-base-att-spm-uncased"
    tokenizer: "wangchanberta"
  - name: "clicknext/phayathaibert"
    tokenizer: "phayathaibert"

training:
  learning_rate: null   # TODO must match the LoRA configs' value
  batch_size: null       # TODO must match the LoRA configs' value
  epochs: null
  seeds: [0, 1, 2]

tasks:
  - wisesight_sentiment
  - thai_ner
EOF

# -----------------------------------------------------------------------------
echo "==> Writing slurm/train_job.sh template..."
cat > slurm/train_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=lora_thai
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
# TODO: fill in partition / account / cpus-per-task once HPC spec is known
# #SBATCH --partition=
# #SBATCH --account=
# #SBATCH --cpus-per-task=

module purge
# module load <cuda/python module — TBD once HPC spec known>

source .venv/bin/activate

# Example usage (edit as needed):
# python src/train_lora.py --config configs/wangchanberta_lora.yaml \
#     --task wisesight_sentiment --rank $1 --seed $2

echo "Job template — fill in the srun/python command above before using."
EOF
mkdir -p slurm/logs
touch slurm/logs/.gitkeep

# -----------------------------------------------------------------------------
echo ""
echo "✅ Scaffold complete."
echo ""
echo "Next steps:"
echo "  1. git add . && git commit -m 'Add repo scaffold'"
echo "  2. Fill in the TODOs in configs/*.yaml (learning_rate, batch_size, exact model checkpoints)"
echo "  3. Start implementing src/data_prep.py"
