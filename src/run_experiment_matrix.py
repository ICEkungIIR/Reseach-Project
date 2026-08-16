"""
run_experiment_matrix.py

Runs the full experiment matrix:
    2 backbones (wangchanberta, phayathaibert)
    x 2 tasks (wisesight_sentiment, thai_ner)
    x [4 LoRA ranks (4, 8, 16, 32) + 1 full FT]
    x 3 seeds (0, 1, 2)
    = 60 runs total

Designed to be resumable across Colab sessions:
    - Before launching a run, checks whether its metrics.json already
      exists (in RUNS_DIR, which honors the same env var override as
      train_lora.py / train_full_ft.py). If so, skips it.
    - If a run's subprocess fails, the failure is logged to
      failed_runs.log and the matrix continues with the next run —
      one bad run should not block the other 59.

Usage (from repo root):
    python src/run_experiment_matrix.py
    python src/run_experiment_matrix.py --only-task wisesight_sentiment
    python src/run_experiment_matrix.py --only-backbone wangchanberta
    python src/run_experiment_matrix.py --dry-run   # print the plan, run nothing

Set RUNS_DIR env var to point outputs at Google Drive on Colab, e.g.:
    RUNS_DIR=/content/drive/MyDrive/Research_Project_runs python src/run_experiment_matrix.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get("RUNS_DIR", str(REPO_ROOT / "experiments" / "runs")))
FAILED_LOG = REPO_ROOT / "experiments" / "failed_runs.log"

BACKBONES = ["wangchanberta", "phayathaibert"]
TASKS = ["wisesight_sentiment", "thai_ner"]
RANKS = [4, 8, 16, 32]
SEEDS = [0, 1, 2]

CONFIG_FOR_BACKBONE = {
    "wangchanberta": REPO_ROOT / "configs" / "wangchanberta_lora.yaml",
    "phayathaibert": REPO_ROOT / "configs" / "phayathaibert_lora.yaml",
}
FULL_FT_CONFIG = REPO_ROOT / "configs" / "full_ft.yaml"


def build_matrix(only_task=None, only_backbone=None):
    """
    Returns a list of run specs. Each spec is a dict with everything
    needed to (a) predict the run's output dir name (must match the
    naming convention in train_lora.py / train_full_ft.py exactly, or
    the resume-skip check won't find existing results) and (b) build
    the subprocess command.
    """
    runs = []

    for backbone in BACKBONES:
        if only_backbone and backbone != only_backbone:
            continue
        for task in TASKS:
            if only_task and task != only_task:
                continue

            # LoRA runs
            for rank in RANKS:
                for seed in SEEDS:
                    run_name = f"{backbone}_{task}_lora_r{rank}_seed{seed}"
                    runs.append({
                        "run_name": run_name,
                        "method": "lora",
                        "cmd": [
                            sys.executable, "src/train_lora.py",
                            "--config", str(CONFIG_FOR_BACKBONE[backbone]),
                            "--task", task,
                            "--rank", str(rank),
                            "--seed", str(seed),
                        ],
                    })

            # Full FT runs
            for seed in SEEDS:
                run_name = f"{backbone}_{task}_fullft_seed{seed}"
                runs.append({
                    "run_name": run_name,
                    "method": "full_ft",
                    "cmd": [
                        sys.executable, "src/train_full_ft.py",
                        "--config", str(FULL_FT_CONFIG),
                        "--model", backbone,
                        "--task", task,
                        "--seed", str(seed),
                    ],
                })

    return runs


def already_done(run_name: str) -> bool:
    return (RUNS_DIR / run_name / "metrics.json").exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-task", choices=["wisesight_sentiment", "thai_ner"], default=None)
    parser.add_argument("--only-backbone", choices=["wangchanberta", "phayathaibert"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runs = build_matrix(only_task=args.only_task, only_backbone=args.only_backbone)

    print(f"Matrix has {len(runs)} runs total. RUNS_DIR={RUNS_DIR}")

    if args.dry_run:
        for r in runs:
            status = "DONE (skip)" if already_done(r["run_name"]) else "pending"
            print(f"  [{status}] {r['run_name']}")
        return

    REPO_ROOT.joinpath("experiments").mkdir(parents=True, exist_ok=True)

    n_skipped = 0
    n_ran = 0
    n_failed = 0

    for i, r in enumerate(runs, start=1):
        if already_done(r["run_name"]):
            print(f"[{i}/{len(runs)}] SKIP (already done): {r['run_name']}")
            n_skipped += 1
            continue

        print(f"[{i}/{len(runs)}] RUNNING: {r['run_name']}")
        env = os.environ.copy()
        env["RUNS_DIR"] = str(RUNS_DIR)

        # Full output (progress bars, per-step loss) is written to a log
        # file in Drive to avoid flooding the notebook cell. Per-epoch
        # eval results (Trainer prints a dict containing "eval_" once
        # per epoch, since eval_strategy="epoch") are additionally
        # echoed to the notebook so progress is still visible live.
        run_dir = RUNS_DIR / r["run_name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train_log.txt"

        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                r["cmd"], cwd=str(REPO_ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in process.stdout:
                log_file.write(line)
                if "eval_" in line:
                    print(f"    {line.rstrip()}")
            process.wait()
            returncode = process.returncode

        if returncode != 0:
            n_failed += 1
            with open(FAILED_LOG, "a", encoding="utf-8") as f:
                f.write(f"{r['run_name']}\treturncode={returncode}\tcmd={' '.join(r['cmd'])}\n")
            print(f"  FAILED (log: {log_path}) — continuing with next run")
        else:
            n_ran += 1
            print(f"  OK: {r['run_name']}")

    print(f"\nDone. ran={n_ran} skipped={n_skipped} failed={n_failed} (of {len(runs)} total)")
    if n_failed:
        print(f"See {FAILED_LOG} for details on failed runs.")


if __name__ == "__main__":
    main()
