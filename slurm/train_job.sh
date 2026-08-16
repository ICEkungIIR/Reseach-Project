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
