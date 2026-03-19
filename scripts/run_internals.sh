#!/bin/bash -l
#SBATCH --job-name=vla-internals
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --gres=min-vram:32
#SBATCH --cpus-per-task=8
#SBATCH --output=outputs/logs/%j_internals.out
#SBATCH --error=outputs/logs/%j_internals.err

module load mamba
conda activate vla

export HF_HOME=/scratch/work/zouz1/hugface

cd /scratch/work/zouz1/VLA_intent/VLA_Intent_Steering

srun python scripts/analyze_internals.py \
    --model_path /scratch/work/zouz1/VLA_intent/models/openvla-7b \
    --output_dir outputs/internals \
    --multi_instruction
