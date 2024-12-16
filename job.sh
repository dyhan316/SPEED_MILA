#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32GB
#SBATCH --time=10:00:00
#SBATCH --cpus-per-gpu=1
#SBATCH --output=sbatch_out/data_physionet.%A.%a.out
#SBATCH --error=sbatch_err/data_physionet.%A.%a.err
#SBATCH --job-name=data_physionet_preprocess

. /etc/profile
module load anaconda/3
conda activate speed

python scripts/preprocess.py --config configs/moabb_physionetMI_TEST.yaml