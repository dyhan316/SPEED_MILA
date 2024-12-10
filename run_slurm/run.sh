#!/bin/bash
#SBATCH -A m4673  #m4244
#SBATCH --job-name=run_preproc
#SBATCH -C cpu #gpu&hbm80g
#SBATCH -q regular #regular, shared, premium  #! shared : gpu 1,2개만 빨리 쓰고싶을떄 
#SBATCH --output=./shell_outputs/preproc%j.out
#SBATCH --error=./shell_outputs/preproc%j.err
#SBATCH -t 48:00:00 #regular : 48h max
#SBATCH -N 1
#SBATCH --ntasks-per-node=1


#################
####shared queue 쓰면, 밑의것으로 더 적게 사용가능
##SBATCH --cpus-per-task=64
##SBATCH --gpus-per-task=2

#################
module load conda
conda activate DIVER
cd .. 

export HDF5_USE_FILE_LOCKING=FALSE
export NUMEXPR_MAX_THREADS=256
python scripts/preprocess.py --config configs/tuh_PREPROC_FIRST_VER.yaml

echo "----------------------------------Done----------------------------------"
echo "----------------------------------Done----------------------------------"
echo "----------------------------------Done----------------------------------"
echo "----------------------------------Done----------------------------------"
echo "----------------------------------Done----------------------------------"
echo "----------------------------------Done----------------------------------"

