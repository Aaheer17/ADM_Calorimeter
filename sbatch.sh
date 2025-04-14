#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=ADM_1000
##SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH --mem=80000
##SBATCH --constraint=a100_80gb
#SBATCH -p bii-gpu
#SBATCH --gres=gpu:1
#SBATCH -A bii_nssac
module load miniforge
#module load texlive

source activate ardiff
# source activate torch_gpu_renew
# export PATH=~/bin:$PATH
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#which dvipng
python3 main.py /project/biocomplexity/fa7sa/Diffusion_multi_step/config.yaml --use_cuda -l 'loss.png'
