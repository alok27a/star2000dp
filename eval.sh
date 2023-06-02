#!/bin/bash

#SBATCH -t 1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hguan6@asu.edu
#SBATCH -p gpu
#SBATCH -q wildfire
#SBATCH --gres=gpu:V100:1
#SBATCH -n 4
#SBATCH --mem 64G
#SBATCH -N 1

source ~/conda.source
conda activate pp

python eval.py 