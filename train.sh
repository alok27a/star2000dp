#!/bin/bash

#SBATCH -t 7-00:00
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

python train.py \
    --output_dir custom_outputs \
    --sequence_len 50 \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --log_level info \
    --seed 42 \
    --noise_multiplier 1.0 \
    --per_sample_max_grad_norm 1.0 \
    --prediction_loss_only \
    --weight_decay 0 \
    --remove_unused_columns False \
    --num_train_epochs 2000.0 \
    --max_grad_norm 0 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-3 \
    --disable_tqdm True \
    --dataloader_num_workers 1 \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 8 \