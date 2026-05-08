#!/usr/bin/env bash
set -x
plm="esm2-8M"

export WANDB_DIR=/NAS/luyq/wandb
export WANDB_CACHE_DIR=$WANDB_DIR
export WANDB_DATA_DIR=$WANDB_DIR
export WANDB_ARTIFACT_DIR=$WANDB_DIR

ROOT_DIR="/home/luyq/PLM_AMP_Regression/scripts"
cd "${ROOT_DIR}"
DATA_DIR="/NAS/luyq/AMP_datasets/splits1"
SAVE_DIR="/NAS/luyq/PLM_AMP_Regression/ckp_retrain_withoutRELU/splits1"
LOG_DIR="/NAS/luyq/PLM_AMP_Regression/log_retrain_withoutRELU/splits1"
RESUME_DIR="/NAS/luyq/PLM_AMP_Regression/ckp_retrain_withoutRELU/splits1"
WANDB_API_KEY="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"
bs=64
lr=5e-06
python train.py \
    --resume ${RESUME_DIR}/${plm}_lr${lr}_bs${bs}_ep78_val_best.pth \
    --data_path ${DATA_DIR} \
    --finetune_plm True\
    --epochs 200 --batch_size ${bs} --lr ${lr} --device 3 \
    --save_dir ${SAVE_DIR} \
    --metrics_name ${plm}_lr${lr}_bs${bs}.csv \
    --use_wandb  \
    --wandb_api_key ${WANDB_API_KEY} \
    --wandb_mode "online" \
    --plm ${plm} \
    --use_scheduler --warmup_ratio 0.1 --min_lr_ratio 0.05 \
    --early_stopping --early_stop_patience 10 --es_min_epoch 90 > ${LOG_DIR}/${plm}_lr${lr}_bs${bs}_resume.log 2>&1 
