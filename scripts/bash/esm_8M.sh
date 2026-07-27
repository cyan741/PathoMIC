#!/usr/bin/env bash
set -x
plm="esm2-8M"

export WANDB_DIR=/NAS/luyq/wandb
export WANDB_CACHE_DIR=$WANDB_DIR
export WANDB_DATA_DIR=$WANDB_DIR
export WANDB_ARTIFACT_DIR=$WANDB_DIR

ROOT_DIR="/home/luyq/PLM_AMP_Regression/scripts"
cd "${ROOT_DIR}"
DATA_DIR="/NAS/luyq/AMP_datasets/splits2"
SAVE_DIR="/NAS/luyq/PLM_AMP_Regression/ckp_esm8m_raw/splits2"
LOG_DIR="/NAS/luyq/PLM_AMP_Regression/log_esm8m_raw/splits2"
CSV_DIR="/NAS/luyq/PLM_AMP_Regression/test_results/raw_sequence"
mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

# WANDB_API_KEY="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"
# Set available GPUs here.
# AVAILABLE_GPUS=(1 3 4 7)
# gpu_num=${#AVAILABLE_GPUS[@]}
# echo "Available GPUs: ${gpu_num}"
# gpu_idx=0
# gpu_id=${AVAILABLE_GPUS[${gpu_idx}]}
# gpu_idx=$(( (gpu_idx + 1) % gpu_num ))
gpu_id=7
bs=64
lr=1e-4
sd=7
python train.py \
    --seed ${sd}\
    --data_path ${DATA_DIR} \
    --finetune_plm True\
    --epochs 50 \
    --save_dir ${SAVE_DIR} \
    --test_results_dir ${CSV_DIR}\
    --metrics_name ${plm}_lr${lr}_bs${bs}_sd${sd}.csv \
    --plm ${plm} \
    --lr ${lr} \
    --batch_size ${bs} \
    --use_scheduler --warmup_ratio 0.1 --min_lr_ratio 0.05 \
    --early_stopping --early_stop_patience 10 --es_min_epoch 20 \
    --device ${gpu_id} > ${LOG_DIR}/${plm}_lr${lr}_bs${bs}_sd${sd}.log 2>&1 

echo "All tasks submitted, waiting for completion."
wait

