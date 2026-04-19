#!/usr/bin/env bash
set -x
plm="esm2-35M"
export WANDB_DIR=/NAS/luyq/wandb
export WANDB_CACHE_DIR=$WANDB_DIR
export WANDB_DATA_DIR=$WANDB_DIR
export WANDB_ARTIFACT_DIR=$WANDB_DIR

ROOT_DIR="/home/luyq/PLM_AMP_Regression/scripts"
cd "${ROOT_DIR}"
DATA_DIR="/NAS/luyq/AMP_datasets/splits2"
SAVE_DIR="/NAS/luyq/PLM_AMP_Regression/ckp/splits2"
LOG_DIR="/NAS/luyq/PLM_AMP_Regression/log/splits2"
WANDB_API_KEY="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"
# Set available GPUs here.
AVAILABLE_GPUS=(0 1 3 6 5)
gpu_num=${#AVAILABLE_GPUS[@]}
echo "Available GPUs: ${gpu_num}"
gpu_idx=0
for bs in 8 16 32; do 
    for lr in 5e-6 1e-5 1e-4; do
        gpu_id=${AVAILABLE_GPUS[${gpu_idx}]}
        gpu_idx=$(( (gpu_idx + 1) % gpu_num ))
        python train.py \
            --data_path ${DATA_DIR} \
            --finetune_plm True\
            --epochs 30 \
            --save_dir ${SAVE_DIR} \
            --metrics_name ${plm}_lr${lr}_bs${bs}.csv \
            --use_wandb  \
            --wandb_api_key ${WANDB_API_KEY} \
            --wandb_mode "online" \
            --plm ${plm} \
            --lr ${lr} \
            --batch_size ${bs} \
            --device ${gpu_id} > ${LOG_DIR}/${plm}_lr${lr}_bs${bs}.log 2>&1 &
    done
done

echo "All tasks submitted, waiting for completion."
wait
