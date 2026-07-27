#!/usr/bin/env bash
set -x
plm="esm2-8M"

ROOT_DIR="/home/luyq/PLM_AMP_Regression/scripts"
cd "${ROOT_DIR}"
DATA_DIR="/NAS/luyq/AMP_datasets/splits2"
SAVE_DIR="/NAS/luyq/PLM_AMP_Regression/ckp_esm8m_sp/splits2"
LOG_DIR="/NAS/luyq/PLM_AMP_Regression/log_esm8m_sp/splits2"
CSV_DIR="/NAS/luyq/PLM_AMP_Regression/test_results/species_text"
mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

bs=64
lr=1e-4
sd=123
gpu_id=4

python train.py \
        --seed ${sd} \
        --epochs 50 \
            --use_species \
            --species_emb_path /NAS/luyq/AMP_datasets/species_embeddings.pkl \
            --data_path ${DATA_DIR} \
            --finetune_plm True\
            --save_dir ${SAVE_DIR} \
            --test_results_dir ${CSV_DIR} \
            --metrics_name ${plm}_lr${lr}_bs${bs}_sd${sd}.csv \
            --plm ${plm} \
            --lr ${lr} \
            --use_scheduler --warmup_ratio 0.1 --min_lr_ratio 0.05 \
            --early_stopping --early_stop_patience 10 --es_min_epoch 20 \
            --device  ${gpu_id} \
            --batch_size ${bs}  > ${LOG_DIR}/${plm}_lr${lr}_bs${bs}_sd${sd}.log 2>&1
