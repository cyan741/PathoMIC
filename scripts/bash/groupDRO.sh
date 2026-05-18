#!/usr/bin/env bash
# Run GNN ablations with group DRO loss 
set -u  # fail on undefined var, but DO NOT use -e -- we want subsequent runs
        # to continue even if an earlier one OOMs / crashes.
set -x

PLM="esm2-150M"
PY=${PY:-/opt/conda/envs/esm-AMP/bin/python}
ROOT_DIR="/root/PLM_AMP_Regression/scripts"
cd "${ROOT_DIR}"

CKP_BASE="/NAS/luyq/PLM_AMP_Regression/debug_groupDRO/ckp"
LOG_BASE="/NAS/luyq/PLM_AMP_Regression/debug_groupDRO/logs"
mkdir -p "${CKP_BASE}" "${LOG_BASE}"

TAXO=/NAS/luyq/AMP_datasets/taxonomy_graph.pt
DEVICE=${DEVICE:-0}
EPOCHS=${EPOCHS:-50}
BS=${BS:-32}
LR=${LR:-1e-5}

run_one() {
    local split="$2"; local fusion="$3"; local gtype="$4"; local loss_dro_base="$5"
    local wd="$6"
    local data="/NAS/luyq/AMP_datasets/${split}"
    local sub="${loss_dro_base}_wd${wd}_${split}"
    local ckp="${CKP_BASE}/${split}/${sub}"
    local log="${LOG_BASE}/${sub}.log"
    local metrics_name="${PLM}_lr${LR}_bs${BS}"
    mkdir -p "${ckp}"

    echo "================ [${PLM}] ${sub} (fusion=${fusion} gnn=${gtype}) ================"
    echo "  start: $(date)"
    "${PY}" train.py \
        --data_path "${data}" \
        --plm "${PLM}" \
        --finetune_plm True \
        --species_mode gnn \
        --taxo_graph_path "${TAXO}" \
        --gnn_hidden 128 --gnn_out_dim 64 --gnn_layers 2 \
        --gnn_type "${gtype}" --gnn_fusion "${fusion}" \
        --batch_size "${BS}" --lr "${LR}" --epochs "${EPOCHS}" \
        --weight_decay "${wd}" \
        --loss_type group_dro \
        --loss_dro_group_by bucket \
        --loss_dro_base "${loss_dro_base}" \
        --loss_dro_normalize_loss \
        --loss_dro_log_path "${ckp}/group_dro.log" \
        --use_scheduler --warmup_ratio 0.1 --min_lr_ratio 0.05 \
        --early_stopping --early_stop_patience 10 --es_min_epoch 20 \
        --save_dir "${ckp}" \
        --metrics_name "${metrics_name}.csv" \
        --device "${DEVICE}" \
        > "${log}" 2>&1 || echo "  [warn] ${sub} exited non-zero -- continuing"
    echo "  done : $(date)"
}

run_one splits1 hier gcn huber 1e-4
run_one splits1 hier gcn mse 1e-4
run_one splits2 hier gcn huber 1e-4
run_one splits2 hier gcn mse 1e-4

run_one splits1 hier gcn huber 0
run_one splits1 hier gcn mse 0
run_one splits2 hier gcn huber 0
run_one splits2 hier gcn mse 0




