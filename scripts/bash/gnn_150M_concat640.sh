#!/usr/bin/env bash
# f_concat640: v15 GNN + concat fusion with gnn_out_dim=640 (ESM hidden width).
# Same setting as Stage-3 f_concat_out640 on splits2 (original run: seed=42 only).
#
# Usage:
#   GPU=0 SEED=7 bash gnn_150M_concat640.sh
#   GPU=4 SEED=123 SPLIT=splits2 bash gnn_150M_concat640.sh

set -u
PLM="esm2-150M"
PY=${PY:-/home/luyq/miniconda3/envs/pytorch3.9/bin/python}
ROOT_DIR="/home/luyq/PLM_AMP_Regression/scripts"
cd "${ROOT_DIR}"

CKP_BASE="/NAS/luyq/PLM_AMP_Regression/gnn_runs/${PLM}"
LOG_BASE="${CKP_BASE}/logs"
mkdir -p "${CKP_BASE}" "${LOG_BASE}"
TAXO=/NAS/luyq/AMP_datasets/taxonomy_graph.pt

GPU=${GPU:?GPU env var is required}
SEED=${SEED:?SEED env var is required}
SPLIT=${SPLIT:-splits2}
EPOCHS=${EPOCHS:-50}
BS=${BS:-32}
LR=${LR:-1e-5}

data="/NAS/luyq/AMP_datasets/${SPLIT}"
sub="f_concat640_${SPLIT}_sd${SEED}"
ckp="${CKP_BASE}/${sub}"
log="${LOG_BASE}/${sub}.log"
mkdir -p "${ckp}"

echo "================ [${PLM}] ${sub} ================"
echo "  start: $(date)"
"${PY}" train.py \
    --data_path "${data}" \
    --plm "${PLM}" \
    --finetune_plm True \
    --seed "${SEED}" \
    --species_mode gnn \
    --taxo_graph_path "${TAXO}" \
    --gnn_hidden 128 --gnn_out_dim 640 --gnn_layers 2 \
    --gnn_type gcn --gnn_fusion hier \
    --gnn_hier_levels species,genus,family \
    --gnn_freeze_init 1 \
    --fusion_strategy concat \
    --species_inject post \
    --batch_size "${BS}" --lr "${LR}" --epochs "${EPOCHS}" \
    --use_scheduler --warmup_ratio 0.1 --min_lr_ratio 0.05 \
    --early_stopping --early_stop_patience 10 --es_min_epoch 20 \
    --save_dir "${ckp}" \
    --metrics_name "${sub}.csv" \
    --test_results_dir "${CKP_BASE}/concat640_test_results" \
    --device "${GPU}" \
    > "${log}" 2>&1
rc=$?
echo "  done : $(date)  rc=${rc}"
exit ${rc}
