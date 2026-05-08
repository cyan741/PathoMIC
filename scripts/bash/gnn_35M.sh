#!/usr/bin/env bash
# Run all 6 GNN ablations at ESM2-35M to convergence (single GPU sequential).
# See bash/gnn_150M.sh for the design rationale.
#
# 35M is ~5x faster than 150M, so we use a slightly larger batch and lr.

set -u
set -x

PLM="esm2-35M"
PY=${PY:-/opt/conda/envs/esm-AMP/bin/python}
ROOT_DIR="/root/PLM_AMP_Regression/scripts"
cd "${ROOT_DIR}"

CKP_BASE="/NAS/luyq/PLM_AMP_Regression/gnn_runs/${PLM}"
LOG_BASE="/NAS/luyq/PLM_AMP_Regression/gnn_runs/${PLM}/logs"
mkdir -p "${CKP_BASE}" "${LOG_BASE}"

TAXO=/NAS/luyq/AMP_datasets/taxonomy_graph.pt
DEVICE=${DEVICE:-0}
EPOCHS=${EPOCHS:-50}
BS=${BS:-64}
LR=${LR:-2e-5}

run_one() {
    local tag="$1"; local split="$2"; local fusion="$3"; local gtype="$4"
    local data="/NAS/luyq/AMP_datasets/${split}"
    local sub="${tag}_${split}"
    local ckp="${CKP_BASE}/${sub}"
    local log="${LOG_BASE}/${sub}.log"
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
        --use_scheduler --warmup_ratio 0.1 --min_lr_ratio 0.05 \
        --early_stopping --early_stop_patience 10 --es_min_epoch 20 \
        --save_dir "${ckp}" \
        --metrics_name "${sub}.csv" \
        --device "${DEVICE}" \
        > "${log}" 2>&1 || echo "  [warn] ${sub} exited non-zero -- continuing"
    echo "  done : $(date)"
}

run_one v1   splits1 leaf gcn
run_one v1   splits2 leaf gcn
run_one v15  splits1 hier gcn
run_one v15  splits2 hier gcn
run_one v2   splits1 leaf gat
run_one v2   splits2 leaf gat

echo "ALL ${PLM} RUNS COMPLETE."
