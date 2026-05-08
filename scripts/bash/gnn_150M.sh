#!/usr/bin/env bash
# Run all 6 GNN ablations at ESM2-150M to convergence (single GPU sequential).
#
# Each run is a separate train.py invocation with its own --save_dir / log /
# metrics_name so that partial progress is preserved if the chain is killed.
# Convergence is automatic via warmup+cosine LR + early stopping.
#
# Hyperparameters mirror the existing bash/esm_150M.sh + bash/esm_8M.sh
# convention (bs=32, lr=1e-5, scheduler, ES min_epoch 20 patience 10, max
# 50 epochs).

set -u  # fail on undefined var, but DO NOT use -e -- we want subsequent runs
        # to continue even if an earlier one OOMs / crashes.
set -x

PLM="esm2-150M"
PY=${PY:-/opt/conda/envs/esm-AMP/bin/python}
ROOT_DIR="/root/PLM_AMP_Regression/scripts"
cd "${ROOT_DIR}"

CKP_BASE="/NAS/luyq/PLM_AMP_Regression/gnn_runs/${PLM}"
LOG_BASE="/NAS/luyq/PLM_AMP_Regression/gnn_runs/${PLM}/logs"
mkdir -p "${CKP_BASE}" "${LOG_BASE}"

TAXO=/NAS/luyq/AMP_datasets/taxonomy_graph.pt
DEVICE=${DEVICE:-0}
EPOCHS=${EPOCHS:-50}
BS=${BS:-32}
LR=${LR:-1e-5}

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

# v1: leaf + GCN  (canonical late-concat)
run_one v1   splits1 leaf gcn
run_one v1   splits2 leaf gcn

# v1.5: hier + GCN  (long-tail / OOD hypothesis)
run_one v15  splits1 hier gcn
run_one v15  splits2 hier gcn

# v2: leaf + GAT
run_one v2   splits1 leaf gat
run_one v2   splits2 leaf gat

echo "ALL ${PLM} RUNS COMPLETE."
