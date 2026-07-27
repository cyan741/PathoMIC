#!/usr/bin/env bash
# Sequential v1 / v1.5 / v2 GNN ablations on splits1 (OOD) + splits2 (IID).
# Each run: ESM2-35M, bs=64, lr=2e-5, 5 epochs, GNN out_dim=64.
# Logs and checkpoints land in /NAS/luyq/PLM_AMP_Regression/{logs,ckp}.

set -euo pipefail

PY=/opt/conda/envs/esm-AMP/bin/python
SCR_DIR=$(cd "$(dirname "$0")" && pwd)
LOG_DIR=/NAS/luyq/PLM_AMP_Regression/logs
CKP_DIR=/NAS/luyq/PLM_AMP_Regression/ckp
mkdir -p "$LOG_DIR" "$CKP_DIR"

EPOCHS=${EPOCHS:-5}
BS=${BS:-64}
LR=${LR:-2e-5}
PLM=${PLM:-esm2-35M}
TAXO=/NAS/luyq/AMP_datasets/taxonomy_graph.pt

run_one() {
    local tag="$1"; local split="$2"; local mode="$3"; local fusion="$4"; local gtype="$5"
    local data="/NAS/luyq/AMP_datasets/${split}"
    local sub="${tag}_${split}"
    local ckp="${CKP_DIR}/${sub}"
    local log="${LOG_DIR}/${sub}.log"
    echo
    echo "================ ${sub}  (mode=${mode} fusion=${fusion} type=${gtype}) ================"
    cd "$SCR_DIR"
    "$PY" train.py \
        --data_path "$data" \
        --plm "$PLM" \
        --species_mode "$mode" \
        --taxo_graph_path "$TAXO" \
        --gnn_hidden 128 --gnn_out_dim 64 --gnn_layers 2 \
        --gnn_type "$gtype" --gnn_fusion "$fusion" \
        --batch_size "$BS" --lr "$LR" --epochs "$EPOCHS" \
        --save_dir "$ckp" \
        --metrics_name "${sub}.csv" \
        --device 0 \
        2>&1 | tee "$log" >/dev/null
    # Print just the summary lines.
    echo "  ===> summary for ${sub}:"
    grep -E "Epoch [0-9]+ (Train|Val|Complete)|Test MSE|BUCKETED|^\[[0-9]" "$log" | tail -20
    echo "  ===> bucketed report:"
    grep -E "^(=|train-count|\[)" "$log" | tail -10
}

# v1: leaf + GCN  (the canonical "GNN late-concat" baseline)
run_one v1   splits1 gnn leaf gcn
run_one v1   splits2 gnn leaf gcn

# v1.5: hier + GCN  (species + genus + family fusion; long-tail hypothesis)
run_one v15  splits1 gnn hier gcn
run_one v15  splits2 gnn hier gcn

# v2: leaf + GAT  (attention conv ablation)
run_one v2   splits1 gnn leaf gat
run_one v2   splits2 gnn leaf gat

echo
echo "ALL RUNS COMPLETE."
echo "Logs:        ${LOG_DIR}"
echo "Checkpoints: ${CKP_DIR}"
