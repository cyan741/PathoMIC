#!/usr/bin/env bash
# Random-embedding baseline on splits2.
#
# Replaces the 768-d PubMedBERT node features with random Gaussian vectors
# (scaled to the real features' std) before the GNN. Everything else matches
# v15: GCN 2-layer hier (species,genus,family), post concat fusion, ESM2-150M
# full fine-tune, lr=1e-5, bs=32, 50 epochs. ONLY difference vs v15 is:
#   --gnn_random_init 1   and   --gnn_out_dim 640 (concat640 output channel).
# Isolates whether the pretrained text semantics matter beyond graph topology.
#
# Usage:
#   GPU=1 SEED=7 bash gnn_150M_random_init.sh

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
OUT_DIM=${OUT_DIM:-640}

data="/NAS/luyq/AMP_datasets/${SPLIT}"
sub="random_init_${SPLIT}_sd${SEED}"
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
    --gnn_hidden 128 --gnn_out_dim "${OUT_DIM}" --gnn_layers 2 \
    --gnn_type gcn --gnn_fusion hier \
    --gnn_hier_levels species,genus,family \
    --fusion_strategy concat \
    --species_inject post \
    --gnn_random_init 1 \
    --batch_size "${BS}" --lr "${LR}" --epochs "${EPOCHS}" \
    --use_scheduler --warmup_ratio 0.1 --min_lr_ratio 0.05 \
    --early_stopping --early_stop_patience 10 --es_min_epoch 20 \
    --save_dir "${ckp}" \
    --metrics_name "${sub}.csv" \
    --test_results_dir "${CKP_BASE}/random_init_test_results" \
    --device "${GPU}" \
    > "${log}" 2>&1
rc=$?
echo "  done : $(date)  rc=${rc}"
exit ${rc}
