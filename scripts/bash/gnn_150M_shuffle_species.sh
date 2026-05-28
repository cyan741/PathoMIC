#!/usr/bin/env bash
# Shuffle-species control on splits1 (5 seeds).
#
# Same architecture/hyperparams as v15 (h3): GCN hier, gnn_out_dim=64, concat post.
# The ONLY change: --shuffle_species_control 1 permutes (peptide -> species)
# pairings on the TRAIN split (val/test keep true pairings).
#
# Usage:
#   GPU=1 SEED=42 bash gnn_150M_shuffle_species.sh

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
SPLIT=${SPLIT:-splits1}
EPOCHS=${EPOCHS:-50}
BS=${BS:-32}
LR=${LR:-1e-5}

data="/NAS/luyq/AMP_datasets/${SPLIT}"
sub="shuffle_species_${SPLIT}_sd${SEED}"
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
    --gnn_hidden 128 --gnn_out_dim 64 --gnn_layers 2 \
    --gnn_type gcn --gnn_fusion hier \
    --gnn_hier_levels species,genus,family \
    --fusion_strategy concat \
    --species_inject post \
    --shuffle_species_control 1 \
    --batch_size "${BS}" --lr "${LR}" --epochs "${EPOCHS}" \
    --use_scheduler --warmup_ratio 0.1 --min_lr_ratio 0.05 \
    --early_stopping --early_stop_patience 10 --es_min_epoch 20 \
    --save_dir "${ckp}" \
    --metrics_name "${sub}.csv" \
    --test_results_dir "${CKP_BASE}/shuffle_species_test_results" \
    --device "${GPU}" \
    > "${log}" 2>&1
rc=$?
echo "  done : $(date)  rc=${rc}"
exit ${rc}
