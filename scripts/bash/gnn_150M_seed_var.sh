#!/usr/bin/env bash
# 5-seed seed-variance study on ESM2-150M:
#   variant = v15      : v1.5 baseline (gnn + hier + h3 = species,genus,family)
#   variant = vanilla  : species_mode=none (peptide-only)
#
# Each invocation runs ONE (variant, split, seed) job. Outputs:
#   ckpt   :  /NAS/luyq/PLM_AMP_Regression/gnn_runs/esm2-150M/<run>/...sd${SEED}.pth
#   log    :  /NAS/luyq/PLM_AMP_Regression/gnn_runs/esm2-150M/logs/<run>.log
#   metric :  /NAS/luyq/PLM_AMP_Regression/gnn_runs/esm2-150M/<run>/<run>.csv
# where <run> = "${variant}_seed_${split}_sd${SEED}"
#
# Usage:
#   GPU=3 VARIANT=v15     SPLIT=splits1 SEED=42   bash gnn_150M_seed_var.sh
#   GPU=4 VARIANT=vanilla SPLIT=splits2 SEED=1337 bash gnn_150M_seed_var.sh
# Mandatory env vars: GPU, VARIANT (v15|vanilla), SPLIT (splits1|splits2), SEED.

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
VARIANT=${VARIANT:?VARIANT env var is required (v15 or vanilla)}
SPLIT=${SPLIT:?SPLIT env var is required (splits1 or splits2)}
EPOCHS=${EPOCHS:-50}
BS=${BS:-32}
LR=${LR:-1e-5}

data="/NAS/luyq/AMP_datasets/${SPLIT}"
sub="${VARIANT}_seed_${SPLIT}_sd${SEED}"
ckp="${CKP_BASE}/${sub}"
log="${LOG_BASE}/${sub}.log"
mkdir -p "${ckp}"

# Variant-specific args
if [ "${VARIANT}" = "v15" ]; then
    species_args=(
        --species_mode gnn
        --taxo_graph_path "${TAXO}"
        --gnn_hidden 128 --gnn_out_dim 64 --gnn_layers 2
        --gnn_type gcn --gnn_fusion hier
        --gnn_hier_levels species,genus,family
        --fusion_strategy concat
        --species_inject post
    )
elif [ "${VARIANT}" = "vanilla" ]; then
    species_args=(--species_mode none)
else
    echo "Unknown VARIANT=${VARIANT}"; exit 2
fi

echo "================ [${PLM}] ${sub} ================"
echo "  start: $(date)"
"${PY}" train.py \
    --data_path "${data}" \
    --plm "${PLM}" \
    --finetune_plm True \
    --seed "${SEED}" \
    "${species_args[@]}" \
    --batch_size "${BS}" --lr "${LR}" --epochs "${EPOCHS}" \
    --use_scheduler --warmup_ratio 0.1 --min_lr_ratio 0.05 \
    --early_stopping --early_stop_patience 10 --es_min_epoch 20 \
    --save_dir "${ckp}" \
    --metrics_name "${sub}.csv" \
    --test_results_dir "${CKP_BASE}/seed_var_test_results" \
    --device "${GPU}" \
    > "${log}" 2>&1
rc=$?
echo "  done : $(date)  rc=${rc}"
exit ${rc}
