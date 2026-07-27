#!/usr/bin/env bash
# Species-adapter shuffle control on splits2 (train-only species permutation).
#
# Usage:
#   GPU=4 SEED=42 bash esm150M_adapter_shuffle_splits2.sh
#   GPU=3 RESUME=1 SEED=42 bash esm150M_adapter_shuffle_splits2.sh
#
# Checkpoints / metrics / bucket CSV:
#   /NAS/luyq/PLM_AMP_Regression/test_results/species_text/shuffle/ckp/
# Test prediction CSV:
#   .../shuffle/esm2-150M/splits2_test/csv/

set -u
PLM="esm2-150M"
PY=${PY:-/home/luyq/miniconda3/envs/pytorch3.9/bin/python}
ROOT_DIR="/home/luyq/PLM_AMP_Regression/scripts"
cd "${ROOT_DIR}"

OUT_BASE="/NAS/luyq/PLM_AMP_Regression/test_results/species_text/shuffle"
CKP_DIR="${OUT_BASE}/ckp"
LOG_DIR="${OUT_BASE}/logs"
mkdir -p "${CKP_DIR}" "${LOG_DIR}"

GPU=${GPU:?GPU env var required}
SEED=${SEED:-42}
SPLIT=${SPLIT:-splits2}
LR=${LR:-1e-4}
BS=${BS:-64}
EPOCHS=${EPOCHS:-50}

data="/NAS/luyq/AMP_datasets/${SPLIT}"
sub="adapter_shuffle_${SPLIT}_sd${SEED}"
log="${LOG_DIR}/${sub}$([ -n "${RESUME:-}" ] && echo '_resume').log"

RESUME_ARGS=()
if [ -n "${RESUME:-}" ]; then
    if [ -n "${RESUME_CKPT:-}" ]; then
        CKPT="${RESUME_CKPT}"
    else
        CKPT=$(ls -t "${CKP_DIR}"/*_val_best_sd${SEED}.pth 2>/dev/null | head -1)
        if [ -z "${CKPT}" ]; then
            CKPT=$(ls -t "${CKP_DIR}"/*_sd${SEED}.pth 2>/dev/null | head -1)
        fi
    fi
    if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
        echo "Resume checkpoint not found under ${CKP_DIR} (seed=${SEED})" >&2
        exit 1
    fi
    RESUME_ARGS=(--resume "${CKPT}")
    echo "  resume: ${CKPT}"
fi

echo "================ [${PLM}] ${sub} ================"
echo "  out=${OUT_BASE}  gpu=${GPU}  lr=${LR}  bs=${BS}"
echo "  start: $(date)"
"${PY}" train.py \
    --data_path "${data}" \
    --plm "${PLM}" \
    --finetune_plm True \
    --seed "${SEED}" \
    --species_mode adapter \
    --species_emb_path /NAS/luyq/AMP_datasets/species_embeddings.pkl \
    --shuffle_species_control 1 \
    --batch_size "${BS}" \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --use_scheduler --warmup_ratio 0.1 --min_lr_ratio 0.05 \
    --early_stopping --early_stop_patience 10 --es_min_epoch 20 \
    --save_dir "${CKP_DIR}" \
    --metrics_name "${sub}.csv" \
    --test_results_dir "${OUT_BASE}" \
    --device "${GPU}" \
    "${RESUME_ARGS[@]}" \
    > "${log}" 2>&1
rc=$?
echo "  done : $(date)  rc=${rc}"
exit ${rc}
