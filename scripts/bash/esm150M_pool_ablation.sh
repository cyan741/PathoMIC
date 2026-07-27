#!/usr/bin/env bash
# Input-protocol ablation on the sequence-only reference model (ESM2-150M-Raw).
#
# Purpose: the main experiments right-fill every peptide with the ambiguous-residue
# symbol 'X' to a fixed width of 70 and mean-pool over all 72 token positions.
# Note that this is NOT an unmasked-padding bug: the filler is a real alphabet
# symbol, the tokenizer never emits <pad>, and the attention mask is all-ones, so
# passing a mask would be a no-op. The only open question is whether sizing the
# window to the true peptide length and pooling over residues alone would lower
# the absolute error. This script answers that on the reference model only, which
# is enough to show the protocol does not drive the Raw/SP/GNN differences.
#
#   VARIANT=masked : --pad_mode none  --plm_output mean_masked
#                    tokenizer pads each batch to its longest member; pooling
#                    averages over true residues only (<cls>/<eos>/<pad> dropped).
#   VARIANT=fill   : --pad_mode fill  --plm_output mean  (the published protocol)
#                    Only needed if you want the paired control re-run inside this
#                    output tree; the published Raw numbers already are this arm
#                    (gnn_runs/esm2-150M/vanilla_seed_splits2_sd${SEED}).
#
# Everything else is held identical to gnn_150M_seed_var.sh VARIANT=vanilla:
# species_mode=none, lr 1e-5, bs 32, 50 epochs, warmup 0.1 + cosine to 0.05,
# early stopping patience 10 from epoch 20, full ESM2 fine-tuning, MSE loss.
#
# Usage:
#   GPU=0 SEED=42 bash esm150M_pool_ablation.sh
#   GPU=4 SEED=42 VARIANT=fill bash esm150M_pool_ablation.sh
#
# NOT launched as part of the main study. Run only if a reviewer asks for it.

set -u
PLM="esm2-150M"
PY=${PY:-/home/luyq/miniconda3/envs/pytorch3.9/bin/python}
ROOT_DIR="/home/luyq/PLM_AMP_Regression/scripts"
cd "${ROOT_DIR}"

GPU=${GPU:?GPU env var is required}
SEED=${SEED:-42}
VARIANT=${VARIANT:-masked}
# The in-distribution split formerly at /NAS/luyq/AMP_datasets/splits2.
DATA=${DATA:-/NAS/luyq/AMP_datasets/PathoMIC/splits}
EPOCHS=${EPOCHS:-50}
BS=${BS:-32}
LR=${LR:-1e-5}

OUT_BASE="/NAS/luyq/PLM_AMP_Regression/gnn_runs/${PLM}/pool_ablation"
LOG_DIR="${OUT_BASE}/logs"

case "${VARIANT}" in
    masked) pool_args=(--pad_mode none --plm_output mean_masked) ;;
    fill)   pool_args=(--pad_mode fill --plm_output mean --max_seq_len 70) ;;
    *)      echo "Unknown VARIANT=${VARIANT} (expected 'masked' or 'fill')" >&2; exit 2 ;;
esac

if [ ! -f "${DATA}/train.csv" ]; then
    echo "No train.csv under DATA=${DATA}" >&2
    exit 2
fi

sub="pool_${VARIANT}_sd${SEED}"
ckp="${OUT_BASE}/${sub}"
log="${LOG_DIR}/${sub}.log"
mkdir -p "${ckp}" "${LOG_DIR}"

echo "================ [${PLM}] ${sub} ================"
echo "  data=${DATA}  gpu=${GPU}  lr=${LR}  bs=${BS}  pool=${pool_args[*]}"
echo "  start: $(date)"
PYTHONUNBUFFERED=1 "${PY}" train.py \
    --data_path "${DATA}" \
    --plm "${PLM}" \
    --finetune_plm True \
    --seed "${SEED}" \
    --species_mode none \
    "${pool_args[@]}" \
    --batch_size "${BS}" --lr "${LR}" --epochs "${EPOCHS}" \
    --use_scheduler --warmup_ratio 0.1 --min_lr_ratio 0.05 \
    --early_stopping --early_stop_patience 10 --es_min_epoch 20 \
    --save_dir "${ckp}" \
    --metrics_name "${sub}.csv" \
    --test_results_dir "${OUT_BASE}/test_results" \
    --device "${GPU}" \
    > "${log}" 2>&1
rc=$?
echo "  done : $(date)  rc=${rc}"
exit ${rc}
