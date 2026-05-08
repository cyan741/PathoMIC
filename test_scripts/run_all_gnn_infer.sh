#!/usr/bin/env bash
# =============================================================================
# Batch inference for the 6 ESM2-150M GNN checkpoints.
#
#   splits1 models  ->  splits1/test.csv
#                       (-> tax_gnn/esm150m/splits1_test/csv/)
#   splits2 models  ->  splits2/test.csv
#                       (-> tax_gnn/esm150m/splits2_test/csv/)
#                   ->  splits2/external_test.csv
#                       (-> tax_gnn/esm150m/external_test/csv/)
#
# Each result file is named   <model_name>_test_results.csv
# =============================================================================
set -uo pipefail   # do NOT use -e: we want to keep going if one model fails.

# ---------- paths ------------------------------------------------------------
GNN_ROOT="/NAS/luyq/PLM_AMP_Regression/gnn_runs/esm2-150M"
SPLITS1_RUNS=(v1_splits1 v15_splits1 v2_splits1)
SPLITS2_RUNS=(v1_splits2 v15_splits2 v2_splits2)

DATA_SPLITS1="/NAS/luyq/AMP_datasets/splits1/test.csv"
DATA_SPLITS2="/NAS/luyq/AMP_datasets/splits2/test.csv"
DATA_EXTERNAL="/NAS/luyq/AMP_datasets/splits2/external_test.csv"

TAXO="/NAS/luyq/AMP_datasets/taxonomy_graph.pt"
SPECIES_EMB="/NAS/luyq/AMP_datasets/species_embeddings.pkl"

OUT_ROOT="/root/PLM_AMP_Regression/test_results/tax_gnn/esm150m"
OUT_SPLITS1="${OUT_ROOT}/splits1_test/csv"
OUT_SPLITS2="${OUT_ROOT}/splits2_test/csv"
OUT_EXTERNAL="${OUT_ROOT}/external_test/csv"

INFER_SCRIPT="/root/PLM_AMP_Regression/test_scripts/infer.py"
PY=${PY:-/opt/conda/envs/esm-AMP/bin/python}
BATCH_SIZE=${BS:-128}
DEVICE=${DEVICE:-0}
LOG_FILE="${OUT_ROOT}/run_all_gnn_infer.log"

mkdir -p "${OUT_SPLITS1}" "${OUT_SPLITS2}" "${OUT_EXTERNAL}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "GNN batch inference started: $(date)"
echo "  device : ${DEVICE}"
echo "  batch  : ${BATCH_SIZE}"
echo "  taxo   : ${TAXO}"
echo "  log    : ${LOG_FILE}"
echo "============================================================"

# ---------- helper -----------------------------------------------------------
find_val_best() {
    # Echo the val_best checkpoint inside a run folder, or empty string.
    local run_dir="$1"
    local f
    f=$(ls "${run_dir}"/*val_best*.pth 2>/dev/null | head -n 1)
    echo "${f}"
}

run_infer() {
    local model_path="$1"
    local test_csv="$2"
    local output_dir="$3"

    local model_name
    model_name=$(basename "${model_path}" .pth)
    local result_file="${output_dir}/${model_name}_test_results.csv"

    if [[ -f "${result_file}" ]]; then
        echo "[SKIP] Already exists: ${result_file}"
        return 0
    fi

    echo ""
    echo "[RUN] $(date +%H:%M:%S) | model: ${model_path}"
    echo "       test : ${test_csv}"
    echo "       out  : ${result_file}"

    "${PY}" "${INFER_SCRIPT}" \
        --model_path       "${model_path}" \
        --test_csv         "${test_csv}" \
        --output_dir       "${output_dir}" \
        --taxo_graph_path  "${TAXO}" \
        --species_emb_path "${SPECIES_EMB}" \
        --batch_size       "${BATCH_SIZE}" \
        --device           "${DEVICE}" \
        2>&1 | grep -v -E "UserWarning|Some weights|newly initialized|TRAIN this model|warnings.warn|FutureWarning"
    echo "[DONE] ${model_name}"
}

# ---------- splits1 GNN models -> splits1/test.csv ---------------------------
echo ""
echo ">>> [splits1 GNN] models on splits1/test.csv"
for run in "${SPLITS1_RUNS[@]}"; do
    ckp=$(find_val_best "${GNN_ROOT}/${run}")
    if [[ -z "${ckp}" ]]; then
        echo "[WARN] no val_best found in ${GNN_ROOT}/${run}, skipping."
        continue
    fi
    # Rename the file in the result dir to include the run tag (v1 / v15 / v2)
    # so multiple runs with the same epoch/lr filename don't collide.
    out_dir_with_tag="${OUT_SPLITS1}"
    model_name=$(basename "${ckp}" .pth)
    tagged_target="${out_dir_with_tag}/${run}_${model_name}_test_results.csv"
    if [[ -f "${tagged_target}" ]]; then
        echo "[SKIP] Already exists: ${tagged_target}"
        continue
    fi
    run_infer "${ckp}" "${DATA_SPLITS1}" "${out_dir_with_tag}"
    # Rename to include the run tag
    if [[ -f "${out_dir_with_tag}/${model_name}_test_results.csv" ]]; then
        mv "${out_dir_with_tag}/${model_name}_test_results.csv" "${tagged_target}"
        echo "[RENAME] -> ${tagged_target}"
    fi
done

# ---------- splits2 GNN models -> splits2/test.csv ---------------------------
echo ""
echo ">>> [splits2 GNN] models on splits2/test.csv"
for run in "${SPLITS2_RUNS[@]}"; do
    ckp=$(find_val_best "${GNN_ROOT}/${run}")
    if [[ -z "${ckp}" ]]; then
        echo "[WARN] no val_best found in ${GNN_ROOT}/${run}, skipping."
        continue
    fi
    out_dir_with_tag="${OUT_SPLITS2}"
    model_name=$(basename "${ckp}" .pth)
    tagged_target="${out_dir_with_tag}/${run}_${model_name}_test_results.csv"
    if [[ -f "${tagged_target}" ]]; then
        echo "[SKIP] Already exists: ${tagged_target}"
        continue
    fi
    run_infer "${ckp}" "${DATA_SPLITS2}" "${out_dir_with_tag}"
    if [[ -f "${out_dir_with_tag}/${model_name}_test_results.csv" ]]; then
        mv "${out_dir_with_tag}/${model_name}_test_results.csv" "${tagged_target}"
        echo "[RENAME] -> ${tagged_target}"
    fi
done

# ---------- splits2 GNN models -> splits2/external_test.csv ------------------
echo ""
echo ">>> [splits2 GNN] models on splits2/external_test.csv"
for run in "${SPLITS2_RUNS[@]}"; do
    ckp=$(find_val_best "${GNN_ROOT}/${run}")
    if [[ -z "${ckp}" ]]; then
        echo "[WARN] no val_best found in ${GNN_ROOT}/${run}, skipping."
        continue
    fi
    out_dir_with_tag="${OUT_EXTERNAL}"
    model_name=$(basename "${ckp}" .pth)
    tagged_target="${out_dir_with_tag}/${run}_${model_name}_test_results.csv"
    if [[ -f "${tagged_target}" ]]; then
        echo "[SKIP] Already exists: ${tagged_target}"
        continue
    fi
    run_infer "${ckp}" "${DATA_EXTERNAL}" "${out_dir_with_tag}"
    if [[ -f "${out_dir_with_tag}/${model_name}_test_results.csv" ]]; then
        mv "${out_dir_with_tag}/${model_name}_test_results.csv" "${tagged_target}"
        echo "[RENAME] -> ${tagged_target}"
    fi
done

# ---------- summary ----------------------------------------------------------
echo ""
echo "============================================================"
echo "All done: $(date)"
echo "splits1_test : $(ls "${OUT_SPLITS1}"/*.csv 2>/dev/null | wc -l) files"
echo "splits2_test : $(ls "${OUT_SPLITS2}"/*.csv 2>/dev/null | wc -l) files"
echo "external_test: $(ls "${OUT_EXTERNAL}"/*.csv 2>/dev/null | wc -l) files"
echo "============================================================"
