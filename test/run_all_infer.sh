#!/usr/bin/env bash
# =============================================================================
# Batch inference: run all val_best checkpoints on their corresponding test sets
#
# splits1 models  ->  splits1/test.csv                (-> splits1_test/)
# splits2 models  ->  splits2/test.csv                (-> splits2_test/)
#                     splits2/external_test.csv        (-> external_test/)
# =============================================================================
set -euo pipefail

# ---------- paths ------------------------------------------------------------
CKP_SPLITS1="/NAS/luyq/PLM_AMP_Regression/ckp/ckp_species_text/splits1"
CKP_SPLITS2="/NAS/luyq/PLM_AMP_Regression/ckp/ckp_species_text/splits2"

DATA_SPLITS1="/NAS/luyq/AMP_datasets/splits1/test.csv"
DATA_SPLITS2="/NAS/luyq/AMP_datasets/splits2/test.csv"
DATA_EXTERNAL="/NAS/luyq/AMP_datasets/splits2/external_test.csv"

WORK_DIR="/home/luyq/PLM_AMP_Regression/test"

OUT_ROOT="/home/luyq/PLM_AMP_Regression/species_text"
OUT_SPLITS1="${OUT_ROOT}/splits1_test"
OUT_SPLITS2="${OUT_ROOT}/splits2_test"
OUT_EXTERNAL="${OUT_ROOT}/external_test"

INFER_SCRIPT="${WORK_DIR}/infer.py"
BATCH_SIZE=512
DEVICE=0
LOG_FILE="${OUT_ROOT}/run_all_infer.log"

# ---------- setup ------------------------------------------------------------
mkdir -p "${OUT_SPLITS1}" "${OUT_SPLITS2}" "${OUT_EXTERNAL}"
exec > >(tee -a "${LOG_FILE}") 2>&1   # log everything to file + stdout

echo "============================================================"
echo "Batch inference started: $(date)"
echo "============================================================"

# ---------- helper function --------------------------------------------------
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
    echo "[RUN] $(date +%H:%M:%S) | model: ${model_name}"
    echo "       test : ${test_csv}"
    echo "       out  : ${result_file}"

    conda run -n pytorch3.9 python "${INFER_SCRIPT}" \
        --model_path  "${model_path}" \
        --test_csv    "${test_csv}" \
        --output_dir  "${output_dir}" \
        --batch_size  "${BATCH_SIZE}" \
        --device      "${DEVICE}" \
        2>&1 | grep -v "UserWarning\|Some weights\|newly initialized\|TRAIN this model\|warnings.warn" \
             | grep -v "^$"

    echo "[DONE] ${model_name}"
}

# ---------- splits1 models -> splits1/test.csv -------------------------------
echo ""
echo ">>> [splits1] models on splits1/test.csv"
for ckp in "${CKP_SPLITS1}"/*val_best*.pth; do
    run_infer "${ckp}" "${DATA_SPLITS1}" "${OUT_SPLITS1}"
done

# # ---------- splits2 models -> splits2/test.csv + external_test.csv -----------
# echo ""
# echo ">>> [splits2] models on splits2/test.csv"
# for ckp in "${CKP_SPLITS2}"/*val_best*.pth; do
#     run_infer "${ckp}" "${DATA_SPLITS2}" "${OUT_SPLITS2}"
# done

# echo ""
# echo ">>> [splits2] models on splits2/external_test.csv"
# for ckp in "${CKP_SPLITS2}"/*val_best*.pth; do
#     run_infer "${ckp}" "${DATA_EXTERNAL}" "${OUT_EXTERNAL}"
# done

# ---------- summary ----------------------------------------------------------
echo ""
echo "============================================================"
echo "All done: $(date)"
echo "splits1_test : $(ls "${OUT_SPLITS1}"/*.csv 2>/dev/null | wc -l) files"
# echo "splits2_test : $(ls "${OUT_SPLITS2}"/*.csv 2>/dev/null | wc -l) files"
# echo "external_test: $(ls "${OUT_EXTERNAL}"/*.csv 2>/dev/null | wc -l) files"
echo "============================================================"
