#!/usr/bin/env bash
# =============================================================================
# orchestrate.sh - launch one stage at a time, with auto-pick-winner of the
# previous stage's runs.
#
# Usage
# -----
#   # Wait until the currently-running stage finishes, then pick its winner:
#   ./orchestrate.sh select <stage_id>
#
#   # Generate this-stage's jobs from prior winners and launch:
#   ./orchestrate.sh launch <stage_id>
#
#   # Wait + select + generate + launch in one go (recommended):
#   ./orchestrate.sh next <next_stage_id>
#
# Stage IDs: 1, 2, 3, 4a, 4b, 4c, 4d, 4e, 4f, 4g, 4h, 5
#
# Examples
# --------
#   ./orchestrate.sh select 1
#   ./orchestrate.sh launch 2          # uses stage1/winner.json
#   ./orchestrate.sh next 3            # selects stage2 winner, then launches 3
# =============================================================================
set -uo pipefail

GPUS="${GPUS:-0,1,2,3}"
V2_ROOT="${V2_ROOT:-/NAS/luyq/PLM_AMP_Regression/gnn_runs_v2}"
SCRIPTS_BASH="/root/PLM_AMP_Regression/scripts/bash"
STAGES_DIR="${SCRIPTS_BASH}/stages"
RUNNER="${SCRIPTS_BASH}/parallel_runner.sh"
PY="${PY:-/opt/conda/envs/esm-AMP/bin/python}"

# ---------------------------------------------------------------------------
variants_for_stage() {
    case "$1" in
        1)  echo "h5,h7,hier_attn"           ;;
        2)  echo "q2_unfrozen_lr1e5,q2_unfrozen_lr5e6,q2_lora_r16" ;;
        3)  echo "f_concat_out64,f_concat_out640,f_gated,f_film,f_xattn,f_bilinear" ;;
        # For Stages 4a..4h and 5, the variant tags are stable in gen_jobs.py
        # but vary in count; we glob them at runtime instead.
        4a|4b|4c|4d|4e|4f|4g|4h|5)
            ls -d "${V2_ROOT}/stage$1"/*_splits1 2>/dev/null \
                | xargs -n1 basename \
                | sed 's/_splits1$//' \
                | sort -u | paste -sd,
            ;;
        *) echo "" ;;
    esac
}

prev_stage_winner_files() {
    case "$1" in
        2)  echo "${V2_ROOT}/stage1/winner.json" ;;
        3)  echo "${V2_ROOT}/stage1/winner.json ${V2_ROOT}/stage2/winner.json" ;;
        4a) echo "${V2_ROOT}/stage3/winner.json" ;;
        4b) echo "${V2_ROOT}/stage4a/winner.json" ;;
        4c) echo "${V2_ROOT}/stage4b/winner.json" ;;
        4d) echo "${V2_ROOT}/stage4c/winner.json" ;;
        4e) echo "${V2_ROOT}/stage4d/winner.json" ;;
        4f) echo "${V2_ROOT}/stage4e/winner.json" ;;
        4g) echo "${V2_ROOT}/stage4f/winner.json" ;;
        4h) echo "${V2_ROOT}/stage4f/winner.json ${V2_ROOT}/stage4g/winner.json" ;;
        5)  echo "${V2_ROOT}/stage4h/winner.json" ;;
        *)  echo "" ;;
    esac
}

# ---------------------------------------------------------------------------
do_select() {
    local stage="$1"
    local variants
    variants="$(variants_for_stage "${stage}")"
    if [[ -z "${variants}" ]]; then
        echo "[orchestrate] cannot determine variants for stage ${stage}." >&2
        exit 1
    fi
    "${PY}" "${STAGES_DIR}/select_winner.py" \
        --stage_dir "${V2_ROOT}/stage${stage}" \
        --variants "${variants}" \
        --output "${V2_ROOT}/stage${stage}/winner.json"
}

do_gen() {
    local stage="$1"
    local out="${STAGES_DIR}/stage${stage}_jobs.txt"
    local winners
    read -ra winners <<< "$(prev_stage_winner_files "${stage}")"
    "${PY}" "${STAGES_DIR}/gen_jobs.py" \
        --stage "${stage}" --output "${out}" \
        --winner_files "${winners[@]}"
}

do_run() {
    local stage="$1"
    local jobs_file="${STAGES_DIR}/stage${stage}_jobs.txt"
    local stage_dir="${V2_ROOT}/stage${stage}"
    local log_dir="${stage_dir}/logs"
    mkdir -p "${log_dir}"
    if [[ ! -f "${jobs_file}" ]]; then
        echo "[orchestrate] ERROR: ${jobs_file} not found. Run \`gen\` first." >&2
        exit 1
    fi
    GPUS="${GPUS}" LOG_DIR="${log_dir}" \
        "${RUNNER}" "${jobs_file}" \
        > "${stage_dir}/runner.log" 2>&1 &
    echo "[orchestrate] launched stage ${stage} runner pid=$! log=${stage_dir}/runner.log"
}

do_launch() {
    local stage="$1"
    do_gen "${stage}"
    do_run "${stage}"
}

do_next() {
    local next="$1"
    local prev
    case "${next}" in
        2)  prev=1   ;;
        3)  prev=2   ;;
        4a) prev=3   ;;
        4b) prev=4a  ;;
        4c) prev=4b  ;;
        4d) prev=4c  ;;
        4e) prev=4d  ;;
        4f) prev=4e  ;;
        4g) prev=4f  ;;
        4h) prev=4g  ;;
        5)  prev=4h  ;;
        *)  echo "Unknown next stage: ${next}" >&2; exit 1 ;;
    esac
    echo "[orchestrate] selecting winner of stage ${prev} ..."
    do_select "${prev}"
    echo "[orchestrate] generating + launching stage ${next} ..."
    do_launch "${next}"
}

# ---------------------------------------------------------------------------
case "${1:-}" in
    select)  do_select  "$2" ;;
    gen)     do_gen     "$2" ;;
    run)     do_run     "$2" ;;
    launch)  do_launch  "$2" ;;
    next)    do_next    "$2" ;;
    *) cat <<EOF
Usage: orchestrate.sh <command> <stage_id>
Commands:
  select  -- pick winner.json from completed stage's runs
  gen     -- generate stage<id>_jobs.txt from prior winners
  launch  -- gen + run a stage
  next    -- (recommended) select prev winner, then launch next stage
EOF
        exit 1 ;;
esac
