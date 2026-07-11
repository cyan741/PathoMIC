#!/usr/bin/env bash
# Launch 40 prefix-tune jobs (8 variants x 5 seeds) on 8 GPUs, 5 concurrent per GPU.
# Runs the dispatcher inside a detached tmux session so SSH disconnect is safe.
#
#   bash run_prefix_tune_tmux.sh          # start
#   tmux attach -t prefix_tune_40         # watch
#   tmux kill-session -t prefix_tune_40   # stop dispatcher (child train jobs keep running)

set -u
SESSION="${SESSION:-prefix_tune_40}"
BASH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/opt/conda/envs/esm-AMP/bin/python}"
DISPATCH="${BASH_DIR}/dispatch_prefix_tune.py"
LOG_DIR="/NAS/luyq/PLM_AMP_Regression/gnn_runs/esm2-150M/logs"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found. Run manually:" >&2
    echo "  ${PY} ${DISPATCH} --gpus 0 1 2 3 4 5 6 7 --max_jobs_per_gpu 5" >&2
    exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session '${SESSION}' already exists."
    echo "  attach: tmux attach -t ${SESSION}"
    echo "  kill:   tmux kill-session -t ${SESSION}"
    exit 1
fi

mkdir -p "${LOG_DIR}"
tmux new-session -d -s "${SESSION}" \
    "${PY} ${DISPATCH} --gpus 0 1 2 3 4 5 6 7 --max_jobs_per_gpu 5"

echo "Started prefix-tune dispatch in tmux session: ${SESSION}"
echo "  attach:  tmux attach -t ${SESSION}"
echo "  events:  ${LOG_DIR}/dispatch_prefix_tune_events.log"
echo "  per-job: ${LOG_DIR}/<VARIANT>_splits1_sd<SEED>.log"
