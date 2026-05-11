#!/usr/bin/env bash
# =============================================================================
# parallel_runner.sh - work-stealing runner across N GPUs.
#
# Reads a "job file" (one bash command per line) and dispatches them across
# the GPUs given in the GPUS env var (default "0,1,2,3"). Whenever a GPU's
# current job finishes, that GPU picks up the next pending job from the queue.
#
# Each job's stdout/stderr is written to the directory specified by
# the LOG_DIR env var (default ./logs). The runner persists a state file
# (job_state.txt) tracking which jobs are running/done/failed so that re-running
# the script will resume only the not-yet-done jobs (status==done is skipped).
#
# Usage
# -----
#   GPUS=0,1,2,3 LOG_DIR=/path/to/logs ./parallel_runner.sh /path/to/jobs.txt
#
# Job file format
# ---------------
# Each line is either:
#   - blank or starts with `#` -> skipped
#   - <job_id>|<bash command>
# job_id is a short tag used in log filename. The command must NOT contain `|`.
#
# Example job file:
#   h5_splits1|cd /path && python train.py --foo bar
#   h5_splits2|cd /path && python train.py --foo bar
#
# Notes
# -----
# - The command must accept a CUDA device id. The runner exports CUDA_VISIBLE_DEVICES
#   for the job; the underlying python should use --device 0 (or CUDA_VISIBLE_DEVICES).
# - One job per GPU at a time (no MPS / time-slicing).
# - Send SIGINT to the runner to stop new dispatches; running jobs are NOT killed.
# =============================================================================
set -uo pipefail

JOBS_FILE="${1:?Usage: parallel_runner.sh <jobs_file>}"
GPUS="${GPUS:-0,1,2,3}"
LOG_DIR="${LOG_DIR:-./logs}"
STATE_FILE="${STATE_FILE:-${LOG_DIR}/job_state.txt}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"

mkdir -p "${LOG_DIR}"
touch "${STATE_FILE}"

# Parse GPU list
IFS=',' read -ra GPU_LIST <<< "${GPUS}"
NUM_GPUS=${#GPU_LIST[@]}

# Read jobs
JOB_IDS=()
JOB_CMDS=()
while IFS= read -r line; do
    [[ -z "${line// }" ]] && continue          # skip blank
    [[ "${line:0:1}" == "#" ]] && continue     # skip comments
    job_id="${line%%|*}"
    job_cmd="${line#*|}"
    JOB_IDS+=("${job_id}")
    JOB_CMDS+=("${job_cmd}")
done < "${JOBS_FILE}"
NUM_JOBS=${#JOB_IDS[@]}

echo "============================================================"
echo "parallel_runner.sh"
echo "  jobs file : ${JOBS_FILE}"
echo "  GPUs      : ${GPUS} (${NUM_GPUS} GPUs)"
echo "  log dir   : ${LOG_DIR}"
echo "  state     : ${STATE_FILE}"
echo "  total jobs: ${NUM_JOBS}"
echo "============================================================"

# Initialise GPU slot tracking: gpu_pid[i] = pid of running job on GPUS[i] (0 = idle)
declare -A gpu_pid
declare -A gpu_jobid
for i in $(seq 0 $((NUM_GPUS-1))); do
    gpu_pid[$i]=0
    gpu_jobid[$i]=""
done

is_done() {
    grep -qE "^${1}\\s+done\\b" "${STATE_FILE}"
}

mark() {
    # mark <job_id> <state>
    sed -i "/^${1}\\s/d" "${STATE_FILE}"
    echo "${1} ${2} $(date +%s)" >> "${STATE_FILE}"
}

next_pending=0
done_count=0
fail_count=0
skip_count=0

while true; do
    # Reap finished jobs
    for i in $(seq 0 $((NUM_GPUS-1))); do
        pid=${gpu_pid[$i]}
        if [[ "${pid}" != "0" ]] && ! kill -0 "${pid}" 2>/dev/null; then
            wait "${pid}" 2>/dev/null
            rc=$?
            jid="${gpu_jobid[$i]}"
            if [[ "${rc}" == "0" ]]; then
                mark "${jid}" "done"
                done_count=$((done_count+1))
                echo "  [done ] gpu${GPU_LIST[$i]} ${jid} (rc=${rc})"
            else
                mark "${jid}" "failed"
                fail_count=$((fail_count+1))
                echo "  [FAIL ] gpu${GPU_LIST[$i]} ${jid} (rc=${rc})"
            fi
            gpu_pid[$i]=0
            gpu_jobid[$i]=""
        fi
    done

    # Dispatch new jobs to idle GPUs
    for i in $(seq 0 $((NUM_GPUS-1))); do
        if [[ "${gpu_pid[$i]}" == "0" ]] && [[ "${next_pending}" -lt "${NUM_JOBS}" ]]; then
            jid="${JOB_IDS[$next_pending]}"
            cmd="${JOB_CMDS[$next_pending]}"
            next_pending=$((next_pending+1))

            if is_done "${jid}"; then
                echo "  [skip ] ${jid} already done"
                skip_count=$((skip_count+1))
                continue
            fi

            mark "${jid}" "running"
            gid="${GPU_LIST[$i]}"
            log="${LOG_DIR}/${jid}.log"
            echo "  [start] gpu${gid} ${jid}  -> ${log}"
            (
                export CUDA_VISIBLE_DEVICES="${gid}"
                bash -c "${cmd}" > "${log}" 2>&1
            ) &
            gpu_pid[$i]=$!
            gpu_jobid[$i]="${jid}"
        fi
    done

    # Done?
    all_idle=1
    for i in $(seq 0 $((NUM_GPUS-1))); do
        if [[ "${gpu_pid[$i]}" != "0" ]]; then all_idle=0; break; fi
    done
    if [[ "${all_idle}" == "1" ]] && [[ "${next_pending}" -ge "${NUM_JOBS}" ]]; then
        break
    fi
    sleep "${POLL_INTERVAL}"
done

echo "============================================================"
echo "all done. ok=${done_count}  fail=${fail_count}  skipped=${skip_count}"
echo "============================================================"
