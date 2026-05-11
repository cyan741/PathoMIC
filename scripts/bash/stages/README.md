# GNN-Only Optimal Search — execution guide

This directory holds the per-stage job lists, the parallel runner, and the
orchestration helpers for the GNN-only-optimal experimental plan
(`/root/PLM_AMP_Regression/gnn-only-optimal.md`, 92 runs total).

## Hardware

- 4× NVIDIA A100-80GB on a single node (GPUs 0–3).
- ~8 GiB / GPU at bs=32 for ESM2-150M.
- One run takes 2.5–3 h to early-stop convergence.
- All 4 GPUs run in parallel via `parallel_runner.sh` (work-stealing queue).

## Files

| File | Purpose |
|---|---|
| `../parallel_runner.sh`     | Generic 4-GPU work-stealing job runner |
| `stage1_jobs.txt`           | Stage 1 jobs (already running) |
| `gen_jobs.py`               | Generate `stage{N}_jobs.txt` from previous winners |
| `select_winner.py`          | Pick a stage's winner by mean test MSE |
| `aggregate_summary.py`      | Collect all stages into `summary.csv` |
| `orchestrate.sh`            | One-command "select prev, launch next" helper |

## Quick reference

After Stage 1 finishes (check `${V2_ROOT}/stage1/logs/job_state.txt` shows
`done` for all 6 entries), run **one command per stage**:

```bash
cd /root/PLM_AMP_Regression/scripts/bash/stages

# After Stage 1 completes:
./orchestrate.sh next 2     # select Stage 1 winner -> launch Stage 2
# wait until Stage 2 done...
./orchestrate.sh next 3     # select Stage 2 winner -> launch Stage 3
# wait...
./orchestrate.sh next 4a
./orchestrate.sh next 4b
./orchestrate.sh next 4c
./orchestrate.sh next 4d
./orchestrate.sh next 4e
./orchestrate.sh next 4f
./orchestrate.sh next 4g
./orchestrate.sh next 4h
./orchestrate.sh next 5     # final candidate + 3 controls
```

`./orchestrate.sh next <id>` is the canonical move:
1. Calls `select_winner.py` on the previous stage's results, writes
   `${V2_ROOT}/stage${prev}/winner.json`.
2. Calls `gen_jobs.py` to materialize `stage${id}_jobs.txt` with the winner's
   hyperparameters baked in.
3. Launches `parallel_runner.sh` in the background on 4 GPUs.

## Live monitoring

```bash
# overall progress
tail -f ${V2_ROOT}/stage<N>/runner.log

# GPU utilisation (refresh every 30 s)
watch -n 30 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv"

# job state machine
cat ${V2_ROOT}/stage<N>/logs/job_state.txt

# is the runner alive?
pgrep -af parallel_runner.sh
```

## How "wait until done" looks

The shell command after `next` returns immediately (runner is in background).
Wait until `runner.log` prints `all done. ok=N fail=0 skipped=0`:

```bash
# Block until current stage finishes
while pgrep -af "parallel_runner.sh.*stage${N}_jobs.txt" >/dev/null; do
    sleep 60
done && echo "Stage ${N} done."
```

Or just check `job_state.txt` periodically.

## When to deviate from the auto-pipeline

`select_winner.py` ranks by **mean test MSE** with bucketed-MSE tie-breakers
(see `gnn-only-optimal.md`). If you want a different rule (e.g. weight
splits1 OOD harder), edit `select_winner.py` or write `winner.json` by hand
before calling `./orchestrate.sh launch <id>`.

The Stage 5 control-comparison setup (`5_ctrl_none`, `5_ctrl_adapter`,
`5_ctrl_both`) requires you to confirm the final winner CLI fragment first —
inspect `stage4h/winner.json`, then check `gen_jobs.py:stage5_jobs()` and
ensure the chained `--gnn_*` / `--loss_*` / `--lr` / `--batch_size` settings
match the actual best configuration.

## Stage runtime estimates

| Stage | # runs | Wall (4 GPUs) |
|---|---|---|
| 1 | 6  | ~5 h  |
| 2 | 6  | ~5 h  |
| 3 | 12 | ~9 h  |
| 4a | 18 | ~14 h |
| 4b | 6  | ~5 h  |
| 4c | 12 | ~10 h |
| 4d | 8  | ~6 h  |
| 4e | 4  | ~3 h  |
| 4f | 6  | ~5 h  |
| 4g | 2  | ~2 h  |
| 4h | 4  | ~3 h  |
| 5 | 12 | ~10 h |
| **Total** | **96** | **~77 h** |

(Stage 6 inference + plots ~30 min.)

## Stage 6: post-hoc inference + plots

After Stage 5 finishes:

```bash
# Aggregate all stages into a single CSV
${PY} ${STAGES_DIR}/aggregate_summary.py --root ${V2_ROOT}

# Run inference on every stage winner's val_best.pth (extends run_all_gnn_infer.sh)
# (TODO: update run_all_gnn_infer.sh to discover val_best paths from V2_ROOT)
```
