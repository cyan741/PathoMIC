#!/usr/bin/env python3
"""Dispatcher for 5-seed variance study.

Total jobs = 2 variants x 2 splits x 5 seeds = 20.

Reuses the GPU-aware scheduling logic from dispatch_stage3.py (poll nvidia-smi,
launch one bash invocation per (variant, split, seed) on an idle GPU,
respect --exclude_gpus / --min_free_mb / --max_jobs_per_gpu).
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


BASH_DIR    = Path("/home/luyq/PLM_AMP_Regression/scripts/bash")
SCRIPT      = BASH_DIR / "gnn_150M_seed_var.sh"
CKP_BASE    = Path("/NAS/luyq/PLM_AMP_Regression/gnn_runs/esm2-150M")
LOG_BASE    = CKP_BASE / "logs"

VARIANTS = ["v15", "vanilla"]
SPLITS   = ["splits1", "splits2"]
SEEDS    = [42, 1337, 2025, 7, 123]


@dataclass
class Job:
    variant: str
    split:   str
    seed:    int
    pid:     Optional[int] = None
    proc:    Optional[subprocess.Popen] = None
    gpu:     Optional[int] = None
    started_at:  Optional[float] = None
    finished_at: Optional[float] = None
    returncode:  Optional[int] = None

    @property
    def name(self) -> str:
        return f"{self.variant}_seed_{self.split}_sd{self.seed}"

    @property
    def metrics_csv(self) -> Path:
        return CKP_BASE / self.name / f"{self.name}.csv"


def query_gpus() -> List[Dict[str, int]]:
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=index,memory.free,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"]
    ).decode()
    res = []
    for line in out.strip().splitlines():
        idx, free, used, util = [int(x.strip()) for x in line.split(",")]
        res.append({"index": idx, "free": free, "used": used, "util": util})
    return res


def pick_gpu(in_use: Dict[int, int], min_free_mb: int,
             max_util: int, max_jobs_per_gpu: int,
             excluded: List[int]) -> Optional[int]:
    stats = query_gpus()
    stats.sort(key=lambda s: (in_use.get(s["index"], 0), -s["free"]))
    for s in stats:
        idx = s["index"]
        if idx in excluded:                       continue
        if in_use.get(idx, 0) >= max_jobs_per_gpu: continue
        if s["free"] < min_free_mb:                continue
        # ignore util threshold when this GPU is already running our own job
        if in_use.get(idx, 0) == 0 and s["util"] > max_util: continue
        return idx
    return None


def has_completed(job: Job, min_epochs: int) -> bool:
    p = job.metrics_csv
    if not p.exists():
        return False
    try:
        with open(p) as f:
            rows = list(csv.DictReader(f))
        return len(rows) >= min_epochs
    except Exception:
        return False


def launch_job(job: Job, gpu: int, event_log) -> subprocess.Popen:
    env = os.environ.copy()
    env["GPU"]     = str(gpu)
    env["VARIANT"] = job.variant
    env["SPLIT"]   = job.split
    env["SEED"]    = str(job.seed)
    bash_log = LOG_BASE / f"{job.name}.bash.log"
    bash_log.parent.mkdir(parents=True, exist_ok=True)
    f = open(bash_log, "w")
    proc = subprocess.Popen(
        ["bash", str(SCRIPT)],
        env=env, cwd=str(SCRIPT.parent),
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    job.pid = proc.pid
    job.proc = proc
    job.gpu = gpu
    job.started_at = time.time()
    msg = (f"[{time.strftime('%F %T')}] LAUNCH {job.name} "
           f"on GPU{gpu} pid={proc.pid}")
    event_log.write(msg + "\n"); event_log.flush()
    print(msg, flush=True)
    return proc


def poll_running(running: List[Job], event_log) -> List[Job]:
    finished = []
    for j in list(running):
        if j.proc is None: continue
        rc = j.proc.poll()
        if rc is None: continue
        j.returncode = rc
        j.finished_at = time.time()
        elapsed = (j.finished_at or 0) - (j.started_at or 0)
        msg = (f"[{time.strftime('%F %T')}] FINISH {j.name} "
               f"on GPU{j.gpu} pid={j.pid} rc={rc} "
               f"elapsed={elapsed/60:.1f}min")
        event_log.write(msg + "\n"); event_log.flush()
        print(msg, flush=True)
        running.remove(j)
        finished.append(j)
    return finished


def build_jobs(variants: List[str], splits: List[str],
               seeds: List[int]) -> List[Job]:
    jobs = []
    for v in variants:
        for s in splits:
            for sd in seeds:
                jobs.append(Job(variant=v, split=s, seed=sd))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--splits",   nargs="+", default=SPLITS,   choices=SPLITS)
    ap.add_argument("--seeds",    nargs="+", type=int, default=SEEDS)
    ap.add_argument("--exclude_gpus", nargs="*", type=int, default=[])
    ap.add_argument("--min_free_mb",  type=int, default=12000)
    ap.add_argument("--max_util",     type=int, default=100)
    ap.add_argument("--max_jobs_per_gpu", type=int, default=1)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--skip_epochs", type=int, default=20)
    ap.add_argument("--max_attempts", type=int, default=2)
    ap.add_argument("--event_log", type=str,
                    default="/NAS/luyq/PLM_AMP_Regression/gnn_runs/esm2-150M/logs/dispatch_seed_events.log")
    args = ap.parse_args()

    LOG_BASE.mkdir(parents=True, exist_ok=True)
    event_log = open(args.event_log, "a")
    event_log.write(
        f"\n[{time.strftime('%F %T')}] DISPATCH START "
        f"variants={args.variants} splits={args.splits} "
        f"seeds={args.seeds} max_per_gpu={args.max_jobs_per_gpu}\n"
    )
    event_log.flush()

    jobs = build_jobs(args.variants, args.splits, args.seeds)
    print(f"Total jobs: {len(jobs)}", flush=True)

    pending, skipped = [], 0
    for j in jobs:
        if has_completed(j, args.skip_epochs):
            skipped += 1
            print(f"[SKIP] {j.name} (already has >= {args.skip_epochs} rows)",
                  flush=True)
            continue
        pending.append(j)
    print(f"Skipped: {skipped} | To run: {len(pending)}", flush=True)

    running: List[Job] = []
    attempts: Dict[str, int] = {}

    try:
        while pending or running:
            poll_running(running, event_log)
            # requeue failed jobs (rc != 0) up to --max_attempts
            for j in list(running):
                # poll_running already removed finished jobs above; nothing here
                pass

            in_use: Dict[int, int] = {}
            for j in running:
                if j.gpu is not None:
                    in_use[j.gpu] = in_use.get(j.gpu, 0) + 1

            while pending:
                gpu = pick_gpu(in_use, args.min_free_mb, args.max_util,
                               args.max_jobs_per_gpu, args.exclude_gpus)
                if gpu is None: break
                job = pending.pop(0)
                attempts.setdefault(job.name, 1)
                launch_job(job, gpu, event_log)
                running.append(job)
                in_use[gpu] = in_use.get(gpu, 0) + 1
                # let nvidia-smi catch up before next pick
                time.sleep(15)

            if not pending and not running:
                break
            time.sleep(args.poll)
    except KeyboardInterrupt:
        msg = (f"[{time.strftime('%F %T')}] INTERRUPTED "
               f"-- {len(running)} jobs left running")
        event_log.write(msg + "\n")
        print(msg, file=sys.stderr, flush=True)

    msg = f"[{time.strftime('%F %T')}] DISPATCH DONE"
    event_log.write(msg + "\n"); event_log.flush()
    print(msg, flush=True)
    event_log.close()


if __name__ == "__main__":
    main()
