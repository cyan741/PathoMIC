#!/usr/bin/env python3
"""Dispatch shuffle-species control runs (splits1, 5 seeds)."""
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

BASH_DIR = Path("/home/luyq/PLM_AMP_Regression/scripts/bash")
SCRIPT = BASH_DIR / "gnn_150M_shuffle_species.sh"
CKP_BASE = Path("/NAS/luyq/PLM_AMP_Regression/gnn_runs/esm2-150M")
LOG_BASE = CKP_BASE / "logs"
SPLIT = "splits1"
SEEDS = [42, 1337, 2025, 7, 123]


@dataclass
class Job:
    seed: int
    proc: Optional[subprocess.Popen] = None
    gpu: Optional[int] = None
    started_at: Optional[float] = None
    returncode: Optional[int] = None

    @property
    def name(self) -> str:
        return f"shuffle_species_{SPLIT}_sd{self.seed}"

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


def pick_gpu(in_use: Dict[int, int], min_free_mb: int, max_jobs_per_gpu: int,
             allowed: List[int]) -> Optional[int]:
    stats = [s for s in query_gpus() if s["index"] in allowed]
    stats.sort(key=lambda s: (in_use.get(s["index"], 0), -s["free"]))
    for s in stats:
        idx = s["index"]
        if in_use.get(idx, 0) >= max_jobs_per_gpu:
            continue
        if s["free"] < min_free_mb:
            continue
        return idx
    return None


def has_completed(job: Job, min_epochs: int) -> bool:
    p = job.metrics_csv
    if not p.exists():
        return False
    try:
        with open(p) as f:
            return len(list(csv.DictReader(f))) >= min_epochs
    except Exception:
        return False


def launch(job: Job, gpu: int, event_log) -> None:
    env = os.environ.copy()
    env["GPU"] = str(gpu)
    env["SEED"] = str(job.seed)
    bash_log = LOG_BASE / f"{job.name}.bash.log"
    bash_log.parent.mkdir(parents=True, exist_ok=True)
    f = open(bash_log, "w")
    proc = subprocess.Popen(
        ["bash", str(SCRIPT)], env=env, cwd=str(SCRIPT.parent),
        stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
    )
    job.proc = proc
    job.gpu = gpu
    job.started_at = time.time()
    msg = f"[{time.strftime('%F %T')}] LAUNCH {job.name} GPU{gpu} pid={proc.pid}"
    event_log.write(msg + "\n"); event_log.flush()
    print(msg, flush=True)


def poll(running: List[Job], event_log) -> None:
    for j in list(running):
        if j.proc is None:
            continue
        rc = j.proc.poll()
        if rc is None:
            continue
        j.returncode = rc
        elapsed = time.time() - (j.started_at or time.time())
        msg = (f"[{time.strftime('%F %T')}] FINISH {j.name} GPU{j.gpu} "
               f"rc={rc} elapsed={elapsed/60:.1f}min")
        event_log.write(msg + "\n"); event_log.flush()
        print(msg, flush=True)
        running.remove(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="+", type=int, default=[1, 6, 7])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--min_free_mb", type=int, default=12000)
    ap.add_argument("--max_jobs_per_gpu", type=int, default=2)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--skip_epochs", type=int, default=20)
    ap.add_argument("--event_log", type=str,
                    default=str(LOG_BASE / "dispatch_shuffle_events.log"))
    args = ap.parse_args()

    LOG_BASE.mkdir(parents=True, exist_ok=True)
    event_log = open(args.event_log, "a")
    event_log.write(f"\n[{time.strftime('%F %T')}] SHUFFLE DISPATCH gpus={args.gpus}\n")
    event_log.flush()

    pending = [Job(seed=s) for s in args.seeds if not has_completed(Job(s), args.skip_epochs)]
    print(f"To run: {len(pending)} / {len(args.seeds)}", flush=True)
    running: List[Job] = []

    try:
        while pending or running:
            poll(running, event_log)
            in_use: Dict[int, int] = {}
            for j in running:
                if j.gpu is not None:
                    in_use[j.gpu] = in_use.get(j.gpu, 0) + 1
            while pending:
                gpu = pick_gpu(in_use, args.min_free_mb, args.max_jobs_per_gpu, args.gpus)
                if gpu is None:
                    break
                job = pending.pop(0)
                launch(job, gpu, event_log)
                running.append(job)
                in_use[gpu] = in_use.get(gpu, 0) + 1
                time.sleep(20)
            if not pending and not running:
                break
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)

    event_log.write(f"[{time.strftime('%F %T')}] DONE\n")
    event_log.close()


if __name__ == "__main__":
    main()
