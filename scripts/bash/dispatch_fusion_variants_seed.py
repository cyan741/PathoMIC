#!/usr/bin/env python3
"""Dispatch Stage-3 fusion variants with seeds 7 and 123 (splits2).

7 variants x 2 seeds = 14 jobs. Seed=42 originals already exist as
``{variant}_splits2`` (no _sd42 suffix).

Launch manually on the target server, e.g. 8x A100 all at once:

    python dispatch_fusion_variants_seed.py \\
        --gpus 0 1 2 3 4 5 6 7 \\
        --max_jobs_per_gpu 2 \\
        --min_free_mb 12000

Single job smoke test:

    GPU=0 SEED=7 VARIANT=f_gated bash gnn_150M_fusion_variant.sh
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

BASH_DIR = Path("/home/luyq/PLM_AMP_Regression/scripts/bash")
SCRIPT = BASH_DIR / "gnn_150M_fusion_variant.sh"
CKP_BASE = Path("/NAS/luyq/PLM_AMP_Regression/gnn_runs/esm2-150M")
LOG_BASE = CKP_BASE / "logs"
SPLIT = "splits2"
VARIANTS = [
    "f_bilinear",
    "f_film",
    "f_gated",
    "f_xattn",
    "h3_raw192",
    "h3_prefix_pep",
    "h3_prefix_all",
]
SEEDS = [7, 123]


@dataclass
class Job:
    variant: str
    seed: int
    split: str
    proc: Optional[subprocess.Popen] = None
    gpu: Optional[int] = None
    started_at: Optional[float] = None
    returncode: Optional[int] = None

    @property
    def name(self) -> str:
        return f"{self.variant}_{self.split}_sd{self.seed}"

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
             allowed: List[int], excluded: List[int]) -> Optional[int]:
    stats = [s for s in query_gpus() if s["index"] in allowed and s["index"] not in excluded]
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
    env["VARIANT"] = job.variant
    env["SPLIT"] = job.split
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
    event_log.write(msg + "\n")
    event_log.flush()
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
        event_log.write(msg + "\n")
        event_log.flush()
        print(msg, flush=True)
        running.remove(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--gpus", nargs="+", type=int,
                    help="Allowed GPU indices (default: all visible).")
    ap.add_argument("--exclude_gpus", nargs="*", type=int, default=[])
    ap.add_argument("--min_free_mb", type=int, default=12000)
    ap.add_argument("--max_jobs_per_gpu", type=int, default=2)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--skip_epochs", type=int, default=20,
                    help="Skip jobs whose metrics csv already has this many epochs.")
    ap.add_argument("--event_log", type=str,
                    default=str(LOG_BASE / "dispatch_fusion_variants_seed_events.log"))
    args = ap.parse_args()
    split = args.split

    if args.gpus is None:
        allowed = [s["index"] for s in query_gpus()]
    else:
        allowed = args.gpus

    LOG_BASE.mkdir(parents=True, exist_ok=True)
    event_log = open(args.event_log, "a")
    event_log.write(
        f"\n[{time.strftime('%F %T')}] FUSION VARIANTS DISPATCH "
        f"split={split} variants={args.variants} seeds={args.seeds} "
        f"gpus={allowed} max_per_gpu={args.max_jobs_per_gpu}\n"
    )
    event_log.flush()

    pending: List[Job] = []
    skipped = 0
    for v in args.variants:
        for s in args.seeds:
            job = Job(variant=v, seed=s, split=split)
            if has_completed(job, args.skip_epochs):
                skipped += 1
                print(f"[SKIP] {job.name}", flush=True)
                continue
            pending.append(job)

    print(f"Total={len(args.variants)*len(args.seeds)} skipped={skipped} "
          f"pending={len(pending)}", flush=True)

    running: List[Job] = []
    try:
        while pending or running:
            poll(running, event_log)
            in_use: Dict[int, int] = {}
            for j in running:
                if j.gpu is not None:
                    in_use[j.gpu] = in_use.get(j.gpu, 0) + 1
            while pending:
                gpu = pick_gpu(in_use, args.min_free_mb, args.max_jobs_per_gpu,
                               allowed, args.exclude_gpus)
                if gpu is None:
                    break
                job = pending.pop(0)
                launch(job, gpu, event_log)
                running.append(job)
                in_use[gpu] = in_use.get(gpu, 0) + 1
                time.sleep(15)
            if not pending and not running:
                break
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)

    event_log.write(f"[{time.strftime('%F %T')}] DONE\n")
    event_log.close()


if __name__ == "__main__":
    main()
