#!/usr/bin/env python3
"""Run SafeFL Cifar10_0.5 attack-defense matrix on a GPU pool."""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/bin/python"
MAIN = ROOT / "main.py"
PROJECT = "SafeFL_Cifar10_0.5"
ACTORS_PER_GPU = 10

ATTACKS = ["vanilla", "dba", "neurotoxin", "pgd", "lga"]
DEFENSES = [
    "vanilla",
    "multikrum",
    "trimmean",
    "rfa",
    "foolsgold",
    "fltrust",
    "dnc",
    "rlr",
    "flame",
    "alignins",
]


def build_jobs() -> list[tuple[str, str]]:
    return [(attack, defense) for attack in ATTACKS for defense in DEFENSES]


def run_job(gpu_id: int, attack: str, defense: str, log_dir: Path) -> int:
    exp_name = f"{attack}_{defense}"
    cmd = [
        str(PYTHON),
        str(MAIN),
        f"experiment={exp_name}",
        f"experiment_name={exp_name}",
        f"logging.name={exp_name}",
        f"logging.project={PROJECT}",
        f"parallel.gpu_ids=[{gpu_id}]",
        f"parallel.actors_per_gpu={ACTORS_PER_GPU}",
    ]
    log_path = log_dir / f"gpu{gpu_id}_{exp_name}.log"
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"[{started}] [GPU {gpu_id}] START {exp_name}\n"
    print(header.strip(), flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(header)
        log_file.flush()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer = f"[{finished}] [GPU {gpu_id}] END {exp_name} exit={proc.returncode}\n"
        log_file.write(footer)
    status = "OK" if proc.returncode == 0 else "FAILED"
    print(f"[GPU {gpu_id}] {status} {exp_name} (exit {proc.returncode})", flush=True)
    return proc.returncode


def worker(
    gpu_id: int,
    job_queue: queue.Queue[tuple[str, str]],
    results: list[tuple[str, str, int]],
    results_lock: threading.Lock,
    log_dir: Path,
) -> None:
    while True:
        try:
            attack, defense = job_queue.get_nowait()
        except queue.Empty:
            return
        rc = run_job(gpu_id, attack, defense, log_dir)
        with results_lock:
            results.append((attack, defense, rc))
        job_queue.task_done()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="GPU ids to use, one experiment per GPU at a time",
    )
    args = parser.parse_args()

    if not PYTHON.exists():
        print(f"Missing virtualenv python: {PYTHON}", file=sys.stderr)
        return 1
    if not MAIN.exists():
        print(f"Missing entrypoint: {MAIN}", file=sys.stderr)
        return 1

    jobs = build_jobs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / "matrix_runs" / stamp
    log_dir.mkdir(parents=True, exist_ok=True)

    job_queue: queue.Queue[tuple[str, str]] = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    results: list[tuple[str, str, int]] = []
    results_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=worker,
            args=(gpu_id, job_queue, results, results_lock, log_dir),
            daemon=True,
        )
        for gpu_id in args.gpus
    ]

    print(
        f"Launching {len(jobs)} experiments on GPUs {args.gpus}, "
        f"project={PROJECT}, actors_per_gpu={ACTORS_PER_GPU}",
        flush=True,
    )
    print(f"Batch logs: {log_dir}", flush=True)

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failed = [(a, d, rc) for a, d, rc in results if rc != 0]
    summary_path = log_dir / "summary.txt"
    lines = [
        f"total={len(results)} failed={len(failed)}",
        *(f"FAILED {a}_{d} exit={rc}" for a, d, rc in failed),
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nFinished {len(results)} experiments, {len(failed)} failed.", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    for attack, defense, rc in failed:
        print(f"  FAILED: {attack}_{defense} (exit {rc})", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
