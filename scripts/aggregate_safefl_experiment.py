#!/usr/bin/env python3
"""Aggregate last-N-round mean Acc/ASR into SafeFL_experiment.csv.

Cell format: "{acc:.2f}/{asr:.2f}" as percentages without the % sign.
Rows = attack methods, columns = defense methods.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_ATTACKS = ["vanilla", "chameleon", "neurotoxin", "pgd"]
DEFAULT_DEFENSES = ["vanilla"]


def _parse_run_name(name: str) -> Optional[Tuple[str, str]]:
    """Parse '{attack}_{defense}' run name."""
    if "_" not in name:
        return None
    attack, defense = name.rsplit("_", 1)
    return attack, defense


def _mean_last_n(values: List[float], n: int) -> Optional[float]:
    if not values:
        return None
    tail = values[-n:] if len(values) >= n else values
    return sum(tail) / len(tail)


def _read_metrics_csv(path: Path, last_n: int) -> Optional[Tuple[float, float]]:
    accs: List[float] = []
    asrs: List[float] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            acc = row.get("global/accuracy") or row.get("accuracy")
            asr = row.get("global/asr") or row.get("asr")
            if acc is None or asr is None or acc == "" or asr == "":
                continue
            accs.append(float(acc))
            asrs.append(float(asr))
    mean_acc = _mean_last_n(accs, last_n)
    mean_asr = _mean_last_n(asrs, last_n)
    if mean_acc is None or mean_asr is None:
        return None
    return mean_acc * 100.0, mean_asr * 100.0


def _read_run_log(path: Path, last_n: int) -> Optional[Tuple[float, float]]:
    accs: List[float] = []
    asrs: List[float] = []
    pat = re.compile(
        r"accuracy:\s*([0-9.]+).*?asr:\s*([0-9.]+)",
        re.IGNORECASE,
    )
    for line in path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            accs.append(float(m.group(1)))
            asrs.append(float(m.group(2)))
    mean_acc = _mean_last_n(accs, last_n)
    mean_asr = _mean_last_n(asrs, last_n)
    if mean_acc is None or mean_asr is None:
        return None
    # run.log values are already ratios in [0, 1]
    return mean_acc * 100.0, mean_asr * 100.0


def _collect_from_logs(
    logs_root: Path,
    project: str,
    last_n: int,
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    results: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for metrics_path in logs_root.rglob("metrics.csv"):
        # Prefer run dirs under the target project name.
        parts = metrics_path.parts
        if project not in parts:
            continue
        run_dir = metrics_path.parent
        # strip trailing timestamp: name_YYYYMMDD_HHMMSS
        base = re.sub(r"_\d{8}_\d{6}$", "", run_dir.name)
        parsed = _parse_run_name(base)
        if parsed is None:
            continue
        parsed_metrics = _read_metrics_csv(metrics_path, last_n)
        if parsed_metrics is None:
            continue
        results[parsed] = parsed_metrics
    return results


def _collect_from_wandb(
    project: str,
    entity: Optional[str],
    last_n: int,
    attacks: Iterable[str],
    defenses: Iterable[str],
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    results: Dict[Tuple[str, str], Tuple[float, float]] = {}
    wanted = {(a, d) for a in attacks for d in defenses}
    for run in api.runs(path):
        name = run.name or ""
        parsed = _parse_run_name(name)
        if parsed is None or parsed not in wanted:
            continue
        hist = run.history(keys=["global/accuracy", "global/asr"], pandas=False)
        accs = [float(r["global/accuracy"]) for r in hist if "global/accuracy" in r]
        asrs = [float(r["global/asr"]) for r in hist if "global/asr" in r]
        mean_acc = _mean_last_n(accs, last_n)
        mean_asr = _mean_last_n(asrs, last_n)
        if mean_acc is None or mean_asr is None:
            continue
        results[parsed] = (mean_acc * 100.0, mean_asr * 100.0)
    return results


def write_table(
    results: Dict[Tuple[str, str], Tuple[float, float]],
    attacks: List[str],
    defenses: List[str],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attack\\defense"] + defenses)
        for attack in attacks:
            row = [attack]
            for defense in defenses:
                cell = results.get((attack, defense))
                if cell is None:
                    row.append("")
                else:
                    acc, asr = cell
                    row.append(f"{acc:.2f}/{asr:.2f}")
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="SafeFL_Cifar10")
    parser.add_argument("--entity", default=None, help="wandb entity (optional)")
    parser.add_argument("--logs-root", type=Path, default=Path("logs"))
    parser.add_argument("--last-n", type=int, default=10)
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=DEFAULT_ATTACKS,
    )
    parser.add_argument(
        "--defenses",
        nargs="+",
        default=DEFAULT_DEFENSES,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("SafeFL_experiment.csv"),
    )
    parser.add_argument(
        "--source",
        choices=["auto", "logs", "wandb"],
        default="auto",
    )
    args = parser.parse_args()

    results: Dict[Tuple[str, str], Tuple[float, float]] = {}
    if args.source in ("auto", "logs"):
        results.update(
            _collect_from_logs(args.logs_root, args.project, args.last_n)
        )
    if args.source in ("auto", "wandb"):
        missing = [
            (a, d)
            for a in args.attacks
            for d in args.defenses
            if (a, d) not in results
        ]
        if missing or args.source == "wandb":
            try:
                wb = _collect_from_wandb(
                    args.project,
                    args.entity,
                    args.last_n,
                    args.attacks,
                    args.defenses,
                )
                results.update(wb)
            except Exception as exc:  # noqa: BLE001
                if args.source == "wandb":
                    raise
                print(f"[warn] wandb fetch skipped: {exc}")

    write_table(results, args.attacks, args.defenses, args.out)
    print(f"Wrote {args.out} with {len(results)} cells filled.")
    for (a, d), (acc, asr) in sorted(results.items()):
        print(f"  {a}_{d}: acc={acc:.2f} asr={asr:.2f}")


if __name__ == "__main__":
    main()
