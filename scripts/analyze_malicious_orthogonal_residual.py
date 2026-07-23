#!/usr/bin/env python3
"""Analyze BadNets dual delta logs with residual-retention x history-misalignment score.

For each logged client snapshot:
  Δm = Wb - Wc  (wb_minus_wc.pt)
  Δc = Wc - Wg  (wc_minus_wg.pt)
  Δg = Wg - Wg1 (wg_minus_wg1.pt)

Malicious orthogonal residual (per weight tensor / "layer"):
  R = Δm - proj_Δc(Δm)

Per layer (top_ratio by |R|, default 5%):
  s = (||R|| / ||Δm||) * (1 - cos(R_top, Δg_top)) / 2

Layers follow reference axis order: all *.weight tensors in network depth order
(conv1 -> ... -> linear), matching the provided scatter figure's y-axis laid as x-axis.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass(frozen=True)
class LayerScore:
    layer_index: int
    layer_name: str
    score: float
    resid_ratio: float
    misalign: float
    cos_val: float
    num_selected: int
    numel: int


def _is_axis_weight_layer(name: str, tensor: torch.Tensor) -> bool:
    """Match reference figure axis: trainable *.weight only (incl. BN / linear)."""
    if not name.endswith(".weight"):
        return False
    if not torch.is_floating_point(tensor):
        return False
    return tensor.numel() > 0


def _iter_layer_names(state: Dict[str, torch.Tensor]) -> List[str]:
    return [k for k, v in state.items() if _is_axis_weight_layer(k, v)]


def orthogonal_residual(
    delta_m: torch.Tensor,
    delta_c: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """R = Δm - proj_Δc(Δm), flattened vector projection in R^n."""
    dm = delta_m.detach().float().reshape(-1)
    dc = delta_c.detach().float().reshape(-1)
    denom = float(torch.dot(dc, dc).item())
    if denom <= eps:
        return dm
    coeff = float(torch.dot(dm, dc).item()) / denom
    return dm - coeff * dc


def top_magnitude_indices(values: torch.Tensor, top_ratio: float) -> torch.Tensor:
    flat = values.reshape(-1)
    n = int(flat.numel())
    if n == 0:
        return torch.empty(0, dtype=torch.long)
    k = max(1, int(np.ceil(n * float(top_ratio))))
    k = min(k, n)
    return torch.topk(flat.abs(), k=k, largest=True, sorted=False).indices


def layer_formula_score(
    residual: torch.Tensor,
    delta_m: torch.Tensor,
    delta_g: torch.Tensor,
    top_ratio: float,
    eps: float = 1e-12,
) -> Tuple[float, float, float, float, int]:
    """Return (score, resid_ratio, misalign, cos, num_selected)."""
    r = residual.reshape(-1)
    dm = delta_m.detach().float().reshape(-1)
    dg = delta_g.detach().float().reshape(-1)

    dm_norm = float(torch.linalg.vector_norm(dm).item())
    r_norm = float(torch.linalg.vector_norm(r).item())
    if dm_norm <= eps:
        resid_ratio = float("nan")
    else:
        resid_ratio = r_norm / dm_norm

    idx = top_magnitude_indices(r, top_ratio)
    n_sel = int(idx.numel())
    if n_sel == 0 or not np.isfinite(resid_ratio):
        return float("nan"), resid_ratio, float("nan"), float("nan"), n_sel

    r_top = r[idx]
    g_top = dg[idx]
    r_top_norm = float(torch.linalg.vector_norm(r_top).item())
    g_top_norm = float(torch.linalg.vector_norm(g_top).item())
    if r_top_norm <= eps or g_top_norm <= eps:
        cos_val = 0.0
    else:
        cos_val = float(torch.dot(r_top, g_top).item()) / (r_top_norm * g_top_norm)
        cos_val = float(max(-1.0, min(1.0, cos_val)))

    # (1 - cos) / 2 in [0, 1]: same direction -> 0, opposite -> 1
    misalign = 0.5 * (1.0 - cos_val)
    score = float(resid_ratio * misalign)
    return score, float(resid_ratio), float(misalign), float(cos_val), n_sel


def score_client_snapshot(
    malicious: Dict[str, torch.Tensor],
    benign: Dict[str, torch.Tensor],
    history: Dict[str, torch.Tensor],
    top_ratio: float,
    eps: float = 1e-12,
) -> List[LayerScore]:
    layer_names = _iter_layer_names(malicious)
    scores: List[LayerScore] = []
    for layer_index, name in enumerate(layer_names):
        if name not in benign or name not in history:
            continue
        dm = malicious[name]
        dc = benign[name]
        dg = history[name]
        if dm.shape != dc.shape or dm.shape != dg.shape:
            continue
        residual = orthogonal_residual(dm, dc, eps=eps)
        score, resid_ratio, misalign, cos_val, n_sel = layer_formula_score(
            residual=residual,
            delta_m=dm,
            delta_g=dg,
            top_ratio=top_ratio,
            eps=eps,
        )
        scores.append(
            LayerScore(
                layer_index=layer_index,
                layer_name=name,
                score=score,
                resid_ratio=resid_ratio,
                misalign=misalign,
                cos_val=cos_val,
                num_selected=n_sel,
                numel=int(dm.numel()),
            )
        )
    return scores


def discover_client_dirs(delta_root: Path) -> List[Path]:
    pattern = re.compile(r"^round_(\d+)$")
    client_dirs: List[Path] = []
    if not delta_root.is_dir():
        return client_dirs
    for round_dir in sorted(delta_root.iterdir()):
        if not round_dir.is_dir() or not pattern.match(round_dir.name):
            continue
        for client_dir in sorted(round_dir.iterdir()):
            if not client_dir.is_dir():
                continue
            required = (
                client_dir / "wb_minus_wc.pt",
                client_dir / "wc_minus_wg.pt",
                client_dir / "wg_minus_wg1.pt",
            )
            if all(p.is_file() for p in required):
                client_dirs.append(client_dir)
    return client_dirs


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected state dict in {path}, got {type(obj)}")
    return obj


def parse_round_client(client_dir: Path) -> Tuple[int, str]:
    round_name = client_dir.parent.name
    m = re.match(r"^round_(\d+)$", round_name)
    round_idx = int(m.group(1)) if m else -1
    return round_idx, client_dir.name


def average_scores_by_layer(
    score_lists: Sequence[Sequence[LayerScore]],
) -> List[LayerScore]:
    if not score_lists:
        return []
    ref = list(score_lists[0])
    out: List[LayerScore] = []
    for i, ref_item in enumerate(ref):
        score_vals: List[float] = []
        resid_vals: List[float] = []
        mis_vals: List[float] = []
        cos_vals: List[float] = []
        n_sel: List[int] = []
        for scores in score_lists:
            if i >= len(scores):
                continue
            item = scores[i]
            if item.layer_name != ref_item.layer_name:
                continue
            if np.isfinite(item.score):
                score_vals.append(item.score)
            if np.isfinite(item.resid_ratio):
                resid_vals.append(item.resid_ratio)
            if np.isfinite(item.misalign):
                mis_vals.append(item.misalign)
            if np.isfinite(item.cos_val):
                cos_vals.append(item.cos_val)
            n_sel.append(item.num_selected)
        out.append(
            LayerScore(
                layer_index=ref_item.layer_index,
                layer_name=ref_item.layer_name,
                score=float(np.mean(score_vals)) if score_vals else float("nan"),
                resid_ratio=float(np.mean(resid_vals)) if resid_vals else float("nan"),
                misalign=float(np.mean(mis_vals)) if mis_vals else float("nan"),
                cos_val=float(np.mean(cos_vals)) if cos_vals else float("nan"),
                num_selected=int(np.mean(n_sel)) if n_sel else 0,
                numel=ref_item.numel,
            )
        )
    return out


def plot_layer_scores_by_depth(
    scores: Sequence[LayerScore],
    title: str,
    out_path: Path,
) -> None:
    # Reference figure: shallow at bottom -> deep at top.
    # Horizontal plot: shallow at left -> deep at right (same depth order).
    ordered = sorted(
        [s for s in scores if np.isfinite(s.score)],
        key=lambda s: s.layer_index,
    )
    if not ordered:
        raise RuntimeError(f"No finite scores to plot for: {title}")

    xs = np.arange(len(ordered))
    ys = [s.score for s in ordered]
    labels = [s.layer_name for s in ordered]

    fig_w = max(14.0, 0.42 * len(ordered))
    fig, ax = plt.subplots(figsize=(fig_w, 5.8))
    ax.scatter(xs, ys, s=28, color="#1f77b4", zorder=3)
    ax.plot(xs, ys, color="#1f77b4", alpha=0.35, linewidth=1.0, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("layer name (reference axis, network depth order)")
    ax.set_ylabel(r"$\|R\|/\|\Delta m\|\cdot(1-\cos(R_{top},\Delta g_{top}))/2$")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_rounds_panel(
    round_to_scores: Dict[int, Sequence[LayerScore]],
    title: str,
    out_path: Path,
) -> None:
    """One subplot per round; each curve is the 4-client layer mean."""
    rounds = sorted(round_to_scores)
    if not rounds:
        raise RuntimeError("No rounds to plot in panel.")

    n = len(rounds)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows), sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, round_idx in zip(axes_flat, rounds):
        ordered = sorted(
            [s for s in round_to_scores[round_idx] if np.isfinite(s.score)],
            key=lambda s: s.layer_index,
        )
        xs = np.arange(len(ordered))
        ys = [s.score for s in ordered]
        ax.scatter(xs, ys, s=14, color="#1f77b4", zorder=3)
        ax.plot(xs, ys, color="#1f77b4", alpha=0.35, linewidth=1.0, zorder=2)
        ax.set_title(f"round {round_idx} (4-client mean)")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if len(ordered) <= 20:
            ax.set_xticks(xs)
            ax.set_xticklabels([s.layer_name for s in ordered], rotation=90, fontsize=5)
        else:
            # Too dense for full labels in panel; show endpoints + stride.
            step = max(1, len(ordered) // 8)
            tick_idx = list(range(0, len(ordered), step))
            if tick_idx[-1] != len(ordered) - 1:
                tick_idx.append(len(ordered) - 1)
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([ordered[i].layer_name for i in tick_idx], rotation=90, fontsize=5)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_scores_csv(scores: Sequence[LayerScore], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        [s for s in scores if np.isfinite(s.score)],
        key=lambda s: s.layer_index,
    )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "layer_index",
                "layer_name",
                "score",
                "resid_ratio",
                "misalign",
                "cos",
                "num_selected",
                "numel",
            ]
        )
        for item in ordered:
            writer.writerow(
                [
                    item.layer_index,
                    item.layer_name,
                    f"{item.score:.8g}",
                    f"{item.resid_ratio:.8g}",
                    f"{item.misalign:.8g}",
                    f"{item.cos_val:.8g}",
                    item.num_selected,
                    item.numel,
                ]
            )


def analyze_one_client(
    client_dir: Path,
    top_ratio: float,
    eps: float,
) -> List[LayerScore]:
    malicious = load_state(client_dir / "wb_minus_wc.pt")
    benign = load_state(client_dir / "wc_minus_wg.pt")
    history = load_state(client_dir / "wg_minus_wg1.pt")
    if not history:
        raise RuntimeError(f"Empty history delta: {client_dir / 'wg_minus_wg1.pt'}")
    return score_client_snapshot(
        malicious=malicious,
        benign=benign,
        history=history,
        top_ratio=top_ratio,
        eps=eps,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze residual-retention x history-misalignment scores from delta_logs."
    )
    parser.add_argument(
        "--delta-root",
        type=str,
        default="logs/2026-07-23/badnets_dual_18-19-16/delta_logs",
        help="Path to delta_logs root (contains round_*/client_*/).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: <delta-root>/../orthogonal_residual_analysis",
    )
    parser.add_argument("--top-ratio", type=float, default=0.05, help="Top ratio by |R|.")
    parser.add_argument("--eps", type=float, default=1e-12, help="Projection denom epsilon.")
    parser.add_argument(
        "--per-client",
        action="store_true",
        help="Also emit per-client plots/CSVs (default: only round-mean + overall-mean).",
    )
    parser.add_argument(
        "--expected-clients",
        type=int,
        default=4,
        help="Expected malicious clients per round for averaging (default: 4).",
    )
    parser.add_argument(
        "--rounds",
        type=str,
        default=None,
        help="Comma-separated round ids to include, e.g. 15,30. Default: all.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if not (0.0 < float(args.top_ratio) <= 1.0):
        raise ValueError("--top-ratio must be in (0, 1].")
    if int(args.expected_clients) <= 0:
        raise ValueError("--expected-clients must be positive.")

    delta_root = Path(args.delta_root).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (delta_root.parent / "orthogonal_residual_analysis").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    round_filter: Optional[set[int]] = None
    if args.rounds:
        round_filter = {int(x.strip()) for x in args.rounds.split(",") if x.strip()}

    client_dirs = discover_client_dirs(delta_root)
    if round_filter is not None:
        filtered = []
        for d in client_dirs:
            round_idx, _ = parse_round_client(d)
            if round_idx in round_filter:
                filtered.append(d)
        client_dirs = filtered

    if not client_dirs:
        raise FileNotFoundError(f"No client delta dirs found under: {delta_root}")

    by_round_scores: Dict[int, List[List[LayerScore]]] = {}
    by_round_clients: Dict[int, List[str]] = {}
    meta_rows = []

    for client_dir in client_dirs:
        round_idx, client_name = parse_round_client(client_dir)
        scores = analyze_one_client(client_dir, top_ratio=args.top_ratio, eps=args.eps)
        by_round_scores.setdefault(round_idx, []).append(scores)
        by_round_clients.setdefault(round_idx, []).append(client_name)
        meta_rows.append(
            {
                "round": round_idx,
                "client": client_name,
                "num_layers": len(scores),
                "mean_score": float(
                    np.nanmean([s.score for s in scores]) if scores else float("nan")
                ),
            }
        )

        if args.per_client:
            stem = f"round_{round_idx:03d}_{client_name}"
            save_scores_csv(scores, out_dir / "per_client" / f"{stem}.csv")
            plot_layer_scores_by_depth(
                scores,
                title=(
                    f"resid×misalign | round {round_idx} {client_name} | "
                    f"top{int(args.top_ratio * 100)}% |R|"
                ),
                out_path=out_dir / "per_client" / f"{stem}.png",
            )

    # One figure per round: layer score = mean over the round's malicious clients.
    round_mean_map: Dict[int, List[LayerScore]] = {}
    for round_idx in sorted(by_round_scores):
        client_list = by_round_clients[round_idx]
        n_clients = len(by_round_scores[round_idx])
        if n_clients != int(args.expected_clients):
            raise RuntimeError(
                f"round {round_idx}: expected {args.expected_clients} malicious clients, "
                f"got {n_clients}: {client_list}"
            )
        mean_scores = average_scores_by_layer(by_round_scores[round_idx])
        round_mean_map[round_idx] = mean_scores
        stem = f"round_{round_idx:03d}_mean4"
        save_scores_csv(mean_scores, out_dir / f"{stem}.csv")
        plot_layer_scores_by_depth(
            mean_scores,
            title=(
                f"resid×misalign | round {round_idx} | "
                f"mean of {n_clients} malicious clients | "
                f"top{int(args.top_ratio * 100)}% |R|"
            ),
            out_path=out_dir / f"{stem}.png",
        )
        print(
            f"round {round_idx:03d}: averaged {n_clients} clients "
            f"({', '.join(client_list)}) -> {stem}.png"
        )

    plot_rounds_panel(
        round_mean_map,
        title=(
            f"resid×misalign per round | each panel = {args.expected_clients}-client mean | "
            f"top{int(args.top_ratio * 100)}% |R|"
        ),
        out_path=out_dir / "all_rounds_panel.png",
    )

    summary = {
        "delta_root": str(delta_root),
        "out_dir": str(out_dir),
        "top_ratio": float(args.top_ratio),
        "expected_clients_per_round": int(args.expected_clients),
        "num_snapshots": len(client_dirs),
        "rounds": sorted(by_round_scores.keys()),
        "clients_per_round": {
            str(k): v for k, v in sorted(by_round_clients.items())
        },
        "snapshots": meta_rows,
        "method": {
            "residual": "R = Δm - proj_Δc(Δm)",
            "select": "top_ratio by |R|",
            "score": "||R||/||Δm|| * (1 - cos(R_top, Δg_top)) / 2",
            "aggregation": "per round, mean over malicious clients (default 4)",
            "plot_order": "reference axis: *.weight in network depth order",
            "layers": "all *.weight tensors (conv/bn/shortcut/linear)",
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Analyzed {len(client_dirs)} snapshots from {delta_root}")
    print(f"Wrote {len(round_mean_map)} per-round figures to {out_dir}")
    print(f"Panel: {out_dir / 'all_rounds_panel.png'}")


if __name__ == "__main__":
    main()
