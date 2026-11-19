"""Neurotoxin durable backdoor attack (single-file v2 implementation)."""
from __future__ import annotations

import csv
import os
import threading
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn

from core.attack.base import BaseAttacker
from core.types import ClientContext, ClientUpdate
from core.utils.registry import ATTACK_REGISTRY
from core.utils.trainer import build_criterion, train_model

_CSV_LOCK = threading.Lock()


def compute_clean_batch_gradients(
    model: nn.Module,
    bx: torch.Tensor,
    by: torch.Tensor,
    loss_fn: nn.Module,
) -> Dict[str, torch.Tensor]:
    """Run one forward-backward pass on a clean batch and clone per-parameter gradients."""
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(bx), by)
    loss.backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return grads


@torch.no_grad()
def build_neurotoxin_masks(
    model: nn.Module,
    reference_gradients: Mapping[str, torch.Tensor],
    mask_ratio: float,
) -> Dict[str, torch.Tensor]:
    """Zero out top ``mask_ratio`` coordinates globally ranked by ``|grad_clean|``."""
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
    if not reference_gradients or mask_ratio == 0.0:
        return {}

    entries = []
    flat_scores = []
    target_dev: Optional[torch.device] = None
    for name, param in model.named_parameters():
        ref = reference_gradients.get(name)
        if not param.requires_grad or ref is None or ref.numel() != param.numel():
            continue
        if target_dev is None:
            target_dev = param.device
        scores = ref.detach().to(device=target_dev, dtype=torch.float32).abs().reshape(-1)
        entries.append((name, param, scores.numel()))
        flat_scores.append(scores)

    if not flat_scores or target_dev is None:
        return {}

    all_scores = torch.cat(flat_scores)
    num_masked = min(all_scores.numel(), int(all_scores.numel() * mask_ratio))
    if num_masked == 0:
        return {}

    blocked = torch.zeros(all_scores.numel(), dtype=torch.bool, device=target_dev)
    blocked[torch.topk(all_scores, k=num_masked, largest=True, sorted=False).indices] = True

    masks: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, param, numel in entries:
        keep = ~blocked[offset : offset + numel]
        masks[name] = keep.reshape(param.shape).to(device=param.device, dtype=param.dtype)
        offset += numel
    return masks


def _append_mask_csv(
    run_dir: str,
    round_idx: int,
    client_id: int,
    masks: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    layers: List[Dict[str, Any]] = []
    n_total = 0
    n_masked = 0
    for name, mask in masks.items():
        total = int(mask.numel())
        blocked = int((mask == 0).sum().item())
        layers.append(
            {
                "round": round_idx,
                "client_id": client_id,
                "layer": name,
                "n_total": total,
                "n_masked": blocked,
                "ratio": f"{(blocked / total if total > 0 else 0.0):.6f}",
            }
        )
        n_total += total
        n_masked += blocked

    if layers:
        csv_path = os.path.join(run_dir, "neurotoxin_layer_masks.csv")
        os.makedirs(run_dir, exist_ok=True)
        with _CSV_LOCK:
            write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["round", "client_id", "layer", "n_total", "n_masked", "ratio"]
                )
                if write_header:
                    writer.writeheader()
                writer.writerows(layers)

    return {
        "nt/mask_ratio": (n_masked / n_total) if n_total > 0 else 0.0,
        "nt/mask_n_masked": float(n_masked),
    }


@ATTACK_REGISTRY.register("neurotoxin")
class NeurotoxinAttack(BaseAttacker):
    """Neurotoxin gradient-masked backdoor training (ICML 2022)."""

    def __init__(
        self,
        mask_ratio: float = 0.10,
        log_layer_masks: bool = True,
        **kwargs: Any,
    ) -> None:
        if not 0.0 <= float(mask_ratio) <= 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
        super().__init__(**kwargs)
        self.mask_ratio = float(mask_ratio)
        self.log_layer_masks = bool(log_layer_masks)

    def step(self, ctx: ClientContext) -> ClientUpdate:
        clean_loader = ctx.loader(shuffle=True, seed=self.seed)
        bx_clean, by_clean = next(iter(clean_loader))
        criterion = build_criterion(ctx.train_cfg)
        ref_grads = compute_clean_batch_gradients(ctx.model, bx_clean, by_clean, criterion)
        masks = build_neurotoxin_masks(ctx.model, ref_grads, self.mask_ratio)

        poison_loader = ctx.loader(poisoner=self.poisoner, seed=self.seed)
        stats = train_model(
            ctx.model,
            poison_loader,
            ctx.train_cfg,
            criterion=criterion,
            grad_mask=masks,
        )

        metrics: Dict[str, float] = {"loss": stats.loss}
        if masks and self.log_layer_masks and ctx.run_dir:
            metrics.update(_append_mask_csv(ctx.run_dir, ctx.round_idx, ctx.client_id, masks))
        return ctx.pack_update(metrics=metrics)
