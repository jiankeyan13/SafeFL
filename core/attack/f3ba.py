"""F3BA: Focused-Flip Federated Backdoor Attack (AAAI 2023)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as torch_f

from core.attack.base import BaseAttacker, collect_raw_batches
from core.attack.poisoner import LearnablePatchPoisoner, stamp_patch_differentiable
from core.types import ClientContext, ClientUpdate
from core.utils.registry import ATTACK_REGISTRY
from core.utils.trainer import build_criterion, train_model

F3BAPoisoner = LearnablePatchPoisoner


@ATTACK_REGISTRY.register("f3ba")
class F3BAAttack(BaseAttacker):
    """F3BA: Focused least-important weight sign flipping + conv1 trigger optimization."""

    default_poisoner_name: str = "learnable_patch"

    def __init__(
        self,
        patch_size: int = 5,
        patch_location: str = "bottom_right",
        init_value: float = 0.5,
        flip_fraction: float = 0.01,
        trigger_lr: float = 0.01,
        trigger_steps: int = 20,
        importance_batches: int = 5,
        max_opt_batches: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            patch_size=int(patch_size),
            patch_location=str(patch_location),
            init_value=float(init_value),
            **kwargs,
        )
        self.patch_size = int(patch_size)
        self.patch_location = str(patch_location)
        self.init_value = float(init_value)
        self.flip_fraction = float(flip_fraction)
        self.trigger_lr = float(trigger_lr)
        self.trigger_steps = int(trigger_steps)
        self.importance_batches = int(importance_batches)
        self.max_opt_batches = None if max_opt_batches is None else int(max_opt_batches)

    def _focused_sign_flip(self, ctx: ClientContext) -> int:
        if self.flip_fraction <= 0.0 or len(ctx.data) == 0:
            return 0

        target_weights: List[nn.Parameter] = []
        seen_ids: Set[int] = set()
        for mod in ctx.model.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                w = mod.weight
                if w.requires_grad and w.ndim >= 2 and id(w) not in seen_ids:
                    seen_ids.add(id(w))
                    target_weights.append(w)

        if not target_weights:
            return 0

        importance: Dict[int, torch.Tensor] = {
            id(w): torch.zeros_like(w, dtype=torch.float32) for w in target_weights
        }

        was_training = ctx.model.training
        ctx.model.eval()
        criterion = build_criterion(ctx.train_cfg)
        batches_seen = 0
        for batch_idx, (bx, by) in enumerate(ctx.loader(shuffle=False)):
            if self.importance_batches > 0 and batch_idx >= self.importance_batches:
                break
            bx = bx.to(ctx.device, non_blocking=True)
            by = by.to(ctx.device, non_blocking=True)
            ctx.model.zero_grad(set_to_none=True)
            loss = criterion(ctx.model(bx), by)
            loss.backward()
            for w in target_weights:
                if w.grad is not None:
                    importance[id(w)].add_((w.grad.detach() * w.detach()).abs())
            batches_seen += 1

        ctx.model.zero_grad(set_to_none=True)
        if was_training:
            ctx.model.train()

        if batches_seen == 0:
            return 0

        flipped = 0
        with torch.no_grad():
            for w in target_weights:
                numel = w.numel()
                k = int(numel * self.flip_fraction)
                if k == 0:
                    k = 1
                k = min(k, numel)
                if k <= 0:
                    continue
                imp_flat = importance[id(w)].view(-1)
                low_idx = torch.topk(imp_flat, k=k, largest=False).indices
                w.data.view(-1)[low_idx].mul_(-1.0)
                flipped += k
        return flipped

    def _optimize_trigger(
        self,
        ctx: ClientContext,
        raw_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        mean_t: torch.Tensor,
        std_t: torch.Tensor,
        init_trigger: torch.Tensor,
    ) -> torch.Tensor:
        first_conv = next(
            (m for m in ctx.model.modules() if isinstance(m, nn.Conv2d)), None
        )
        if first_conv is None:
            return init_trigger.detach().cpu()

        trigger = init_trigger.detach().clone().to(ctx.device)
        was_training = ctx.model.training
        ctx.model.eval()

        for _ in range(self.trigger_steps):
            for raw_x, _ in raw_batches:
                clean_norm = (raw_x - mean_t) / std_t
                with torch.no_grad():
                    clean_act = torch_f.relu(first_conv(clean_norm))

                t_var = trigger.detach().requires_grad_(True)
                poisoned_norm = stamp_patch_differentiable(
                    raw_x, t_var, self.patch_location, mean_t, std_t
                )
                poison_act = torch_f.relu(first_conv(poisoned_norm))

                loss = -torch.mean((poison_act - clean_act).pow(2))
                ctx.model.zero_grad(set_to_none=True)
                loss.backward()

                if t_var.grad is not None:
                    trigger = torch.clamp(
                        trigger - self.trigger_lr * t_var.grad.sign(),
                        min=0.0,
                        max=1.0,
                    ).detach()

        ctx.model.zero_grad(set_to_none=True)
        if was_training:
            ctx.model.train()
        return trigger.detach().cpu()

    def step(self, ctx: ClientContext) -> ClientUpdate:
        if isinstance(self.poisoner, LearnablePatchPoisoner):
            self.poisoner.load_state(ctx.state)

        num_flipped = self._focused_sign_flip(ctx)

        raw_batches, mean_t, std_t = collect_raw_batches(ctx, max_batches=self.max_opt_batches)
        if raw_batches and isinstance(self.poisoner, LearnablePatchPoisoner):
            channels = int(raw_batches[0][0].shape[1])
            init_trig = self.poisoner.ensure_trigger(channels, ctx.device)
            opt_trig = self._optimize_trigger(ctx, raw_batches, mean_t, std_t, init_trig)
            self.poisoner.trigger = opt_trig
            ctx.state["trigger"] = opt_trig.clone()

        stats = train_model(ctx.model, ctx.loader(poisoner=self.poisoner), ctx.train_cfg)
        return ctx.pack_update(
            metrics={"loss": stats.loss, "flipped_params": float(num_flipped)}
        )
