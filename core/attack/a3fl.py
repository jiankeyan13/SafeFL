"""A3FL: Adversarially Adaptive Backdoor Attacks to Federated Learning (NeurIPS 2023)."""
from __future__ import annotations

import copy
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn.functional as torch_f

from core.attack.base import BaseAttacker, collect_raw_batches
from core.attack.poisoner import LearnablePatchPoisoner, stamp_patch_differentiable
from core.types import ClientContext, ClientUpdate, compute_param_delta
from core.utils.registry import ATTACK_REGISTRY
from core.utils.trainer import train_model

A3FLPoisoner = LearnablePatchPoisoner


@ATTACK_REGISTRY.register("a3fl")
class A3FLAttack(BaseAttacker):
    """A3FL: Adversarially adaptive trigger optimization surviving simulated unlearning."""

    default_poisoner_name: str = "learnable_patch"

    def __init__(
        self,
        patch_size: int = 5,
        patch_location: str = "bottom_right",
        init_value: float = 0.5,
        trigger_lr: float = 0.01,
        trigger_steps: int = 20,
        adv_model_count: int = 1,
        adv_epochs: int = 5,
        adv_lr: float = 0.01,
        adv_momentum: float = 0.9,
        adv_weight_decay: float = 5e-4,
        adv_lambda: float = 1.0,
        max_opt_batches: Optional[int] = None,
        scale_factor: float = 1.0,
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
        self.trigger_lr = float(trigger_lr)
        self.trigger_steps = int(trigger_steps)
        self.adv_model_count = int(adv_model_count)
        self.adv_epochs = int(adv_epochs)
        self.adv_lr = float(adv_lr)
        self.adv_momentum = float(adv_momentum)
        self.adv_weight_decay = float(adv_weight_decay)
        self.adv_lambda = float(adv_lambda)
        self.max_opt_batches = None if max_opt_batches is None else int(max_opt_batches)
        self.scale_factor = float(scale_factor)

    def _optimize_trigger(
        self,
        ctx: ClientContext,
        raw_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        mean_t: torch.Tensor,
        std_t: torch.Tensor,
        init_trigger: torch.Tensor,
    ) -> torch.Tensor:
        trigger = init_trigger.detach().clone().to(ctx.device)
        target_label = self.poisoner.target_label

        adv_pairs = []
        for _ in range(max(0, self.adv_model_count)):
            adv_m = copy.deepcopy(ctx.model).to(ctx.device)
            adv_opt = torch.optim.SGD(
                adv_m.parameters(),
                lr=self.adv_lr,
                momentum=self.adv_momentum,
                weight_decay=self.adv_weight_decay,
            )
            adv_pairs.append((adv_m, adv_opt))

        for _ in range(self.trigger_steps):
            # 1. Adversarial unlearning phase: force surrogate models to unlearn trigger -> orig_y.
            for adv_m, adv_opt in adv_pairs:
                adv_m.train()
                for _ in range(self.adv_epochs):
                    for raw_x, orig_y in raw_batches:
                        poisoned_norm = stamp_patch_differentiable(
                            raw_x, trigger.detach(), self.patch_location, mean_t, std_t
                        )
                        adv_opt.zero_grad(set_to_none=True)
                        loss_unlearn = torch_f.cross_entropy(adv_m(poisoned_norm), orig_y)
                        loss_unlearn.backward()
                        adv_opt.step()
                adv_m.eval()

            # 2. Trigger update phase: survive unlearned surrogates while preserving global ASR.
            ctx.model.eval()
            for raw_x, by in raw_batches:
                non_target = by != target_label
                if not bool(non_target.any()):
                    continue

                clean_x = raw_x[non_target]
                t_var = trigger.detach().requires_grad_(True)
                poisoned_norm = stamp_patch_differentiable(
                    clean_x, t_var, self.patch_location, mean_t, std_t
                )
                target_y = torch.full(
                    (poisoned_norm.size(0),),
                    target_label,
                    dtype=torch.long,
                    device=ctx.device,
                )

                loss_global = torch_f.cross_entropy(ctx.model(poisoned_norm), target_y)
                if adv_pairs:
                    loss_adv = sum(
                        torch_f.cross_entropy(adv_m(poisoned_norm), target_y)
                        for adv_m, _ in adv_pairs
                    ) / len(adv_pairs)
                    loss = loss_global + self.adv_lambda * loss_adv
                else:
                    loss = loss_global

                ctx.model.zero_grad(set_to_none=True)
                for adv_m, _ in adv_pairs:
                    adv_m.zero_grad(set_to_none=True)
                loss.backward()

                if t_var.grad is not None:
                    trigger = torch.clamp(
                        trigger - self.trigger_lr * t_var.grad.sign(),
                        min=0.0,
                        max=1.0,
                    ).detach()

        ctx.model.zero_grad(set_to_none=True)
        return trigger.detach().cpu()

    def step(self, ctx: ClientContext) -> ClientUpdate:
        if isinstance(self.poisoner, LearnablePatchPoisoner):
            self.poisoner.load_state(ctx.state)

        raw_batches, mean_t, std_t = collect_raw_batches(ctx, max_batches=self.max_opt_batches)
        if raw_batches and isinstance(self.poisoner, LearnablePatchPoisoner):
            channels = int(raw_batches[0][0].shape[1])
            init_trig = self.poisoner.ensure_trigger(channels, ctx.device)
            opt_trig = self._optimize_trigger(ctx, raw_batches, mean_t, std_t, init_trig)
            self.poisoner.trigger = opt_trig
            ctx.state["trigger"] = opt_trig.clone()

        ctx.reset_model()
        stats = train_model(ctx.model, ctx.loader(poisoner=self.poisoner), ctx.train_cfg)

        if self.scale_factor != 1.0:
            param_delta, buffers = compute_param_delta(ctx.model, ctx.initial_params)
            scaled_delta = {k: v * self.scale_factor for k, v in param_delta.items()}
            return ctx.pack_update(
                param_delta=scaled_delta,
                buffers=buffers,
                metrics={"loss": stats.loss},
            )
        return ctx.pack_update(metrics={"loss": stats.loss})
