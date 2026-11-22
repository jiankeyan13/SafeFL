"""Mirage: Discriminator-evading trigger-optimization backdoor attack (CVPR 2025, single-file v2)."""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from torch.utils.data import DataLoader, TensorDataset

from core.attack.base import BaseAttacker
from core.attack.poisoner import BasePoisoner
from core.types import ClientContext, ClientUpdate
from core.utils.registry import ATTACK_REGISTRY
from core.utils.trainer import train_model


class MiragePoisoner(BasePoisoner):
    """Blend-based poisoner holding a per-attacker optimizable trigger tensor."""

    def __init__(
        self,
        target_label: int = 5,
        poison_ratio: float = 0.5,
        blend_alpha: float = 0.3,
        seed: Optional[int] = 42,
        exclude_target_class: bool = False,
        **_: Any,
    ) -> None:
        super().__init__(
            target_label=target_label,
            poison_ratio=poison_ratio,
            seed=seed,
            exclude_target_class=exclude_target_class,
        )
        self.blend_alpha = float(blend_alpha)
        self.trigger: Optional[torch.Tensor] = None

    def ensure_trigger(self, shape: Sequence[int], device: torch.device) -> torch.Tensor:
        if self.trigger is None or tuple(self.trigger.shape) != tuple(shape):
            self.trigger = torch.full(tuple(shape), 0.5, dtype=torch.float32, device="cpu")
        return self.trigger.to(device=device, dtype=torch.float32)

    def load_state(self, state: Mapping[str, Any]) -> None:
        trig = state.get("trigger")
        if torch.is_tensor(trig):
            self.trigger = trig.detach().cpu().clone()

    def _blend(self, bx: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        trig = self.ensure_trigger(bx.shape[1:], bx.device).unsqueeze(0)
        alpha = self.blend_alpha
        if mask is None:
            bx.mul_(1.0 - alpha).add_(trig * alpha)
        elif bool(mask.any()):
            bx[mask] = bx[mask] * (1.0 - alpha) + trig * alpha
        return bx

    def _stamp_train(
        self,
        bx: torch.Tensor,
        mask: torch.Tensor,
        *,
        client_id: Optional[int] = None,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del client_id, round_idx
        return self._blend(bx, mask)

    def _stamp_eval(
        self,
        bx: torch.Tensor,
        *,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del round_idx
        return self._blend(bx, None)


def _replace_classifier_head(model: nn.Module) -> nn.Module:
    """Clone ``model`` and replace its final Linear classifier with a 2-class MLP head."""
    disc = copy.deepcopy(model)
    for param in disc.parameters():
        param.requires_grad = False

    last_parent: Optional[nn.Module] = None
    last_name: Optional[str] = None
    last_linear: Optional[nn.Linear] = None

    for parent in disc.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                last_parent = parent
                last_name = name
                last_linear = child

    if last_parent is None or last_name is None or last_linear is None:
        raise ValueError("Mirage requires a model with at least one nn.Linear classifier layer.")

    head = nn.Sequential(
        nn.Linear(last_linear.in_features, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
    )
    for param in head.parameters():
        param.requires_grad = True
    setattr(last_parent, last_name, head)
    return disc


@ATTACK_REGISTRY.register("mirage")
class MirageAttack(BaseAttacker):
    """Mirage: Per-attacker trigger optimization against a local OOD/ID discriminator."""

    def __init__(
        self,
        blend_alpha: float = 0.3,
        target_labels: Optional[Sequence[int]] = None,
        discriminator_train_samples_per_class: int = 50,
        discriminator_batch_size: int = 64,
        discriminator_lr: float = 0.01,
        discriminator_momentum: float = 0.9,
        discriminator_weight_decay: float = 0.0005,
        discriminator_epochs: int = 30,
        trigger_search_steps: int = 50,
        trigger_lr: float = 0.001,
        **kwargs: Any,
    ) -> None:
        super().__init__(blend_alpha=float(blend_alpha), **kwargs)
        self.blend_alpha = float(blend_alpha)
        self.target_labels = [int(x) for x in target_labels] if target_labels else None
        self.disc_samples_per_class = int(discriminator_train_samples_per_class)
        self.disc_batch_size = int(discriminator_batch_size)
        self.disc_lr = float(discriminator_lr)
        self.disc_momentum = float(discriminator_momentum)
        self.disc_weight_decay = float(discriminator_weight_decay)
        self.disc_epochs = int(discriminator_epochs)
        self.trigger_search_steps = int(trigger_search_steps)
        self.trigger_lr = float(trigger_lr)

    def _build_default_poisoner(self, base_kwargs: Dict[str, Any]) -> MiragePoisoner:
        return MiragePoisoner(**base_kwargs)

    def _resolve_target_label(self, ctx: ClientContext) -> int:
        if self.target_labels:
            rank = ctx.attacker_rank if ctx.attacker_rank is not None else ctx.client_id
            return self.target_labels[int(rank) % len(self.target_labels)]
        return self.poisoner.target_label

    @torch.no_grad()
    def _build_discriminator_loader(
        self,
        model: nn.Module,
        batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        trigger: torch.Tensor,
        target_label: int,
        device: torch.device,
    ) -> Optional[DataLoader]:
        model.eval()
        ce_none = nn.CrossEntropyLoss(reduction="none")
        per_class_x: dict[int, List[torch.Tensor]] = {}

        for bx, by in batches:
            bx = bx.to(device)
            by = by.to(device)
            for cls_id in by.unique().tolist():
                c = int(cls_id)
                per_class_x.setdefault(c, []).append(bx[by == c])

        if target_label not in per_class_x:
            return None

        rep_x: dict[int, torch.Tensor] = {}
        for c, chunks in per_class_x.items():
            samples = torch.cat(chunks, dim=0)
            logits = model(samples)
            tmp_y = torch.full((samples.size(0),), c, dtype=torch.long, device=device)
            losses = ce_none(logits, tmp_y)
            k = samples.size(0) if c == target_label else min(samples.size(0), self.disc_samples_per_class)
            top_idx = torch.topk(losses, k=k, largest=False).indices
            rep_x[c] = samples[top_idx]

        all_x: List[torch.Tensor] = []
        all_y: List[torch.Tensor] = []
        alpha = self.blend_alpha
        trig = trigger.detach().unsqueeze(0)

        for c, samples in rep_x.items():
            if c == target_label:
                continue
            poisoned = samples * (1.0 - alpha) + trig * alpha
            all_x.append(poisoned)
            all_y.append(torch.ones(samples.size(0), dtype=torch.long, device=device))

        if not all_x:
            return None

        target_samples = rep_x[target_label]
        all_x.append(target_samples)
        all_y.append(torch.zeros(target_samples.size(0), dtype=torch.long, device=device))

        ds = TensorDataset(torch.cat(all_x, dim=0), torch.cat(all_y, dim=0))
        return DataLoader(ds, batch_size=self.disc_batch_size, shuffle=True)

    def _train_discriminator(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> nn.Module:
        disc = _replace_classifier_head(model).to(device)
        disc.train()
        opt = torch.optim.SGD(
            [p for p in disc.parameters() if p.requires_grad],
            lr=self.disc_lr,
            momentum=self.disc_momentum,
            weight_decay=self.disc_weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        for _ in range(self.disc_epochs):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss = criterion(disc(bx), by)
                loss.backward()
                opt.step()
        disc.eval()
        return disc

    def _search_trigger(
        self,
        model: nn.Module,
        batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        init_trigger: torch.Tensor,
        target_label: int,
        device: torch.device,
    ) -> torch.Tensor:
        model.eval()
        t = init_trigger.detach().clone().to(device)
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
        alpha = self.blend_alpha

        for _ in range(self.trigger_search_steps):
            disc_loader = self._build_discriminator_loader(model, batches, t, target_label, device)
            if disc_loader is None:
                break
            disc = self._train_discriminator(model, disc_loader, device)

            for bx, by in batches:
                bx, by = bx.to(device), by.to(device)
                non_target = by != target_label
                if not bool(non_target.any()):
                    continue

                clean_x = bx[non_target]
                t_var = t.detach().requires_grad_(True)
                poisoned_x = clean_x * (1.0 - alpha) + t_var.unsqueeze(0) * alpha
                poisoned_y = torch.full((clean_x.size(0),), target_label, dtype=torch.long, device=device)

                disc_logits = disc(poisoned_x)
                loss_disc = torch_f.cross_entropy(
                    disc_logits,
                    torch.zeros(clean_x.size(0), dtype=torch.long, device=device),
                )
                with torch.no_grad():
                    clean_logits = model(clean_x)
                poison_logits = model(poisoned_x)
                loss_asr = torch_f.cross_entropy(poison_logits, poisoned_y)
                loss_sim = cos_sim(poison_logits, clean_logits).mean()

                loss = loss_disc + loss_asr + loss_sim
                if t_var.grad is not None:
                    t_var.grad.zero_()
                loss.backward()
                if t_var.grad is not None:
                    t = torch.clamp(t - t_var.grad.sign() * self.trigger_lr, min=0.0, max=1.0).detach()

        return t.detach().cpu()

    def step(self, ctx: ClientContext) -> ClientUpdate:
        target_label = self._resolve_target_label(ctx)
        self.poisoner.target_label = target_label
        if isinstance(self.poisoner, MiragePoisoner):
            self.poisoner.load_state(ctx.state)

        clean_batches = [(bx.detach().cpu(), by.detach().cpu()) for bx, by in ctx.loader(shuffle=False)]
        if clean_batches and isinstance(self.poisoner, MiragePoisoner):
            sample_shape = clean_batches[0][0].shape[1:]
            init_trig = self.poisoner.ensure_trigger(sample_shape, ctx.device)
            opt_trig = self._search_trigger(ctx.model, clean_batches, init_trig, target_label, ctx.device)
            self.poisoner.trigger = opt_trig
            ctx.state["trigger"] = opt_trig.clone()

        stats = train_model(ctx.model, ctx.loader(poisoner=self.poisoner), ctx.train_cfg)
        return ctx.pack_update(metrics={"loss": stats.loss})
