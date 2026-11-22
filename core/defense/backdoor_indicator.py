"""BackdoorIndicator defense: proactive OOD watermark injection + detection (USENIX Security 2024, single-file v2)."""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from torch.utils.data import DataLoader, TensorDataset

from core.defense.base import BaseDefense, build_client_weights
from core.types import (
    ClientUpdate,
    DefenseOutput,
    ServerContext,
    StateDict,
    load_model_state,
    split_model_state,
)
from core.utils import functional as F
from core.utils.registry import DEFENSE_REGISTRY


@DEFENSE_REGISTRY.register("backdoor_indicator")
class BackdoorIndicatorDefense(BaseDefense):
    """Proactive OOD watermarking before broadcast + post-round watermark accuracy filtering."""

    def __init__(
        self,
        wm_epochs: int = 200,
        wm_lr: float = 0.005,
        wm_momentum: float = 0.9,
        wm_weight_decay: float = 0.0005,
        wm_mu: float = 0.0,
        wm_milestones: Sequence[int] = (10, 20, 30, 160, 200, 240, 280, 320, 360),
        wm_lr_gamma: float = 0.8,
        is_projection_grad: bool = False,
        projection_norm: float = 0.8,
        detection_threshold: float = 0.95,
        replace_original_bn: bool = True,
        norm_clip: bool = False,
        ood_samples: int = 800,
        ood_batch_size: int = 64,
        watermark_start_round: int = 0,
        watermark_end_round: Optional[int] = None,
        watermark_interval: int = 1,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.wm_epochs = int(wm_epochs)
        self.wm_lr = float(wm_lr)
        self.wm_momentum = float(wm_momentum)
        self.wm_weight_decay = float(wm_weight_decay)
        self.wm_mu = float(wm_mu)
        self.wm_milestones = [int(m) for m in wm_milestones]
        self.wm_lr_gamma = float(wm_lr_gamma)
        self.is_projection_grad = bool(is_projection_grad)
        self.projection_norm = float(projection_norm)
        # Accept either [0, 1] fraction or [0, 100] percentage (official YAML uses 95).
        thr = float(detection_threshold)
        self.detection_threshold = thr / 100.0 if thr > 1.0 else thr
        self.replace_original_bn = bool(replace_original_bn)
        self.norm_clip = bool(norm_clip)
        self.ood_samples = int(ood_samples)
        self.ood_batch_size = int(ood_batch_size)
        self.watermark_start_round = int(watermark_start_round)
        self.watermark_end_round = None if watermark_end_round is None else int(watermark_end_round)
        self.watermark_interval = max(1, int(watermark_interval))
        self.seed = int(seed)

        self._wm_loader: Optional[DataLoader] = None
        self._wm_params: Optional[StateDict] = None
        self._after_wm_bn: Optional[StateDict] = None
        self._round_active: bool = False

    def _is_watermark_round(self, round_idx: int) -> bool:
        if round_idx < self.watermark_start_round:
            return False
        if self.watermark_end_round is not None and round_idx > self.watermark_end_round:
            return False
        return (round_idx - self.watermark_start_round) % self.watermark_interval == 0

    def _ensure_wm_loader(self, ctx: ServerContext) -> DataLoader:
        if self._wm_loader is not None:
            return self._wm_loader

        rng = np.random.default_rng(self.seed)
        if ctx.proxy_data is not None and len(ctx.proxy_data) > 0:
            xs: List[torch.Tensor] = []
            for bx, _ in ctx.proxy_data.loader(
                batch_size=self.ood_batch_size, shuffle=False, device=torch.device("cpu")
            ):
                xs.append(bx)
                if sum(t.size(0) for t in xs) >= self.ood_samples:
                    break
            data_x = torch.cat(xs, dim=0)[: self.ood_samples]
        else:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed)
            data_x = torch.randn((self.ood_samples, *ctx.input_shape), generator=gen, dtype=torch.float32)

        n = int(data_x.size(0))
        assigned = np.array([i % ctx.num_classes for i in range(n)], dtype=np.int64)
        rng.shuffle(assigned)
        data_y = torch.from_numpy(assigned)
        self._wm_loader = DataLoader(
            TensorDataset(data_x, data_y),
            batch_size=min(self.ood_batch_size, n),
            shuffle=False,
        )
        return self._wm_loader

    def pre_broadcast(
        self,
        global_model: nn.Module,
        ctx: ServerContext,
    ) -> Optional[Tuple[StateDict, StateDict]]:
        self._round_active = self._is_watermark_round(ctx.round_idx)
        if not self._round_active:
            self._wm_params = None
            self._after_wm_bn = None
            return None

        wm_loader = self._ensure_wm_loader(ctx)
        wm_model = copy.deepcopy(global_model).to(ctx.device)
        init_params, before_bn = split_model_state(wm_model)

        wm_model.train()
        opt = torch.optim.SGD(
            wm_model.parameters(),
            lr=self.wm_lr,
            momentum=self.wm_momentum,
            weight_decay=self.wm_weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=self.wm_milestones, gamma=self.wm_lr_gamma
        )

        for _ in range(self.wm_epochs):
            for bx, by in wm_loader:
                bx = bx.to(ctx.device, non_blocking=True)
                by = by.to(ctx.device, non_blocking=True)
                opt.zero_grad()
                loss = torch_f.cross_entropy(wm_model(bx), by)
                if self.wm_mu > 0.0:
                    dist_sq = sum(
                        torch.sum((p - init_params[name]).pow(2))
                        for name, p in wm_model.named_parameters()
                        if name in init_params
                    )
                    loss = loss + 0.5 * self.wm_mu * torch.sqrt(dist_sq + 1e-12)
                loss.backward()
                opt.step()

                if self.is_projection_grad:
                    with torch.no_grad():
                        norm = torch.sqrt(
                            sum(
                                torch.sum((p - init_params[name]).pow(2))
                                for name, p in wm_model.named_parameters()
                                if name in init_params
                            )
                            + 1e-12
                        )
                        if float(norm.item()) > self.projection_norm:
                            scale = self.projection_norm / float(norm.item())
                            for name, p in wm_model.named_parameters():
                                if name in init_params:
                                    p.copy_(init_params[name] + (p - init_params[name]) * scale)
            scheduler.step()

        wm_params, after_bn = split_model_state(wm_model)
        self._wm_params = wm_params
        self._after_wm_bn = after_bn
        broadcast_bn = before_bn if self.replace_original_bn else after_bn
        return wm_params, broadcast_bn

    @torch.no_grad()
    def _max_per_class_wm_acc(
        self,
        model: nn.Module,
        wm_loader: DataLoader,
        num_classes: int,
        device: torch.device,
    ) -> float:
        model.eval()
        class_correct = np.zeros(num_classes, dtype=np.int64)
        class_total = np.zeros(num_classes, dtype=np.int64)
        for bx, by in wm_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            preds = model(bx).argmax(dim=1)
            for c in range(num_classes):
                mask = by == c
                cnt = int(mask.sum().item())
                if cnt > 0:
                    class_total[c] += cnt
                    class_correct[c] += int((preds[mask] == c).sum().item())
        accs = [
            float(class_correct[c]) / float(class_total[c])
            for c in range(num_classes)
            if class_total[c] > 0
        ]
        return max(accs) if accs else 0.0

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        n = len(updates)
        if not self._round_active or self._wm_params is None:
            param_delta, buffers = F.aggregate_updates(updates, weight_by_samples=True, device=ctx.device)
            return DefenseOutput(
                param_delta=param_delta,
                buffers=buffers,
                metrics={"defense/kept_clients": float(n)},
                client_weights=build_client_weights(updates, weight_by_samples=True),
            )

        wm_loader = self._ensure_wm_loader(ctx)
        check_model = copy.deepcopy(global_model).to(ctx.device)
        benign_idx: List[int] = []

        for idx, u in enumerate(updates):
            local_params = {
                k: self._wm_params[k] + u.param_delta[k].to(device=ctx.device, dtype=self._wm_params[k].dtype)
                for k in self._wm_params
                if k in u.param_delta
            }
            check_bufs = self._after_wm_bn if self.replace_original_bn else u.buffers
            load_model_state(check_model, local_params, check_bufs)
            max_wm_acc = self._max_per_class_wm_acc(check_model, wm_loader, ctx.num_classes, ctx.device)
            if max_wm_acc < self.detection_threshold:
                benign_idx.append(idx)

        if not benign_idx:
            benign_idx = list(range(n))

        kept = [updates[i] for i in benign_idx]
        if self.norm_clip and len(kept) > 0:
            delta_mat, keys = F.stack_flat_deltas(kept, device=ctx.device)
            norms = torch.norm(delta_mat, p=2, dim=1)
            clip_val = float(torch.median(norms).item())
            weights = [float(max(int(u.num_samples), 1)) for u in kept]
            param_delta = F.clip_and_average_flat(
                delta_mat,
                list(range(len(kept))),
                clip_val,
                kept[0].param_delta,
                keys,
                norms=norms,
                weights=weights,
            )
            buffers = F.average_buffers(kept, weight_by_samples=True, device=ctx.device)
            return DefenseOutput(
                param_delta=param_delta,
                buffers=buffers,
                metrics={"defense/kept_clients": float(len(kept))},
                client_weights=build_client_weights(updates, kept_updates=kept, weight_by_samples=True),
            )

        param_delta, buffers = F.aggregate_updates(kept, weight_by_samples=True, device=ctx.device)
        return DefenseOutput(
            param_delta=param_delta,
            buffers=buffers,
            metrics={"defense/kept_clients": float(len(kept))},
            client_weights=build_client_weights(updates, kept_updates=kept, weight_by_samples=True),
        )
