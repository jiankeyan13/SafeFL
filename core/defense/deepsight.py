"""DeepSight ensemble clustering + norm-clipping defense (Rieger et al., NDSS 2022, single-file v2)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import hdbscan
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF
from torch.func import functional_call, vmap

from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext, split_model_state
from core.utils import functional as F
from core.utils.registry import DEFENSE_REGISTRY


@DEFENSE_REGISTRY.register("deepsight")
class DeepSightDefense(BaseDefense):
    """NEUP/TE + classifier bias cosine + DDif ensemble HDBSCAN + median norm clipping."""

    def __init__(
        self,
        num_seeds: int = 3,
        num_samples: int = 20000,
        batch_size: int = 512,
        tau: float = 1.0 / 3.0,
        epsilon: float = 1e-7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_seeds = int(num_seeds)
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.tau = float(tau)
        self.epsilon = float(epsilon)
        self._noise_cache: Dict[Tuple[Any, ...], List[torch.Tensor]] = {}

    @staticmethod
    def _resolve_output_layer(param_keys: List[str]) -> Tuple[Optional[str], Optional[str]]:
        for prefix in ("fc", "linear", "classifier"):
            if f"{prefix}.weight" in param_keys and f"{prefix}.bias" in param_keys:
                return f"{prefix}.weight", f"{prefix}.bias"
        for k in reversed(param_keys):
            if k.endswith(".weight"):
                b_key = k[: -len("weight")] + "bias"
                if b_key in param_keys:
                    return k, b_key
        return None, None

    @staticmethod
    def _infer_input_shape(model: nn.Module, num_classes: int) -> Tuple[int, int, int]:
        if hasattr(model, "input_shape"):
            return tuple(model.input_shape)
        in_ch = 3
        for mod in model.modules():
            if isinstance(mod, nn.Conv2d):
                in_ch = int(mod.in_channels)
                break
        if in_ch == 1:
            return (1, 28, 28)
        if num_classes >= 200:
            return (in_ch, 64, 64)
        return (in_ch, 32, 32)

    @staticmethod
    def _dists_from_clusters(clusters: np.ndarray, n: int) -> np.ndarray:
        dists = np.ones((n, n), dtype=np.float64)
        same = (clusters[:, None] == clusters[None, :]) & (clusters[:, None] != -1)
        dists[same] = 0.0
        return dists

    def _get_noise_batches(
        self,
        device: torch.device,
        input_shape: Tuple[int, int, int],
        seed: int,
    ) -> List[torch.Tensor]:
        key = (str(device), input_shape, self.num_samples, self.batch_size, seed)
        cached = self._noise_cache.get(key)
        if cached is not None:
            return cached

        remaining = self.num_samples
        num_batches = max(1, math.ceil(self.num_samples / self.batch_size))
        cap = max(1, math.ceil(self.num_samples / num_batches))
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        batches: List[torch.Tensor] = []
        while remaining > 0 and len(batches) < num_batches:
            cur = min(cap, remaining)
            batches.append(torch.rand((cur, *input_shape), device=device, generator=gen))
            remaining -= cur
        self._noise_cache[key] = batches
        return batches

    def _calc_neups_tes(
        self,
        updates: List[ClientUpdate],
        w_name: str,
        b_name: str,
        num_classes: int,
        device: torch.device,
    ) -> Tuple[np.ndarray, List[int]]:
        w_stack = torch.stack([u.param_delta[w_name].to(device).float() for u in updates], dim=0)
        b_stack = torch.stack([u.param_delta[b_name].to(device).float() for u in updates], dim=0)
        red_dims = tuple(range(2, w_stack.dim())) if w_stack.dim() > 2 else ()
        diff_w = torch.sum(w_stack.abs(), dim=red_dims) if red_dims else w_stack.abs()
        energy_sq = (diff_w + b_stack.abs()).square()
        tot_e = energy_sq.view(energy_sq.shape[0], -1).sum(dim=1, keepdim=True)
        neup = energy_sq / torch.where(tot_e <= 0, torch.ones_like(tot_e), tot_e)
        if bool((tot_e <= 0).any()):
            unif = torch.full_like(neup, 1.0 / max(energy_sq.shape[1], 1))
            neup = torch.where(tot_e <= 0, unif, neup)

        max_neup = neup.max(dim=1).values
        thresh = max(0.01, 1.0 / max(num_classes, 1)) * max_neup
        tes = torch.sum(neup >= thresh.unsqueeze(1), dim=1).to(torch.int).cpu().tolist()
        return neup.detach().cpu().numpy(), tes

    @torch.no_grad()
    def _calc_ddifs(
        self,
        updates: List[ClientUpdate],
        global_model: nn.Module,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        device: torch.device,
    ) -> np.ndarray:
        g_params, g_buffers = split_model_state(global_model, trainable_only=False)
        stacked_params = {
            k: g_t + torch.stack(
                [u.param_delta.get(k, torch.zeros_like(g_t)).to(device=g_t.device, dtype=g_t.dtype) for u in updates],
                dim=0,
            )
            for k, g_t in g_params.items()
        }
        stacked_buffers = {
            k: torch.stack(
                [u.buffers.get(k, g_b).to(device=g_b.device, dtype=g_b.dtype) for u in updates],
                dim=0,
            )
            for k, g_b in g_buffers.items()
        }

        was_training = global_model.training
        global_model.eval()
        ddifs = []
        try:
            for seed in range(self.num_seeds):
                noise_batches = self._get_noise_batches(device, input_shape, seed)
                seed_ddif = torch.zeros(len(updates), num_classes, device=device)
                for inputs in noise_batches:
                    with torch.inference_mode():
                        prob_global = tF.softmax(global_model(inputs), dim=-1)

                    def _per_client(p_dict, b_dict):
                        logits = functional_call(global_model, (p_dict, b_dict), (inputs,))
                        prob_loc = tF.softmax(logits, dim=-1)
                        return torch.div(prob_loc, prob_global + self.epsilon).sum(dim=0)

                    batch_ddif = vmap(_per_client, in_dims=(0, 0))(stacked_params, stacked_buffers)
                    seed_ddif.add_(batch_ddif)
                ddifs.append(seed_ddif / float(self.num_samples))
        finally:
            if was_training:
                global_model.train()

        return torch.stack(ddifs, dim=0).detach().cpu().numpy()

    def _decide_benign(self, clusters: np.ndarray, suspicious: List[bool]) -> List[int]:
        ben_cnt: Dict[int, int] = {}
        tot_cnt: Dict[int, int] = {}
        for idx, c in enumerate(clusters):
            if c == -1:
                continue
            tot_cnt[int(c)] = tot_cnt.get(int(c), 0) + 1
            if not suspicious[idx]:
                ben_cnt[int(c)] = ben_cnt.get(int(c), 0) + 1

        benign: List[int] = []
        for idx, c in enumerate(clusters):
            if c == -1:
                if not suspicious[idx]:
                    benign.append(idx)
                continue
            tot = tot_cnt.get(int(c), 1)
            ben = ben_cnt.get(int(c), 0)
            if (tot - ben) / max(tot, 1) < self.tau:
                benign.append(idx)
        return benign

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        n = len(updates)
        norms = F.delta_l2_norms(updates, device=ctx.device)
        clip_value = float(np.median(norms)) if norms else 1.0

        param_keys = F.ordered_keys(updates[0].param_delta)
        w_name, b_name = self._resolve_output_layer(param_keys)
        if n < 2 or w_name is None or b_name is None:
            benign_idx = list(range(n))
        else:
            num_classes = int(updates[0].param_delta[b_name].numel())
            in_shape = self._infer_input_shape(global_model, num_classes)

            neups, tes = self._calc_neups_tes(updates, w_name, b_name, num_classes, ctx.device)
            med_te = float(np.median(tes)) if tes else 0.0
            suspicious = [te <= med_te * 0.5 for te in tes]

            bias_mat = torch.stack(
                [u.param_delta[b_name].to(ctx.device).float().reshape(-1) for u in updates], dim=0
            )
            cos_dist = F.pairwise_cosine_distance_np(bias_mat, eps=1e-9)
            ddifs = self._calc_ddifs(updates, global_model, in_shape, num_classes, ctx.device)

            ddif_dists = np.mean(
                [self._dists_from_clusters(hdbscan.HDBSCAN().fit_predict(d), n) for d in ddifs],
                axis=0,
            )
            neup_dists = self._dists_from_clusters(hdbscan.HDBSCAN().fit_predict(neups), n)
            cos_dists = self._dists_from_clusters(
                hdbscan.HDBSCAN(metric="precomputed").fit_predict(cos_dist), n
            )
            merged = np.mean([ddif_dists, neup_dists, cos_dists], axis=0)
            final_clusters = hdbscan.HDBSCAN(
                metric="precomputed", allow_single_cluster=True
            ).fit_predict(merged)
            benign_idx = self._decide_benign(final_clusters, suspicious)

        if not benign_idx:
            return DefenseOutput(
                param_delta=F.zero_delta_like(updates[0].param_delta, device=ctx.device),
                buffers=None,
                metrics={"defense/kept_clients": 0.0, "defense/clip_value": clip_value},
            )

        benign_deltas = [updates[i].param_delta for i in benign_idx]
        benign_norms = [norms[i] for i in benign_idx]
        clipped = F.clip_deltas_by_norm(
            benign_deltas, max_norm=clip_value, norms=benign_norms, device=ctx.device
        )
        param_delta = F.weighted_average_dicts(clipped, [1.0] * len(clipped), device=ctx.device)
        benign_buffers = [updates[i].buffers for i in benign_idx if updates[i].buffers]
        buffers = (
            F.weighted_average_dicts(benign_buffers, [1.0] * len(benign_buffers), device=ctx.device)
            if benign_buffers
            else None
        )
        return DefenseOutput(
            param_delta=param_delta,
            buffers=buffers,
            metrics={
                "defense/kept_clients": float(len(benign_idx)),
                "defense/clip_value": clip_value,
            },
        )
