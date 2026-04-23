from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import torch

from core.utils.registry import SCREENER_REGISTRY
from .base_screener import BaseScreener


@SCREENER_REGISTRY.register("alignins")
class AlignInsScreener(BaseScreener):
    """
    AlignIns screening stage:
    1. TDA: cosine alignment between client delta and current global model.
    2. MPSA: sign alignment on each client's top-k important coordinates.
    3. Z-score filtering on both indicators.
    """

    def __init__(self, top_ratio: float = 0.3,  # top_ratio: fraction of coordinates kept for MPSA.
                 tda_threshold: float = 1.0,  # tda_threshold: z-score cutoff for TDA filtering.
                 mpsa_threshold: float = 1.0,  # mpsa_threshold: z-score cutoff for MPSA filtering.
                 eps: float = 1e-12,  # eps: numerical stability guard for norm and std checks.
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not (0.0 < float(top_ratio) <= 1.0):
            raise ValueError("top_ratio must be in (0, 1].")

        self.top_ratio = float(top_ratio)
        self.tda_threshold = float(tda_threshold)
        self.mpsa_threshold = float(mpsa_threshold)
        self.eps = float(eps)

    def screen(self, client_deltas: List[Dict[str, torch.Tensor]], num_samples: List[float], 
                        global_model: torch.nn.Module = None, context: Dict[str, Any] = None) -> Tuple[List[float], Dict[str, Any]]:
        del num_samples
        context = context or {}
        num_clients = len(client_deltas)

        if num_clients == 0:
            context["alignins_benign_indices"] = []
            return [], context

        learnable_keys = self._get_learnable_keys(global_model, client_deltas[0])
        client_vectors = [self._flatten_delta(delta, learnable_keys) for delta in client_deltas]

        if any(vector.numel() == 0 for vector in client_vectors):
            benign_indices = list(range(num_clients))
            context["alignins_benign_indices"] = benign_indices
            return [1.0] * num_clients, context

        global_vector = self._flatten_global_model(global_model, learnable_keys, client_vectors[0].device)
        tda_scores = [self._cosine_similarity(vector, global_vector).item() for vector in client_vectors]

        principal_sign = self._build_principal_sign(client_vectors)
        mpsa_scores = [self._compute_mpsa(vector, principal_sign) for vector in client_vectors]

        tda_zscores = self._compute_zscores(tda_scores)
        mpsa_zscores = self._compute_zscores(mpsa_scores)

        benign_indices = [idx for idx, (tda_z, mpsa_z) in enumerate(zip(tda_zscores, mpsa_zscores)) if tda_z <= self.tda_threshold and mpsa_z <= self.mpsa_threshold] or list(range(num_clients))
        screen_scores = [1.0 if idx in benign_indices else 0.0 for idx in range(num_clients)]
        context["alignins_benign_indices"] = benign_indices
        context["alignins_client_norms"] = [float(v.norm(p=2).item()) for v in client_vectors]
        return screen_scores, context

    def _get_learnable_keys(self, global_model: torch.nn.Module, first_delta: Dict[str, torch.Tensor]) -> Sequence[str]:
        if global_model is not None:
            return [name for name, param in global_model.named_parameters() if param.requires_grad]
        return list(first_delta.keys())

    def _flatten_delta(self, delta: Dict[str, torch.Tensor], keys: Sequence[str]) -> torch.Tensor:
        flat_parts = [delta[name].detach().float().reshape(-1) for name in keys if name in delta]
        if not flat_parts:
            return torch.zeros(0, dtype=torch.float32)
        return torch.cat(flat_parts, dim=0)

    def _flatten_global_model(self, global_model: torch.nn.Module, keys: Sequence[str], device: torch.device) -> torch.Tensor:
        state_dict = global_model.state_dict()
        flat_parts = [state_dict[name].detach().to(device=device, dtype=torch.float32).reshape(-1) for name in keys if name in state_dict]
        if not flat_parts:
            return torch.zeros(0, device=device, dtype=torch.float32)
        return torch.cat(flat_parts, dim=0)

    def _cosine_similarity(self, vector_a: torch.Tensor, vector_b: torch.Tensor) -> torch.Tensor:
        if vector_a.numel() == 0 or vector_b.numel() == 0:
            device = vector_a.device if vector_a.numel() > 0 else vector_b.device
            return torch.tensor(1.0, device=device)

        denom = vector_a.norm(p=2) * vector_b.norm(p=2)
        return torch.tensor(1.0, device=vector_a.device) if float(denom) <= self.eps else torch.dot(vector_a, vector_b) / denom.clamp_min(self.eps)

    def _build_principal_sign(self, client_vectors: Sequence[torch.Tensor]) -> torch.Tensor:
        principal_sign = torch.sign(torch.stack([torch.sign(vector) for vector in client_vectors], dim=0).sum(dim=0))
        if torch.any(principal_sign == 0):
            raw_sum_sign = torch.sign(torch.stack(client_vectors, dim=0).sum(dim=0))
            principal_sign = torch.where(principal_sign == 0, raw_sum_sign, principal_sign)
            principal_sign = torch.where(principal_sign == 0, torch.ones_like(principal_sign), principal_sign)
        return principal_sign

    def _compute_mpsa(self, client_vector: torch.Tensor, principal_sign: torch.Tensor) -> float:
        num_dims = int(client_vector.numel())
        if num_dims == 0:
            return 1.0

        top_k = max(1, int(math.ceil(num_dims * self.top_ratio)))
        top_indices = torch.topk(client_vector.abs(), k=top_k, largest=True).indices
        client_sign = torch.sign(client_vector[top_indices])
        reference_sign = principal_sign[top_indices]
        mismatch = (client_sign != reference_sign).sum().item()
        return 1.0 - float(mismatch) / float(top_k)

    def _compute_zscores(self, values: Sequence[float]) -> List[float]:
        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.numel() == 0:
            return []

        median, std = torch.median(tensor), torch.std(tensor, unbiased=False)
        if float(std) <= self.eps:
            return [0.0] * tensor.numel()
        return (torch.abs(tensor - median) / std.clamp_min(self.eps)).tolist()
