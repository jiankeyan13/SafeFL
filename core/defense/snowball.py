"""Snowball defense: bottom-up KMeans+CH election + top-down VAE expansion (AAAI 2024, single-file v2)."""
from __future__ import annotations

import math
from typing import Any, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from torch.utils.data import DataLoader, TensorDataset

from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext
from core.utils import functional as F
from core.utils.registry import DEFENSE_REGISTRY


class _SnowballVAE(nn.Module):
    """Compact MLP VAE trained per round on pairwise update differences."""

    def __init__(self, input_dim: int, latent_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.fc_e1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc_e2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc_d1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_d2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_d3 = nn.Linear(hidden_dim, self.input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch_f.relu(self.fc_e1(x.view(-1, self.input_dim)))
        h = torch_f.relu(self.fc_e2(h))
        return self.fc_mean(h), torch_f.softplus(self.fc_logvar(h))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = torch_f.relu(self.fc_d1(z))
        h = torch_f.relu(self.fc_d2(h))
        return torch.sigmoid(self.fc_d3(h)).view(-1, self.input_dim)

    @staticmethod
    def sample_normal(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        sd = torch.exp(logvar * 0.5)
        return torch.randn_like(sd).mul(sd).add_(mean)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.sample_normal(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def recon_prob(self, x: torch.Tensor, num_samples: int = 10) -> float:
        x_sig = torch.sigmoid(x.unsqueeze(0))
        mu, logvar = self.encode(x_sig)
        total = 0.0
        for _ in range(num_samples):
            z = self.sample_normal(mu, logvar)
            recon = self.decode(z)
            total += float(torch_f.mse_loss(recon, x_sig, reduction="sum").item())
        return total / float(num_samples)


def _init_kaiming(model: nn.Module) -> None:
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(mod.weight.data, nonlinearity="relu")


def _build_dif_set(vecs: Sequence[torch.Tensor]) -> torch.Tensor:
    diffs = [vecs[i] - vecs[j] for i in range(len(vecs)) for j in range(len(vecs)) if i != j]
    if not diffs:
        return torch.zeros((1, vecs[0].numel()), device=vecs[0].device, dtype=vecs[0].dtype)
    return torch.stack(diffs, dim=0)


def _obtain_dif(base_vecs: Sequence[torch.Tensor], target: torch.Tensor) -> List[torch.Tensor]:
    diffs: List[torch.Tensor] = []
    for item in base_vecs:
        d = item - target
        if float(d.abs().sum().item()) != 0.0:
            diffs.append(d)
            diffs.append(-d)
    return diffs


def _train_vae(
    vae: _SnowballVAE | None,
    dif_tensor: torch.Tensor,
    epochs: int,
    device: torch.device,
    latent_dim: int,
    hidden_dim: int,
) -> _SnowballVAE:
    data = torch.sigmoid(dif_tensor.detach().to(device=device, dtype=torch.float32))
    if vae is None:
        vae = _SnowballVAE(input_dim=int(data.shape[1]), latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
        _init_kaiming(vae)
    else:
        vae = vae.to(device)

    vae.train()
    loader = DataLoader(TensorDataset(data), batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(vae.parameters())
    for _ in range(epochs):
        for (bx,) in loader:
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(bx)
            recon = torch_f.mse_loss(recon_x, bx, reduction="sum")
            kl = torch.mean(-0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp()))
            loss = recon + kl
            loss.backward()
            optimizer.step()
    return vae


@DEFENSE_REGISTRY.register("snowball")
class SnowballDefense(BaseDefense):
    """Two-stage bottom-up KMeans+CH + top-down VAE reconstruction defense."""

    def __init__(
        self,
        ct: int = 10,
        vt: float = 0.5,
        v_step: float = 0.05,
        vae_hidden: int = 256,
        vae_latent: int = 64,
        vae_initial: int = 270,
        vae_tuning: int = 30,
        warmup_rounds: int = 100,
        layer_keywords: Sequence[str] = ("conv1", "fc2", "linear"),
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ct = int(ct)
        self.vt = float(vt)
        self.v_step = float(v_step)
        self.vae_hidden = int(vae_hidden)
        self.vae_latent = int(vae_latent)
        self.vae_initial = int(vae_initial)
        self.vae_tuning = int(vae_tuning)
        self.warmup_rounds = int(warmup_rounds)
        self.layer_keywords = tuple(str(k) for k in layer_keywords)
        self.seed = int(seed)

    def _select_layer_keys(self, updates: Sequence[ClientUpdate]) -> List[str]:
        all_keys = list(updates[0].param_delta.keys())
        matched = [
            k for k in all_keys
            if any(kw in k for kw in self.layer_keywords)
        ]
        if matched:
            return matched
        if len(all_keys) >= 2:
            return [all_keys[0], all_keys[-1]]
        return all_keys

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        del global_model
        n = len(updates)
        if n <= 2:
            param_delta, buffers = F.aggregate_updates(updates, weight_by_samples=True, device=ctx.device)
            return DefenseOutput(param_delta=param_delta, buffers=buffers, metrics={"defense/kept_clients": float(n)})

        layer_keys = self._select_layer_keys(updates)
        if not layer_keys:
            param_delta, buffers = F.aggregate_updates(updates, weight_by_samples=True, device=ctx.device)
            return DefenseOutput(param_delta=param_delta, buffers=buffers, metrics={"defense/kept_clients": float(n)})

        eff_ct = max(1, min(self.ct, n - 2))
        cnt = np.zeros(n, dtype=np.float64)

        for key in layer_keys:
            layer_vecs = [
                u.param_delta[key].detach().to(dtype=torch.float32).cpu().flatten().numpy()
                for u in updates
            ]
            benign_lists: List[np.ndarray] = []
            scores = np.zeros(n, dtype=np.float64)

            for i in range(n):
                ddif = np.stack([layer_vecs[i] - layer_vecs[j] for j in range(n)], axis=0)
                norms = np.linalg.norm(ddif, axis=1)
                suspicious_idx = np.argsort(norms)[-eff_ct:]
                centroid_ids = [i] + suspicious_idx.tolist()
                init_centroids = ddif[centroid_ids]

                labels = KMeans(
                    n_clusters=len(centroid_ids),
                    init=init_centroids,
                    n_init=1,
                    random_state=self.seed,
                ).fit_predict(ddif)

                n_unique = len(np.unique(labels))
                score = float(calinski_harabasz_score(ddif, labels)) if 1 < n_unique < n else 0.0
                benign_lists.append(np.argwhere(labels == labels[i]).flatten())
                scores[i] = score

            effective_ids = np.argwhere(scores > 0).flatten()
            min_eff = max(1, int(n * 0.1))
            if len(effective_ids) < min_eff:
                effective_ids = np.argsort(-scores)[:min_eff]

            s_min, s_max = float(np.min(scores)), float(np.max(scores))
            norm_scores = (scores - s_min) / (s_max - s_min) if (s_max - s_min) > 1e-12 else np.ones_like(scores)
            for idx_client in effective_ids:
                for idx_b in benign_lists[int(idx_client)]:
                    cnt[int(idx_b)] += norm_scores[int(idx_client)]

        cnt_rank = np.argsort(-cnt)
        selected_ids: List[int] = cnt_rank[: max(1, math.ceil(n * 0.1))].tolist()

        if ctx.round_idx < self.warmup_rounds or int(n * self.vt) <= len(selected_ids):
            kept = [updates[i] for i in selected_ids]
            param_delta, buffers = F.aggregate_updates(kept, weight_by_samples=True, device=ctx.device)
            return DefenseOutput(
                param_delta=param_delta,
                buffers=buffers,
                metrics={"defense/kept_clients": float(len(kept))},
            )

        flat_vecs = [
            torch.cat(
                [u.param_delta[k].detach().to(device=ctx.device, dtype=torch.float32).flatten() for k in layer_keys]
            )
            for u in updates
        ]

        dif_tensor = _build_dif_set([flat_vecs[i] for i in selected_ids])
        vae = _train_vae(
            None,
            dif_tensor,
            self.vae_initial,
            ctx.device,
            self.vae_latent,
            self.vae_hidden,
        )

        target_count = min(n, max(len(selected_ids), int(n * self.vt)))
        step_count = max(1, math.ceil(n * self.v_step))
        while len(selected_ids) < target_count:
            dif_tensor = _build_dif_set([flat_vecs[i] for i in selected_ids])
            vae = _train_vae(
                vae,
                dif_tensor,
                self.vae_tuning,
                ctx.device,
                self.vae_latent,
                self.vae_hidden,
            )
            vae.eval()
            rest_ids = [i for i in range(n) if i not in selected_ids]
            if not rest_ids:
                break
            losses: List[float] = []
            base_vecs = [flat_vecs[i] for i in selected_ids]
            for idx in rest_ids:
                difs = _obtain_dif(base_vecs, flat_vecs[idx])
                if not difs:
                    losses.append(0.0)
                else:
                    losses.append(sum(vae.recon_prob(d) for d in difs) / float(len(difs)))
            add_n = min(step_count, target_count - len(selected_ids))
            rank_ = np.argsort(losses)[:add_n]
            selected_ids.extend([rest_ids[int(r)] for r in rank_])

        kept = [updates[i] for i in selected_ids]
        param_delta, buffers = F.aggregate_updates(kept, weight_by_samples=True, device=ctx.device)
        return DefenseOutput(
            param_delta=param_delta,
            buffers=buffers,
            metrics={"defense/kept_clients": float(len(kept))},
        )
