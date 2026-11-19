"""SparAlignFL: locate-concentrate federated backdoor attack.

This module implements the attack primitives from scratch:
  1) contrastive scoring on weight layers only, then expand with the previous
     weight and matching BN (BN is in L_c but does not count toward K)
  2) a single global coordinate mask over L_c learnable tensors, blocking the
     largest benign-update magnitudes at mask_ratio of those coordinates.
     BN running stats are in L_c but do not participate in ranking or masking
  3) LGA-style layer-wise projection after each Stage-II epoch, applied only to
     non-critical tensors so L_c keeps its concentrated backdoor update
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from core.attack.data.poison_dataset import PoisonedDatasetWrapper
from core.attack.data.triggers import PatchTrigger
from core.utils.registry import ATTACK_REGISTRY

LayerScore = Tuple[float, str]
_CONV_TO_BN_RE = re.compile(r"(^|\.)conv(\d+)$")


@torch.no_grad()
def project_layerwise(
    global_state: Mapping[str, torch.Tensor],
    current_state: Mapping[str, torch.Tensor],
    prev_global_delta: Optional[Mapping[str, torch.Tensor]],
    eps: float = 1e-8,
    skip_keys: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Project non-critical tensors' deltas to the previous global delta norm.

    Keys in ``skip_keys`` (typically L_c) are copied as-is and are not scaled.
    """
    skipped = set(skip_keys or ())
    projected: Dict[str, torch.Tensor] = {}
    for key, current in current_state.items():
        if not torch.is_tensor(current):
            projected[key] = current
            continue

        if key in skipped:
            projected[key] = current.clone()
            continue

        if key.endswith("num_batches_tracked"):
            if key in global_state and torch.is_tensor(global_state[key]):
                projected[key] = global_state[key].clone()
            else:
                projected[key] = current.clone()
            continue

        if (
            prev_global_delta is None
            or key not in prev_global_delta
            or key not in global_state
            or not torch.is_tensor(global_state[key])
            or not torch.is_tensor(prev_global_delta[key])
        ):
            projected[key] = current.clone()
            continue

        base = global_state[key]
        current_aligned = current.to(device=base.device, dtype=base.dtype)
        delta = current_aligned - base
        reference = prev_global_delta[key].to(device=base.device, dtype=torch.float32)
        delta_norm = torch.norm(delta.float())
        scale = min(
            1.0,
            float(torch.norm(reference).item()) / (float(delta_norm.item()) + float(eps)),
        )
        projected[key] = (delta * scale + base).to(dtype=current.dtype)

    return projected


def _is_tensor(value: object) -> bool:
    return torch.is_tensor(value)


def is_floating_tensor(value: object) -> bool:
    return _is_tensor(value) and torch.is_floating_point(value)


def module_prefix(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else name


def is_bn_module(prefix: str, state: Mapping[str, torch.Tensor]) -> bool:
    last = prefix.split(".")[-1]
    if last.startswith("bn"):
        return True
    return f"{prefix}.running_mean" in state


def is_bn_statistic_key(name: str) -> bool:
    """BN running buffers: not learnable, must not enter mask ranking."""
    return name.endswith(".running_mean") or name.endswith(".running_var") or name.endswith(
        ".num_batches_tracked"
    )


def is_weight_layer_key(name: str, state: Mapping[str, torch.Tensor]) -> bool:
    """Scoreable layer: conv/linear `.weight` with ndim >= 2, excluding BN."""
    if not name.endswith(".weight"):
        return False
    value = state.get(name)
    if not is_floating_tensor(value) or value.ndim < 2:
        return False
    return not is_bn_module(module_prefix(name), state)


def eligible_layer_names(state: Mapping[str, torch.Tensor]) -> List[str]:
    return [name for name in state if is_weight_layer_key(name, state)]


def _module_keys(prefix: str, state: Mapping[str, torch.Tensor]) -> List[str]:
    keys = []
    for name in state:
        if name == prefix or name.startswith(prefix + "."):
            keys.append(name)
    return keys


def previous_weight_key(weight_key: str, state: Mapping[str, torch.Tensor]) -> Optional[str]:
    """Previous weight in network order, including downsample 1x1 convs."""
    ordered = eligible_layer_names(state)
    try:
        index = ordered.index(weight_key)
    except ValueError:
        return None
    if index <= 0:
        return None
    return ordered[index - 1]


def weight_module_keys(weight_key: str, state: Mapping[str, torch.Tensor]) -> List[str]:
    """Tensors of the weight module itself, without the paired BN."""
    prefix = module_prefix(weight_key)
    keys: List[str] = []
    for name in _module_keys(prefix, state):
        if is_bn_module(module_prefix(name), state):
            continue
        keys.append(name)
    return keys


def paired_bn_prefix(weight_key: str, state: Mapping[str, torch.Tensor]) -> Optional[str]:
    """Matching BN module: convN -> bnN, shortcut.0 -> shortcut.1. Linear has none."""
    prefix = module_prefix(weight_key)
    if prefix.endswith("shortcut.0"):
        cand = prefix[:-1] + "1"
        return cand if is_bn_module(cand, state) else None
    if _CONV_TO_BN_RE.search(prefix) is None:
        return None
    cand = _CONV_TO_BN_RE.sub(r"\1bn\2", prefix)
    return cand if is_bn_module(cand, state) else None


def companion_bn_keys(weight_key: str, state: Mapping[str, torch.Tensor]) -> List[str]:
    """All tensors of the matching BN. Empty if the weight has no BN."""
    bn_prefix = paired_bn_prefix(weight_key, state)
    if bn_prefix is None:
        return []
    return [name for name in _module_keys(bn_prefix, state) if name in state]


def expand_scored_weight(weight_key: str, state: Mapping[str, torch.Tensor]) -> List[str]:
    """A score-picked weight always brings its predecessor and both matching BNs.

    BN tensors enter L_c but are ignored by K.
    """
    keys: List[str] = []
    seen = set()
    prev = previous_weight_key(weight_key, state)
    for candidate in (prev, weight_key):
        if candidate is None:
            continue
        for name in (*weight_module_keys(candidate, state), *companion_bn_keys(candidate, state)):
            if name in seen or name not in state:
                continue
            seen.add(name)
            keys.append(name)
    return keys


def _selected_weight_count(keys: Iterable[str], state: Mapping[str, torch.Tensor]) -> int:
    """K counts weight layers only, not BN."""
    return sum(1 for name in keys if is_weight_layer_key(name, state))


@torch.no_grad()
def score_contrastive_layers(
    global_state: Mapping[str, torch.Tensor],
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
    eps: float = 1e-8,
) -> List[LayerScore]:
    """S_l = ||Δm - Δb||_F^2 / (||Δm||_F^2 + ||Δb||_F^2 + eps), weights only."""
    scores: List[LayerScore] = []
    for name in eligible_layer_names(global_state):
        global_param = global_state[name]
        benign_param = benign_state.get(name)
        malicious_param = malicious_state.get(name)
        if not is_weight_layer_key(name, benign_state) or not is_weight_layer_key(name, malicious_state):
            continue
        if not is_floating_tensor(benign_param) or not is_floating_tensor(malicious_param):
            continue

        base = global_param.detach().float()
        delta_b = benign_param.detach().to(device=base.device, dtype=torch.float32) - base
        delta_m = malicious_param.detach().to(device=base.device, dtype=torch.float32) - base
        numerator = (delta_m - delta_b).square().sum()
        denominator = delta_m.square().sum() + delta_b.square().sum() + float(eps)
        score = float((numerator / denominator).item())
        scores.append((score, name))

    scores.sort(key=lambda item: item[0], reverse=True)
    return scores


def select_critical_layers(
    ranked: Sequence[LayerScore],
    num_critical_layers: int,
    state: Mapping[str, torch.Tensor],
    selection_strategy: str = "expanded_topk",
    random_seed: Optional[int] = None,
) -> List[str]:
    """Select critical layers using the configured layer-selection strategy.

    ``expanded_topk`` preserves the original SparAlignFL behavior: each picked
    weight also includes its predecessor and matching BNs. ``random_k`` and
    ``bottom_k`` select only scoreable weight keys, without expansion.

    Example: pick layer4.1.conv2.weight -> also layer4.1.conv1.weight and both BNs.
    If the next score is layer4.1.conv1.weight, expand it again and pull in
    the weight in front of conv1 (possibly a downsample 1x1) plus its BN.
    K counts weight layers only; BN does not occupy a slot.
    """
    if num_critical_layers <= 0 or not ranked:
        return []

    strategy = str(selection_strategy).lower().replace("-", "_")
    if strategy in {"topk", "top_k", "score_topk"}:
        strategy = "weight_topk"
    if strategy in {"expanded_top_k", "expanded_topk", "default"}:
        strategy = "expanded_topk"

    if strategy in {"random_k", "bottom_k", "weight_topk"}:
        weight_ranked = [
            (score, weight_key)
            for score, weight_key in ranked
            if is_weight_layer_key(weight_key, state)
        ]
        count = min(int(num_critical_layers), len(weight_ranked))
        if count <= 0:
            return []
        if strategy == "random_k":
            rng = random.Random(0 if random_seed is None else int(random_seed))
            return [
                weight_key
                for _, weight_key in rng.sample(weight_ranked, count)
            ]
        if strategy == "bottom_k":
            return [weight_key for _, weight_key in weight_ranked[-count:]]
        return [weight_key for _, weight_key in weight_ranked[:count]]

    if strategy != "expanded_topk":
        raise ValueError(
            f"Unknown SparAlignFL layer_selection_strategy: {selection_strategy!r}. "
            "Expected 'expanded_topk', 'random_k', 'bottom_k', or 'weight_topk'."
        )

    selected_keys: List[str] = []
    selected_set = set()
    for _, weight_key in ranked:
        if not is_weight_layer_key(weight_key, state):
            continue
        for name in expand_scored_weight(weight_key, state):
            if name in selected_set or name not in state:
                continue
            selected_keys.append(name)
            selected_set.add(name)
        if _selected_weight_count(selected_keys, state) >= int(num_critical_layers):
            break
    return selected_keys


def summarize_layer_scores(
    ranked: Sequence[LayerScore],
    critical_layers: Iterable[str],
) -> Dict[str, Any]:
    """Make the contrastive score ranking serializable for client telemetry.

    Scores are computed for weight layers only.  ``critical_layers`` may also
    contain predecessor weights and BN tensors, so ``selected`` is evaluated
    against the expanded set while the ranking itself remains weight-only.
    """
    critical_set = set(critical_layers)
    layers = [
        {
            "rank": rank,
            "name": name,
            "score": float(score),
            "selected": name in critical_set,
        }
        for rank, (score, name) in enumerate(ranked)
    ]
    return {
        "attack": "sparalignfl",
        "layers": layers,
        "n_scored": len(layers),
        "selected_weight_layers": [
            item["name"] for item in layers if item["selected"]
        ],
    }


@torch.no_grad()
def build_benign_active_masks(
    global_state: Mapping[str, torch.Tensor],
    benign_state: Mapping[str, torch.Tensor],
    critical_layers: Iterable[str],
    mask_ratio: float,
) -> Dict[str, torch.Tensor]:
    """Global keep-mask over L_c learnable tensors.

    Concatenate |Δb| of every rankable L_c tensor, then block the top
    ``mask_ratio`` coordinates jointly. 1 keeps the coordinate, 0 blocks it.
    Ranking is not done independently per layer. BN running stats are skipped.
    """
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")

    entries = []
    flat_scores = []
    for name in critical_layers:
        if is_bn_statistic_key(name):
            continue
        global_param = global_state.get(name)
        benign_param = benign_state.get(name)
        if not is_floating_tensor(global_param) or not is_floating_tensor(benign_param):
            continue

        reference = benign_param.detach()
        delta_abs = (
            reference.to(dtype=torch.float32)
            - global_param.detach().to(device=reference.device, dtype=torch.float32)
        ).abs().reshape(-1)
        entries.append((name, reference, delta_abs.numel()))
        flat_scores.append(delta_abs.detach().to(device="cpu", dtype=torch.float32))

    if not entries:
        return {}

    scores = torch.cat(flat_scores)
    blocked = torch.zeros(scores.numel(), dtype=torch.bool)
    num_masked = min(scores.numel(), int(scores.numel() * mask_ratio))
    if num_masked > 0:
        blocked[torch.topk(scores, k=num_masked, largest=True, sorted=False).indices] = True

    masks: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, reference, numel in entries:
        keep = ~blocked[offset:offset + numel]
        masks[name] = keep.reshape_as(reference).to(device=reference.device, dtype=reference.dtype)
        offset += numel
    return masks


def summarize_layer_masks(masks: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    """Per-layer blocked-coordinate counts. Mask 0 = blocked, 1 = kept."""
    layers: List[Dict[str, Any]] = []
    n_total = 0
    n_masked = 0
    for name, mask in masks.items():
        if not torch.is_tensor(mask):
            continue
        total = int(mask.numel())
        blocked = int((mask == 0).sum().item())
        layers.append(
            {
                "name": name,
                "n_total": total,
                "n_masked": blocked,
                "ratio": (blocked / total) if total > 0 else 0.0,
            }
        )
        n_total += total
        n_masked += blocked
    return {
        "attack": "sparalignfl",
        "layers": layers,
        "n_total": n_total,
        "n_masked": n_masked,
        "ratio": (n_masked / n_total) if n_total > 0 else 0.0,
    }


@torch.no_grad()
def concentrated_upload_delta(
    global_state: Mapping[str, torch.Tensor],
    concentrated_state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Upload the Stage-II model delta as-is."""
    upload: Dict[str, torch.Tensor] = {}
    for name, global_param in global_state.items():
        concentrated = concentrated_state.get(name)
        if _is_tensor(concentrated):
            upload[name] = concentrated.detach() - global_param.detach().to(
                device=concentrated.device, dtype=concentrated.dtype
            )
        else:
            upload[name] = torch.zeros_like(global_param)
    return upload


@ATTACK_REGISTRY.register("sparalignfl")
class SparAlignFLAttack:
    """Locate-concentrate backdoor attack with a BadNets trigger."""

    client_class = None

    def __init__(
        self,
        target_label: int = 5,
        poison_ratio: float = 0.5,
        patch_size: int = 5,
        patch_value: float = 1.0,
        patch_location: str = "bottom_right",
        seed: Optional[int] = None,
        num_critical_layers: int = 5,
        mask_ratio: float = 0.1,
        probe_epochs: int = 3,
        probe_poison_ratio: float = 1.0,
        eps: float = 1e-8,
        process_critical_layers: bool = True,
        process_non_critical_layers: bool = True,
        layer_selection_strategy: str = "expanded_topk",
    ) -> None:
        if num_critical_layers < 0:
            raise ValueError(f"num_critical_layers must be >= 0, got {num_critical_layers}")
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
        if probe_epochs < 1:
            raise ValueError(f"probe_epochs must be >= 1, got {probe_epochs}")
        if probe_poison_ratio <= 0.0:
            raise ValueError(f"probe_poison_ratio must be > 0, got {probe_poison_ratio}")

        self.target_label = int(target_label)
        self.poison_ratio = float(poison_ratio)
        self.patch_size = int(patch_size)
        self.patch_value = float(patch_value)
        self.patch_location = str(patch_location)
        self.seed = seed
        self.num_critical_layers = int(num_critical_layers)
        self.mask_ratio = float(mask_ratio)
        self.probe_epochs = int(probe_epochs)
        self.probe_poison_ratio = float(probe_poison_ratio)
        self.eps = float(eps)
        self.process_critical_layers = bool(process_critical_layers)
        self.process_non_critical_layers = bool(process_non_critical_layers)
        normalized_strategy = str(layer_selection_strategy).lower().replace("-", "_")
        if normalized_strategy in {"topk", "top_k", "score_topk"}:
            normalized_strategy = "weight_topk"
        if normalized_strategy in {"expanded_top_k", "default"}:
            normalized_strategy = "expanded_topk"
        if normalized_strategy not in {
            "expanded_topk",
            "random_k",
            "bottom_k",
            "weight_topk",
        }:
            raise ValueError(
                f"Unknown SparAlignFL layer_selection_strategy: {layer_selection_strategy!r}. "
                "Expected 'expanded_topk', 'random_k', 'bottom_k', or 'weight_topk'."
            )
        self.layer_selection_strategy = normalized_strategy
        self.last_layer_mask_stats: Dict[str, Any] = {}
        self.last_layer_score_stats: Dict[str, Any] = {}
        self._trigger = PatchTrigger(
            patch_size=self.patch_size,
            patch_value=self.patch_value,
            location=self.patch_location,
        )

    def _wrap_dataset(
        self,
        dataset: Dataset,
        *,
        poison_ratio: float,
        mode: str,
        seed: Optional[int],
        return_original_label: bool = False,
    ) -> Dataset:
        return PoisonedDatasetWrapper(
            original_dataset=dataset,
            trigger_transform=self._trigger,
            target_label=self.target_label,
            poison_ratio=poison_ratio,
            mode=mode,
            seed=seed,
            return_original_label=return_original_label,
        )

    def poison_dataset(
        self,
        dataset: Dataset,
        mode: str,
        split: str = "",
        client_id: Optional[str] = None,
        round_idx: Optional[int] = None,
        **kwargs,
    ) -> Dataset:
        del split, client_id, round_idx
        return_original_label = bool(kwargs.get("return_original_label", False))
        seed = kwargs["seed"] if "seed" in kwargs else self.seed
        return self._wrap_dataset(
            dataset,
            poison_ratio=self.poison_ratio,
            mode=mode,
            seed=seed,
            return_original_label=return_original_label,
        )

    def build_probe_poison_dataset(
        self,
        dataset: Dataset,
        *,
        client_id: Optional[str] = None,
        round_idx: Optional[int] = None,
    ) -> Dataset:
        """100% triggered dataset used only for contrastive layer localization."""
        del client_id, round_idx
        return self._wrap_dataset(
            dataset,
            poison_ratio=self.probe_poison_ratio,
            mode="train",
            seed=self.seed,
        )

    def locate_critical_layers(
        self,
        global_state: Mapping[str, torch.Tensor],
        benign_state: Mapping[str, torch.Tensor],
        malicious_state: Mapping[str, torch.Tensor],
        *,
        client_id: Optional[str] = None,
        round_idx: Optional[int] = None,
    ) -> Tuple[List[str], List[LayerScore]]:
        ranked = score_contrastive_layers(
            global_state=global_state,
            benign_state=benign_state,
            malicious_state=malicious_state,
            eps=self.eps,
        )
        random_seed: Optional[int] = None
        if self.layer_selection_strategy == "random_k":
            raw_seed = (
                f"{int(self.seed) if self.seed is not None else 0}|"
                f"{client_id if client_id is not None else ''}|"
                f"{round_idx if round_idx is not None else -1}"
            )
            random_seed = int(hashlib.md5(raw_seed.encode("utf-8")).hexdigest()[:16], 16)
        selected = select_critical_layers(
            ranked,
            self.num_critical_layers,
            global_state,
            selection_strategy=self.layer_selection_strategy,
            random_seed=random_seed,
        )
        self.last_layer_score_stats = summarize_layer_scores(ranked, selected)
        self.last_layer_score_stats["selection_strategy"] = self.layer_selection_strategy
        return selected, ranked

    def build_masks(
        self,
        global_state: Mapping[str, torch.Tensor],
        benign_state: Mapping[str, torch.Tensor],
        critical_layers: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        masks = build_benign_active_masks(
            global_state=global_state,
            benign_state=benign_state,
            critical_layers=critical_layers,
            mask_ratio=self.mask_ratio,
        )
        self.last_layer_mask_stats = summarize_layer_masks(masks)
        return masks

    def project_layerwise(
        self,
        global_state: Mapping[str, torch.Tensor],
        current_state: Mapping[str, torch.Tensor],
        prev_global_delta: Optional[Mapping[str, torch.Tensor]],
        skip_keys: Optional[Iterable[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Project non-critical layers after a local epoch; skip L_c."""
        return project_layerwise(
            global_state=global_state,
            current_state=current_state,
            prev_global_delta=prev_global_delta,
            eps=self.eps,
            skip_keys=skip_keys,
        )

    def poison_upload(
        self,
        update: Mapping[str, torch.Tensor],
        initial_weights: Mapping[str, torch.Tensor],
        client_id: Optional[str] = None,
        round_idx: Optional[int] = None,
        num_samples: Optional[int] = None,
        concentrated_state: Optional[Mapping[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        del update, round_idx, num_samples, client_id, kwargs
        if concentrated_state is None:
            raise RuntimeError("SparAlignFL.poison_upload requires concentrated_state")
        return concentrated_upload_delta(
            global_state=initial_weights,
            concentrated_state=concentrated_state,
        )


from core.client.sparalignfl_client import SparAlignFLClient  # noqa: E402

SparAlignFLAttack.client_class = SparAlignFLClient
