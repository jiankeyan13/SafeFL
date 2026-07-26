"""
Batman: official BadSP / AlphaEdit alignment (upload-stage).

Core action runs in poison_upload after sequential clean-then-poison training.
Alignment ops live in this file (one attack == one upload module file).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from core.attack.data.badnets import BadNetsAttack
from core.utils.registry import ATTACK_REGISTRY

_LAYER_LOG_LOCK = threading.Lock()
_logger = logging.getLogger(__name__)

# (score, layer_name)
LayerScore = Tuple[float, str]


def count_eligible_layers(state: Mapping[str, torch.Tensor]) -> int:
    """Count weight tensors eligible for Batman layer selection (ndim >= 2)."""
    return sum(
        1
        for param in state.values()
        if torch.is_tensor(param) and param.ndim >= 2
    )


def resolve_num_selected_layers(
    eligible_count: int,
    num_selected_layers: int,
    selected_layer_ratio: Optional[float],
) -> int:
    """Prefer ratio when set; otherwise use absolute count."""
    if selected_layer_ratio is not None:
        if not 0.0 < selected_layer_ratio <= 1.0:
            raise ValueError(
                f"selected_layer_ratio must be in (0, 1], got {selected_layer_ratio}"
            )
        if eligible_count <= 0:
            return 0
        # half-up: ResNet18 (21 eligible) * 0.5 -> 11
        k = int(math.floor(eligible_count * selected_layer_ratio + 0.5))
        return max(1, min(eligible_count, k))
    return max(0, min(eligible_count, int(num_selected_layers)))


@torch.no_grad()
def score_layers_by_deviation(
    poisoned_state: Mapping[str, torch.Tensor],
    initial_state: Mapping[str, torch.Tensor],
) -> List[LayerScore]:
    """Score weight tensors by Frobenius deviation from initial/global."""
    scores: List[LayerScore] = []
    for name, poisoned_param in poisoned_state.items():
        if not torch.is_tensor(poisoned_param) or poisoned_param.ndim < 2:
            continue
        if name not in initial_state:
            continue
        initial_param = initial_state[name]
        score = torch.norm(
            poisoned_param.detach().float() - initial_param.detach().float()
        ).item()
        scores.append((score, name))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores


def select_layers_by_deviation(
    poisoned_state: Mapping[str, torch.Tensor],
    initial_state: Mapping[str, torch.Tensor],
    number_of_layers: int,
) -> List[str]:
    """Select weight tensors with largest Frobenius deviation from initial/global."""
    if number_of_layers <= 0:
        return []
    scores = score_layers_by_deviation(poisoned_state, initial_state)
    return [name for _, name in scores[:number_of_layers]]


@torch.no_grad()
def _normalized_mean_abs_diff_score(
    local_param: torch.Tensor,
    global_param: torch.Tensor,
) -> float:
    """Official score: mean(abs(Wp-Wg) / sum(abs(Wp-Wg)))."""
    diff = torch.abs(local_param.detach().float() - global_param.detach().float())
    total = torch.sum(diff)
    if float(total.item()) == 0.0:
        return 0.0
    return float(torch.mean(diff / total).item())


@torch.no_grad()
def select_layers_by_diff_top(
    poisoned_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    select_len: int,
) -> List[str]:
    """
    Diff-top layer selection (first round only).

    Rank conv weights (ndim > 2) by normalized mean abs diff and take top-K.
    Linear layers (ndim == 2) are not considered.
    """
    if select_len <= 0:
        return []
    diff_mean: Dict[str, float] = {}

    for name, para in poisoned_state.items():
        if not torch.is_tensor(para) or para.ndim <= 2:
            continue
        if name not in global_state or not torch.is_tensor(global_state[name]):
            continue
        diff_mean[name] = _normalized_mean_abs_diff_score(para, global_state[name])
    top_names = sorted(diff_mean.items(), key=lambda item: item[1], reverse=True)[:select_len]
    return [key for key, _ in top_names]


@torch.no_grad()
def score_layers_by_diff_top(
    poisoned_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
) -> List[LayerScore]:
    """Score conv layers (ndim > 2) with normalized-mean abs diff (for logging)."""
    scores: List[LayerScore] = []
    for name, para in poisoned_state.items():
        if not torch.is_tensor(para) or para.ndim <= 2:
            continue
        if name not in global_state or not torch.is_tensor(global_state[name]):
            continue
        score = _normalized_mean_abs_diff_score(para, global_state[name])
        scores.append((score, name))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores


def dedupe_layer_names(names: Iterable[str]) -> List[str]:
    """Preserve first-seen order while removing duplicate layer names."""
    return list(dict.fromkeys(names))


def eligible_layer_names(state: Mapping[str, torch.Tensor]) -> List[str]:
    """Stable-ordered eligible layer names (ndim >= 2)."""
    return [
        name
        for name, param in state.items()
        if torch.is_tensor(param) and param.ndim >= 2
    ]


def score_layers_by_random(
    poisoned_state: Mapping[str, torch.Tensor],
    rng: random.Random,
) -> List[LayerScore]:
    """Random permutation of eligible layers; score is a dummy rank for logging."""
    names = eligible_layer_names(poisoned_state)
    rng.shuffle(names)
    n = len(names)
    # Higher score = earlier in the random permutation (selected first).
    return [(float(n - i), name) for i, name in enumerate(names)]


def select_layers_by_random(
    poisoned_state: Mapping[str, torch.Tensor],
    number_of_layers: int,
    rng: random.Random,
) -> List[str]:
    """Uniformly sample critical layers without replacement."""
    if number_of_layers <= 0:
        return []
    scores = score_layers_by_random(poisoned_state, rng)
    return [name for _, name in scores[:number_of_layers]]


def make_layer_selection_rng(
    *,
    base_seed: int,
    client_id: str,
    round_idx: Optional[int],
) -> random.Random:
    """Deterministic RNG per (seed, round, client) for reproducible random selection."""
    raw = f"{base_seed}|{client_id}|{round_idx if round_idx is not None else -1}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def rank_critical_layers(
    poisoned_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    strategy: str = "global_deviation",
    rng: Optional[random.Random] = None,
) -> List[LayerScore]:
    """Return all eligible layers ranked by the chosen strategy (desc)."""
    if strategy == "diff_top":
        return score_layers_by_diff_top(poisoned_state, global_state)
    if strategy == "global_deviation":
        return score_layers_by_deviation(poisoned_state, global_state)
    if strategy == "random":
        if rng is None:
            raise ValueError("layer_selection='random' requires an RNG")
        return score_layers_by_random(poisoned_state, rng)
    raise ValueError(
        f"Unknown layer_selection strategy: {strategy!r}. "
        "Expected 'diff_top', 'global_deviation', or 'random'."
    )


def select_critical_layers(
    poisoned_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    number_of_layers: int,
    strategy: str = "global_deviation",
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Dispatch layer selection by strategy name (deduped for alignment)."""
    if number_of_layers <= 0:
        return []
    if strategy == "diff_top":
        return dedupe_layer_names(
            select_layers_by_diff_top(poisoned_state, global_state, number_of_layers)
        )
    ranked = rank_critical_layers(
        poisoned_state,
        global_state,
        strategy=strategy,
        rng=rng,
    )
    return [name for _, name in ranked[:number_of_layers]]


def _default_layer_log_path() -> str:
    try:
        from hydra.core.hydra_config import HydraConfig

        out_dir = HydraConfig.get().runtime.output_dir
        if out_dir:
            return os.path.join(out_dir, "batman_selected_layers.jsonl")
    except Exception:
        pass
    return os.path.abspath("batman_selected_layers.jsonl")


def append_layer_selection_record(path: str, record: Mapping[str, Any]) -> None:
    """Append one JSONL record for selected critical layers."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _LAYER_LOG_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def shape_restruct(param: torch.Tensor) -> torch.Tensor:
    """Official BadSP reshape: 4D [out,in,h,w] -> [out*h, in*w]."""
    if param.ndim > 2:
        return param.reshape(param.shape[0] * param.shape[2], param.shape[1] * param.shape[3])
    return param


def shape_restore(param: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """Restore a reshaped matrix to original parameter shape."""
    if len(original_shape) > 2:
        return param.reshape(original_shape)
    return param


@torch.no_grad()
def split_rank(weight_mat: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Truncated SVD: Wfirst (top rank) and Wlast = W - Wfirst."""
    effective_rank = min(int(rank), weight_mat.shape[0], weight_mat.shape[1])
    u, s, vh = torch.linalg.svd(weight_mat, full_matrices=False)
    u_first = u[:, :effective_rank]
    s_first = torch.diag(s[:effective_rank])
    vh_first = vh[:effective_rank, :]
    w_first = u_first @ s_first @ vh_first
    w_last = weight_mat - w_first
    return w_first, w_last


@torch.no_grad()
def compute_perturbation(
    w_malicious: torch.Tensor,
    w_standard: torch.Tensor,
    lambda_reg: float,
) -> torch.Tensor:
    """Official AlphaEdit left-null-space regularized solve on Wfirst."""
    u, s, _vh = torch.linalg.svd(w_malicious, full_matrices=False)
    threshold = 1e-6
    small_singular_indices = (s < threshold).nonzero(as_tuple=True)[0]
    zero_space_basis = u[:, small_singular_indices]
    p_m = zero_space_basis @ zero_space_basis.transpose(0, 1)

    residual = w_standard - w_malicious
    eye = torch.eye(w_malicious.shape[0], device=w_malicious.device, dtype=w_malicious.dtype)
    h = p_m @ (w_malicious @ w_malicious.transpose(0, 1)) + float(lambda_reg) * eye
    return torch.linalg.solve(h, p_m @ residual)


@torch.no_grad()
def _align_layer(
    poisoned_weight: torch.Tensor,
    clean_weight: torch.Tensor,
    global_weight: torch.Tensor,
    rank: int,
    lamda_reg: float,
    beta_reg: float,
) -> torch.Tensor:
    """Official AlphaEdit alignment for one weight tensor."""
    if poisoned_weight.ndim < 2:
        return poisoned_weight.clone()

    if poisoned_weight.shape != clean_weight.shape:
        raise ValueError("poisoned_weight and clean_weight shape mismatch")
    if poisoned_weight.shape != global_weight.shape:
        raise ValueError("poisoned_weight and global_weight shape mismatch")
    if rank <= 0:
        raise ValueError("rank must be > 0")
    if lamda_reg < 0 or beta_reg < 0:
        raise ValueError("lamda_reg and beta_reg must be >= 0")

    original_shape = poisoned_weight.shape
    original_dtype = poisoned_weight.dtype

    wp = shape_restruct(poisoned_weight.detach().float())
    wc = shape_restruct(clean_weight.detach().float())
    wg = shape_restruct(global_weight.detach().float())

    w_first, w_last = split_rank(wp, rank)
    delta_tg = compute_perturbation(w_first, wg, lamda_reg)
    delta_tc = compute_perturbation(w_first, wc, beta_reg)
    aligned = w_first + 0.5 * (delta_tg + delta_tc) + w_last
    return shape_restore(aligned, original_shape).to(original_dtype)


@torch.no_grad()
def postprocess(
    poisoned_state: Mapping[str, torch.Tensor],
    clean_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    selected_layers: Iterable[str],
    rank: int,
    lamda_reg: float,
    beta_reg: float,
) -> OrderedDict[str, torch.Tensor]:
    """Apply official AlphaEdit on selected layers; leave others unchanged."""
    selected = set(selected_layers)
    aligned_state: OrderedDict[str, torch.Tensor] = OrderedDict()

    for name, poisoned_param in poisoned_state.items():
        if name not in clean_state:
            raise KeyError(f"clean_state missing parameter: {name}")
        if name not in global_state:
            raise KeyError(f"global_state missing parameter: {name}")

        if name not in selected or not torch.is_tensor(poisoned_param) or poisoned_param.ndim < 2:
            aligned_state[name] = (
                poisoned_param.clone() if torch.is_tensor(poisoned_param) else poisoned_param
            )
            continue

        aligned_state[name] = _align_layer(
            poisoned_weight=poisoned_param,
            clean_weight=clean_state[name],
            global_weight=global_state[name],
            rank=rank,
            lamda_reg=lamda_reg,
            beta_reg=beta_reg,
        )

    return aligned_state


@ATTACK_REGISTRY.register("batman")
class BatmanAttack:
    """
    Upload-stage Batman attack (official BadSP / AlphaEdit).

    - poison_dataset: BadNets trigger poisoning for the malicious train branch / ASR eval
    - poison_upload: left-null-space AlphaEdit using cached clean reference state
    """

    # Assigned after BatmanClient is importable (avoids circular import at class body time).
    client_class = None

    def __init__(
        self,
        target_label: int = 5,
        poison_ratio: float = 0.5,
        patch_size: int = 5,
        patch_value: float = 1.0,
        patch_location: str = "bottom_right",
        seed: Optional[int] = None,
        rank: int = 4,
        regularization: Optional[float] = None,
        lamda_reg: Optional[float] = None,
        beta_reg: Optional[float] = None,
        num_selected_layers: int = 5,
        selected_layer_ratio: Optional[float] = None,
        stats_top_layers: Optional[int] = None,
        stats_layer_ratio: Optional[float] = None,
        layer_selection: str = "global_deviation",
        layer_selection_seed: Optional[int] = None,
        log_selected_layers: bool = True,
        layer_selection_log_path: Optional[str] = None,
    ):
        self._badnets = BadNetsAttack(
            target_label=target_label,
            poison_ratio=poison_ratio,
            patch_size=patch_size,
            patch_value=patch_value,
            patch_location=patch_location,
            seed=seed,
        )
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.seed = seed
        self.rank = rank
        # Legacy single regularization overrides both when lamda/beta are omitted.
        if lamda_reg is None and beta_reg is None and regularization is not None:
            self.lamda_reg = float(regularization)
            self.beta_reg = float(regularization)
        else:
            self.lamda_reg = 20.0 if lamda_reg is None else float(lamda_reg)
            self.beta_reg = 15.0 if beta_reg is None else float(beta_reg)
        self.regularization = regularization
        self.num_selected_layers = num_selected_layers
        self.selected_layer_ratio = selected_layer_ratio
        self.stats_top_layers = stats_top_layers
        self.stats_layer_ratio = stats_layer_ratio
        self.layer_selection = layer_selection
        self.layer_selection_seed = (
            layer_selection_seed if layer_selection_seed is not None else (seed if seed is not None else 0)
        )
        self.log_selected_layers = log_selected_layers
        self.layer_selection_log_path = layer_selection_log_path
        self._clean_states: Dict[str, Dict[str, torch.Tensor]] = {}

    def _resolve_log_path(self) -> str:
        if self.layer_selection_log_path:
            return self.layer_selection_log_path
        base = _default_layer_log_path()
        root, ext = os.path.splitext(base)
        return f"{root}_{self.layer_selection}{ext or '.jsonl'}"

    def _resolve_stats_top(self, eligible_count: int, num_align: int) -> int:
        """How many top-ranked layers to record for statistics (not necessarily aligned)."""
        if self.stats_layer_ratio is not None:
            k = resolve_num_selected_layers(
                eligible_count=eligible_count,
                num_selected_layers=0,
                selected_layer_ratio=self.stats_layer_ratio,
            )
        elif self.stats_top_layers is not None:
            k = max(0, min(eligible_count, int(self.stats_top_layers)))
        else:
            k = num_align
        return max(num_align, k)

    def _log_selected_layers(
        self,
        *,
        client_id: str,
        round_idx: Optional[int],
        eligible_count: int,
        num_align: int,
        num_stats: int,
        aligned_scores: Sequence[LayerScore],
        stats_scores: Sequence[LayerScore],
        ranked_all: Sequence[LayerScore],
        selected_layers_raw: Optional[Sequence[str]] = None,
    ) -> None:
        if not self.log_selected_layers:
            return
        aligned_names = [name for _, name in aligned_scores]
        raw_names = list(selected_layers_raw) if selected_layers_raw is not None else aligned_names
        record = {
            "round": round_idx,
            "client_id": str(client_id),
            "strategy": self.layer_selection,
            "selected_layer_ratio": self.selected_layer_ratio,
            "stats_layer_ratio": self.stats_layer_ratio,
            "num_selected_layers_cfg": self.num_selected_layers,
            "stats_top_layers_cfg": self.stats_top_layers,
            "layer_selection_seed": self.layer_selection_seed,
            "eligible_count": eligible_count,
            "num_align": num_align,
            "num_stats": num_stats,
            # official raw list may contain duplicates
            "selected_layers_raw": raw_names,
            # layers actually used for SVD alignment (deduped)
            "aligned_layers": aligned_names,
            "aligned_scores": [
                {"name": name, "score": score} for score, name in aligned_scores
            ],
            # top-N for statistics (may be larger than aligned)
            "stats_top_layers": [name for _, name in stats_scores],
            "stats_top_scores": [
                {"name": name, "score": score} for score, name in stats_scores
            ],
            # backward-compatible aliases (= aligned)
            "num_selected": len(aligned_names),
            "selected_layers": aligned_names,
            "selected_scores": [
                {"name": name, "score": score} for score, name in aligned_scores
            ],
            "ranked_layers": [name for _, name in ranked_all],
        }
        path = self._resolve_log_path()
        append_layer_selection_record(path, record)
        _logger.info(
            "Batman align=%d stats_top=%d/%d [%s] round=%s client=%s raw=%s align=%s",
            len(aligned_names),
            num_stats,
            eligible_count,
            self.layer_selection,
            round_idx,
            client_id,
            raw_names,
            aligned_names,
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
        return self._badnets.poison_dataset(
            dataset,
            mode=mode,
            split=split,
            client_id=client_id,
            round_idx=round_idx,
            **kwargs,
        )

    def cache_clean_state(self, client_id: str, state: Mapping[str, torch.Tensor]) -> None:
        """Store clean-branch weights for a client (overwrites prior round)."""
        self._clean_states[str(client_id)] = {
            k: v.detach().clone() if torch.is_tensor(v) else v for k, v in state.items()
        }

    def pop_clean_state(self, client_id: str) -> Dict[str, torch.Tensor]:
        key = str(client_id)
        if key not in self._clean_states:
            raise KeyError(f"No clean_state cached for client {client_id}")
        return self._clean_states.pop(key)

    def poison_upload(
        self,
        update: Dict[str, torch.Tensor],
        *,
        initial_weights: Dict[str, torch.Tensor],
        client_id: Optional[str] = None,
        round_idx: Optional[int] = None,
        num_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Align poisoned absolute weights, then convert back to delta vs global."""
        del num_samples  # unused; kept for MaliciousClient hook signature
        kwargs.clear()

        if client_id is None:
            raise ValueError("BatmanAttack.poison_upload requires client_id")

        clean_state = self.pop_clean_state(client_id)
        poisoned_state: Dict[str, torch.Tensor] = {}
        for k, init_v in initial_weights.items():
            if k in update and torch.is_tensor(update[k]) and torch.is_tensor(init_v):
                poisoned_state[k] = init_v + update[k]
            else:
                poisoned_state[k] = update[k] if k in update else init_v

        eligible_count = count_eligible_layers(poisoned_state)
        # Alignment count: fixed num_selected_layers, unless selected_layer_ratio is set.
        num_align = resolve_num_selected_layers(
            eligible_count=eligible_count,
            num_selected_layers=self.num_selected_layers,
            selected_layer_ratio=self.selected_layer_ratio,
        )
        num_stats = self._resolve_stats_top(eligible_count, num_align)
        rng = None
        if self.layer_selection == "random":
            rng = make_layer_selection_rng(
                base_seed=int(self.layer_selection_seed),
                client_id=str(client_id),
                round_idx=round_idx,
            )

        score_by_name = {
            name: score
            for score, name in rank_critical_layers(
                poisoned_state=poisoned_state,
                global_state=initial_weights,
                strategy=self.layer_selection,
                rng=rng,
            )
        }
        ranked = sorted(score_by_name.items(), key=lambda x: x[1], reverse=True)
        ranked_scores: List[LayerScore] = [(score, name) for name, score in ranked]

        selected_raw: List[str]
        if self.layer_selection == "diff_top":
            selected_raw = select_layers_by_diff_top(
                poisoned_state, initial_weights, num_align
            )
            selected = dedupe_layer_names(selected_raw)
        else:
            selected = [name for _, name in ranked_scores[:num_align]]
            selected_raw = selected

        aligned_scores = [
            (score_by_name.get(name, 0.0), name) for name in selected
        ]
        stats_scores = ranked_scores[:num_stats]
        self._log_selected_layers(
            client_id=client_id,
            round_idx=round_idx,
            eligible_count=eligible_count,
            num_align=num_align,
            num_stats=num_stats,
            aligned_scores=aligned_scores,
            stats_scores=stats_scores,
            ranked_all=ranked_scores,
            selected_layers_raw=selected_raw,
        )
        aligned_state = postprocess(
            poisoned_state=poisoned_state,
            clean_state=clean_state,
            global_state=initial_weights,
            selected_layers=selected,
            rank=self.rank,
            lamda_reg=self.lamda_reg,
            beta_reg=self.beta_reg,
        )

        return {
            k: aligned_state[k] - initial_weights[k]
            for k in aligned_state
            if k in initial_weights and torch.is_tensor(aligned_state[k])
        }


# Late bind client class to avoid import cycles with BatmanClient.
from core.client.batman_client import BatmanClient  # noqa: E402

BatmanAttack.client_class = BatmanClient
