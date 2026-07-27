"""
LP-Attack: Layer-wise Poisoning via Backdoor-Critical (BC) layers.

Flow (no adaptive layer-count search):
1) Reference long-train from Wg on the train split only (excludes held-out val):
   benign until clean train acc >= ref_benign_acc_threshold (eval every epoch),
   then malicious until ASR >= ref_malicious_asr_threshold (eval every epoch;
   ASR on held-out poison val). FLS + hybrid warm-start greedy BLS on atomic
   modules (Conv+BN / Linear) select BC groups (BSR on held-out val).
   Craft/substitution swaps weight/bias only; BN running stats stay benign.
2) Local benign + local malicious train from Wg for local_ep on full local data
   (clean full / poison full).
3) Craft with lambda=1: start from local benign, replace BC groups with the
   local malicious weights.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from core.attack.data.badnets import BadNetsAttack
from core.utils.registry import ATTACK_REGISTRY

_LAYER_LOG_LOCK = threading.Lock()

# Half-bin default for N_val=125 (resolution 1/125=0.008): eps=0.005
# so a 1-sample BSR change is Neg/Pos, and only exact-0 deltas are plateau.
DEFAULT_BLS_EPS = 0.005

# (delta_bsr, group_name); more negative => more backdoor-critical
LayerScore = Tuple[float, str]

_SAFE_NAME_RE = re.compile(r"[^\w.\-]+")
_CONV_TO_BN_RE = re.compile(r"(^|\.)conv(\d+)$")
_BN_TENSOR_SUFFIXES = ("weight", "bias", "running_mean", "running_var")
_BN_STAT_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def derive_lp_seed(base_seed: int, *parts: Any) -> int:
    """Stable 63-bit seed from global seed + parts (process-independent)."""
    raw = "|".join(str(p) for p in (int(base_seed), *parts))
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)

def _is_bn_stat_key(key: str) -> bool:
    """BN running stats are kept from the benign base; only weight/bias are swapped."""
    return any(key.endswith(f".{suffix}") for suffix in _BN_STAT_SUFFIXES)


def _replaceable_keys(keys: Sequence[str]) -> List[str]:
    return [key for key in keys if not _is_bn_stat_key(key)]


@dataclass(frozen=True)
class AtomicGroup:
    """One selectable module: Conv+BN or Linear (weight+bias)."""

    name: str
    keys: Tuple[str, ...]


def _default_layer_score_dir() -> str:
    try:
        from hydra.core.hydra_config import HydraConfig

        out_dir = HydraConfig.get().runtime.output_dir
        if out_dir:
            return os.path.join(out_dir, "lp_layer_scores")
    except Exception:
        pass
    return os.path.abspath("lp_layer_scores")


def sanitize_client_name(client_id: Any) -> str:
    name = str(client_id).strip() or "unknown"
    return _SAFE_NAME_RE.sub("_", name)


def layer_score_path(score_dir: str, round_idx: Optional[int], client_id: Any) -> str:
    round_name = f"round_{int(round_idx)}" if round_idx is not None else "round_unknown"
    client_name = f"client_{sanitize_client_name(client_id)}"
    return os.path.join(score_dir, round_name, f"{client_name}.json")


def build_layer_selection_record(
    *,
    client_id: str,
    round_idx: Optional[int],
    attack_list: Sequence[str],
    ranked: Sequence[LayerScore],
    bsr_malicious: float,
    tau: float,
    val_ratio: float,
    final_bsr: Optional[float] = None,
    ref_benign_acc: Optional[float] = None,
    ref_benign_epochs: Optional[int] = None,
    ref_malicious_asr: Optional[float] = None,
    ref_malicious_epochs: Optional[int] = None,
    num_atomic_groups: Optional[int] = None,
    selected_param_keys: Optional[Sequence[str]] = None,
    group_key_map: Optional[Mapping[str, Sequence[str]]] = None,
    eps: Optional[float] = None,
    selection_method: str = "warm_start_greedy",
) -> Dict[str, Any]:
    selected_set = set(attack_list)
    return {
        "round": round_idx,
        "client_id": str(client_id),
        "tau": float(tau),
        "val_ratio": float(val_ratio),
        "eps": None if eps is None else float(eps),
        "selection_method": selection_method,
        "bsr_malicious": float(bsr_malicious),
        "final_bsr": None if final_bsr is None else float(final_bsr),
        "ref_benign_acc": None if ref_benign_acc is None else float(ref_benign_acc),
        "ref_benign_epochs": ref_benign_epochs,
        "ref_malicious_asr": None if ref_malicious_asr is None else float(ref_malicious_asr),
        "ref_malicious_epochs": ref_malicious_epochs,
        "num_atomic_groups": num_atomic_groups,
        "num_selected": len(attack_list),
        "selected_layers": list(attack_list),
        "selected_param_keys": list(selected_param_keys or []),
        "group_keys": {
            name: list(keys)
            for name, keys in (group_key_map or {}).items()
            if name in selected_set
        },
        "layer_scores": [
            {
                "rank": i,
                "name": name,
                "delta_bsr": float(score),
                "selected": name in selected_set,
                "keys": list((group_key_map or {}).get(name, [])),
            }
            for i, (score, name) in enumerate(ranked)
        ],
    }


def save_layer_score_record(score_dir: str, record: Mapping[str, Any]) -> str:
    path = layer_score_path(score_dir, record.get("round"), record.get("client_id"))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, indent=2)
    with _LAYER_LOG_LOCK:
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")
    return path


def append_layer_selection_jsonl(score_dir: str, record: Mapping[str, Any]) -> str:
    """Append one compact selection summary line for cross-round browsing."""
    os.makedirs(score_dir, exist_ok=True)
    path = os.path.join(score_dir, "selections.jsonl")
    summary = {
        "round": record.get("round"),
        "client_id": record.get("client_id"),
        "tau": record.get("tau"),
        "eps": record.get("eps"),
        "selection_method": record.get("selection_method"),
        "bsr_malicious": record.get("bsr_malicious"),
        "final_bsr": record.get("final_bsr"),
        "ref_benign_acc": record.get("ref_benign_acc"),
        "ref_benign_epochs": record.get("ref_benign_epochs"),
        "ref_malicious_asr": record.get("ref_malicious_asr"),
        "ref_malicious_epochs": record.get("ref_malicious_epochs"),
        "num_atomic_groups": record.get("num_atomic_groups"),
        "num_selected": record.get("num_selected"),
        "selected_layers": list(record.get("selected_layers") or []),
        "layer_ranking": [
            {
                "rank": item.get("rank"),
                "name": item.get("name"),
                "delta_bsr": item.get("delta_bsr"),
                "selected": item.get("selected"),
            }
            for item in (record.get("layer_scores") or [])
        ],
    }
    line = json.dumps(summary, ensure_ascii=False)
    with _LAYER_LOG_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")
    return path


@torch.no_grad()
def compute_bsr(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        data, target = batch[0], batch[1]
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        preds = model(data).argmax(dim=1)
        correct += int(preds.eq(target).sum().item())
        total += int(target.size(0))
    return float(correct) / float(total) if total > 0 else 0.0


@torch.no_grad()
def compute_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Clean-task accuracy on a loader with original labels."""
    return compute_bsr(model, loader, device)


def _available_keys(
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
):
    keys = set()
    for key, val in benign_state.items():
        if key not in malicious_state:
            continue
        if not torch.is_tensor(val) or not torch.is_tensor(malicious_state[key]):
            continue
        keys.add(key)
    return keys


def _module_param_keys(module_name: str, available) -> List[str]:
    prefix = f"{module_name}."
    return sorted(k for k in available if k.startswith(prefix))


def _bn_group_keys(bn_name: str, available) -> List[str]:
    keys = []
    for suffix in _BN_TENSOR_SUFFIXES:
        key = f"{bn_name}.{suffix}"
        if key in available:
            keys.append(key)
    return keys


def _paired_bn_name(conv_name: str, bn_names) -> Optional[str]:
    if conv_name.endswith("shortcut.0"):
        cand = conv_name[: -len("0")] + "1"
        return cand if cand in bn_names else None
    if _CONV_TO_BN_RE.search(conv_name) is None:
        return None
    cand = _CONV_TO_BN_RE.sub(r"\1bn\2", conv_name)
    return cand if cand in bn_names else None


def build_atomic_groups(
    model: nn.Module,
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
) -> List[AtomicGroup]:
    """
    Build selectable atomic modules:
    - Conv + paired BN (group tracks all BN tensors; swap uses weight/bias only)
    - Linear (weight + bias)
    Each group counts as one layer for FLS/BLS.
    """
    available = _available_keys(benign_state, malicious_state)
    conv_names: List[str] = []
    bn_names = set()
    linear_names: List[str] = []

    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, nn.Conv2d):
            conv_names.append(name)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_names.add(name)
        elif isinstance(module, nn.Linear):
            linear_names.append(name)

    groups: List[AtomicGroup] = []
    for conv_name in conv_names:
        keys = _module_param_keys(conv_name, available)
        bn_name = _paired_bn_name(conv_name, bn_names)
        if bn_name is not None:
            keys.extend(_bn_group_keys(bn_name, available))
            bn_tail = bn_name.split(".")[-1]
            if bn_tail == "1" and conv_name.endswith("shortcut.0"):
                group_name = f"{conv_name}+bn"
            else:
                group_name = f"{conv_name}+{bn_tail}"
        else:
            group_name = conv_name
        uniq_keys: List[str] = []
        seen = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            uniq_keys.append(key)
        if uniq_keys:
            groups.append(AtomicGroup(name=group_name, keys=tuple(uniq_keys)))

    for linear_name in linear_names:
        keys = [
            key
            for key in (f"{linear_name}.weight", f"{linear_name}.bias")
            if key in available
        ]
        if keys:
            groups.append(AtomicGroup(name=linear_name, keys=tuple(keys)))

    return groups


def expand_group_keys(
    group_names: Sequence[str],
    group_map: Mapping[str, Sequence[str]],
) -> List[str]:
    keys: List[str] = []
    seen = set()
    for name in group_names:
        for key in group_map.get(name, []):
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return keys


def _clone_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in state.items()
    }


def _substitute_keys(
    base_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
    keys: Sequence[str],
) -> Dict[str, torch.Tensor]:
    mixed = _clone_state(base_state)
    for key in _replaceable_keys(keys):
        if key in source_state and torch.is_tensor(source_state[key]):
            mixed[key] = source_state[key].detach().clone()
    return mixed


@torch.no_grad()
def fls(
    model: nn.Module,
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
    bsr_malicious: float,
    val_loader: DataLoader,
    device: torch.device,
    groups: Optional[Sequence[AtomicGroup]] = None,
) -> Tuple[List[str], List[float], Dict[str, Tuple[str, ...]]]:
    """
    Forward Layer Substitution over atomic groups:
    replace each malicious group with benign, measure BSR drop.
    """
    if groups is None:
        groups = build_atomic_groups(model, benign_state, malicious_state)
    group_map = {g.name: g.keys for g in groups}
    key_arr: List[str] = []
    value_arr: List[float] = []

    eval_model = model
    was_training = eval_model.training
    eval_model.eval()

    for group in groups:
        mixed = _substitute_keys(malicious_state, benign_state, group.keys)
        eval_model.load_state_dict(mixed, strict=False)
        bsr = compute_bsr(eval_model, val_loader, device)
        key_arr.append(group.name)
        value_arr.append(float(bsr) - float(bsr_malicious))

    if was_training:
        eval_model.train()
    return key_arr, value_arr, group_map


def _group_param_l2(
    group_keys: Sequence[str],
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
) -> float:
    """L2 parameter shift of a group; used only as a tie-break (not network order)."""
    total = 0.0
    for key in _replaceable_keys(group_keys):
        if key not in benign_state or key not in malicious_state:
            continue
        b = benign_state[key]
        m = malicious_state[key]
        if not torch.is_tensor(b) or not torch.is_tensor(m):
            continue
        total += float((m.detach().float() - b.detach().float()).pow(2).sum().item())
    return total


@torch.no_grad()
def _eval_selected_bsr(
    model: nn.Module,
    selected: Sequence[str],
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
    val_loader: DataLoader,
    device: torch.device,
    group_map: Mapping[str, Sequence[str]],
) -> float:
    mixed = _substitute_keys(
        benign_state,
        malicious_state,
        expand_group_keys(selected, group_map),
    )
    model.load_state_dict(mixed, strict=False)
    return compute_bsr(model, val_loader, device)


@torch.no_grad()
def bls(
    model: nn.Module,
    key_arr: Sequence[str],
    value_arr: Sequence[float],
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
    bsr_malicious: float,
    val_loader: DataLoader,
    device: torch.device,
    group_map: Mapping[str, Sequence[str]],
    tau: float = 0.95,
    eps: float = DEFAULT_BLS_EPS,
) -> Tuple[List[str], Optional[float]]:
    """
    Hybrid warm-start greedy + backtracking pruning over atomic groups.

    1) Warm-start: greedily add FLS-negative layers (delta < -eps), most negative first.
    2) Conditional greedy on plateau layers (|delta| <= eps); expand to positive on stall.
    3) If single-layer gains stall, try the best complementary pair; else L2 tie-break.
    4) Backtracking prune while BSR stays >= tau * BSR_mal.
    """
    if not key_arr:
        return [], None

    threshold = float(bsr_malicious) * float(tau)
    eps = float(eps)
    delta = {key_arr[i]: float(value_arr[i]) for i in range(len(key_arr))}

    neg = [g for g in key_arr if delta[g] < -eps]
    zero = [g for g in key_arr if abs(delta[g]) <= eps]
    pos = [g for g in key_arr if delta[g] > eps]

    was_training = model.training
    model.eval()

    selected: List[str] = []
    selected_set = set()
    cur_bsr = _eval_selected_bsr(
        model, selected, benign_state, malicious_state, val_loader, device, group_map
    )

    # Phase 1: warm-start — add Neg one-by-one, most negative first.
    for g in sorted(neg, key=lambda name: (delta[name], name)):
        if cur_bsr >= threshold:
            break
        selected.append(g)
        selected_set.add(g)
        cur_bsr = _eval_selected_bsr(
            model, selected, benign_state, malicious_state, val_loader, device, group_map
        )

    def _l2(name: str) -> float:
        return _group_param_l2(group_map.get(name, ()), benign_state, malicious_state)

    def _best_single(candidates: Sequence[str]) -> Tuple[Optional[str], float, float]:
        best_name: Optional[str] = None
        best_gain = float("-inf")
        best_bsr = cur_bsr
        best_l2 = float("-inf")
        for g in candidates:
            trial = selected + [g]
            bsr = _eval_selected_bsr(
                model, trial, benign_state, malicious_state, val_loader, device, group_map
            )
            gain = float(bsr) - float(cur_bsr)
            l2 = _l2(g)
            # Prefer higher gain; tie-break by larger param shift (not network index).
            if (
                best_name is None
                or gain > best_gain + 1e-15
                or (abs(gain - best_gain) <= 1e-15 and l2 > best_l2)
            ):
                best_name = g
                best_gain = gain
                best_bsr = float(bsr)
                best_l2 = l2
        return best_name, best_gain, best_bsr

    def _best_pair(candidates: Sequence[str]) -> Tuple[Optional[Tuple[str, str]], float, float]:
        best_pair: Optional[Tuple[str, str]] = None
        best_gain = float("-inf")
        best_bsr = cur_bsr
        best_l2 = float("-inf")
        n = len(candidates)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = candidates[i], candidates[j]
                trial = selected + [a, b]
                bsr = _eval_selected_bsr(
                    model,
                    trial,
                    benign_state,
                    malicious_state,
                    val_loader,
                    device,
                    group_map,
                )
                gain = float(bsr) - float(cur_bsr)
                l2 = _l2(a) + _l2(b)
                if (
                    best_pair is None
                    or gain > best_gain + 1e-15
                    or (abs(gain - best_gain) <= 1e-15 and l2 > best_l2)
                ):
                    best_pair = (a, b)
                    best_gain = gain
                    best_bsr = float(bsr)
                    best_l2 = l2
        return best_pair, best_gain, best_bsr

    # Phase 2: conditional greedy on plateau (then positive if needed).
    candidates = [g for g in zero if g not in selected_set]
    pos_pending = [g for g in pos if g not in selected_set]
    pos_merged = False

    while cur_bsr < threshold:
        pool = [g for g in candidates if g not in selected_set]
        if not pool:
            if not pos_merged and pos_pending:
                candidates = list(dict.fromkeys(candidates + pos_pending))
                pos_merged = True
                continue
            break

        g_star, gain, new_bsr = _best_single(pool)
        if g_star is not None and gain > eps:
            selected.append(g_star)
            selected_set.add(g_star)
            cur_bsr = new_bsr
            continue

        # Stall: expand positive layers into the candidate pool once.
        if not pos_merged and pos_pending:
            candidates = list(dict.fromkeys(candidates + pos_pending))
            pos_merged = True
            continue

        # Complementary pair probe among remaining candidates.
        pair, pair_gain, pair_bsr = _best_pair(pool)
        if pair is not None and pair_gain > eps:
            for g in pair:
                if g not in selected_set:
                    selected.append(g)
                    selected_set.add(g)
            cur_bsr = pair_bsr
            continue

        # Last resort: force-add the best single by (gain, L2) to escape deadlock.
        if g_star is None:
            break
        selected.append(g_star)
        selected_set.add(g_star)
        cur_bsr = new_bsr

    # Phase 3: backtracking prune — drop any layer that is not necessary for tau.
    changed = True
    while changed and selected:
        changed = False
        # Try removals in reverse addition order first (later adds more likely redundant).
        for idx in range(len(selected) - 1, -1, -1):
            trial = selected[:idx] + selected[idx + 1 :]
            bsr = _eval_selected_bsr(
                model, trial, benign_state, malicious_state, val_loader, device, group_map
            )
            if float(bsr) >= threshold:
                selected = trial
                selected_set = set(selected)
                cur_bsr = float(bsr)
                changed = True
                break

    if was_training:
        model.train()
    return list(selected), float(cur_bsr)


def craft_lp_state(
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    attack_list: Sequence[str],
    lambda_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Official-style craft (Eq.4). At lambda=1:
    BC param keys <- malicious, others <- benign (reference benign as base).

    attack_list must be state_dict keys already expanded from atomic groups.
    BN running_mean/running_var always stay benign; only weight/bias are swapped.
    """
    attack_set = set(attack_list)
    lam = float(lambda_scale)
    relu_term = max(0.0, 1.0 - lam)
    crafted: Dict[str, torch.Tensor] = {}

    for key, benign_val in benign_state.items():
        if not torch.is_tensor(benign_val):
            crafted[key] = benign_val
            continue
        if (
            key in attack_set
            and key in malicious_state
            and torch.is_tensor(malicious_state[key])
            and not _is_bn_stat_key(key)
        ):
            if key in global_state and torch.is_tensor(global_state[key]):
                g = global_state[key]
                crafted[key] = (
                    g
                    + lam * (malicious_state[key] - g)
                    + relu_term * (benign_val - g)
                ).detach().clone()
            else:
                crafted[key] = malicious_state[key].detach().clone()
        else:
            crafted[key] = benign_val.detach().clone()
    return crafted


def craft_lp_state_from_groups(
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    attack_groups: Sequence[str],
    group_map: Mapping[str, Sequence[str]],
    lambda_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Craft by atomic groups: expand selected modules, then swap all member keys."""
    return craft_lp_state(
        benign_state=benign_state,
        malicious_state=malicious_state,
        global_state=global_state,
        attack_list=expand_group_keys(attack_groups, group_map),
        lambda_scale=lambda_scale,
    )


def assemble_lp_state(
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
    attack_list: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """BC <- malicious; else <- benign. Equivalent to craft at lambda=1."""
    return craft_lp_state(
        benign_state=benign_state,
        malicious_state=malicious_state,
        global_state=benign_state,
        attack_list=attack_list,
        lambda_scale=1.0,
    )


@ATTACK_REGISTRY.register("lp")
class LPAttack:
    """
    Upload-stage LP attack: reference FLS + warm-start greedy BLS(tau) on atomic groups;
    craft = local benign + local-mal BC.
    """

    client_class = None

    def __init__(
        self,
        target_label: int = 5,
        poison_ratio: float = 0.5,
        patch_size: int = 5,
        patch_value: float = 1.0,
        patch_location: str = "bottom_right",
        seed: Optional[int] = None,
        tau: float = 0.95,
        val_ratio: float = 0.2,
        lambda_scale: float = 1.0,
        ref_benign_acc_threshold: float = 0.80,
        ref_malicious_asr_threshold: float = 0.90,
        ref_malicious_max_epochs: int = 32,
        # Half-bin for N_val=125 (1/125=0.008): only exact-0 FLS deltas are plateau.
        eps: float = DEFAULT_BLS_EPS,
        log_selected_layers: bool = True,
        layer_score_dir: Optional[str] = None,
        # Backward-compatible aliases / ignored legacy knobs.
        top_k: Optional[int] = None,
        min_ndim: Optional[int] = None,
        benign_acc_threshold: Optional[float] = None,
        lsa_max_epoch_mult: Optional[int] = None,
        lsa_malicious_epoch_mult: Optional[int] = None,
        lsa_interval: Optional[int] = None,
        layer_selection_log_path: Optional[str] = None,
        ref_benign_max_epochs: Optional[int] = None,
        ref_benign_eval_every: Optional[int] = None,
        ref_malicious_epochs: Optional[int] = None,
    ):
        del (
            top_k,
            min_ndim,
            benign_acc_threshold,
            lsa_max_epoch_mult,
            lsa_malicious_epoch_mult,
            lsa_interval,
            layer_selection_log_path,
            ref_benign_max_epochs,
            ref_benign_eval_every,
            ref_malicious_epochs,
        )
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
        self.tau = float(tau)
        self.val_ratio = float(val_ratio)
        self.lambda_scale = float(lambda_scale)
        self.ref_benign_acc_threshold = float(ref_benign_acc_threshold)
        self.ref_malicious_asr_threshold = float(ref_malicious_asr_threshold)
        self.ref_malicious_max_epochs = max(1, int(ref_malicious_max_epochs))
        self.eps = float(eps)
        self.log_selected_layers = log_selected_layers
        self.layer_score_dir = layer_score_dir

        self._benign_states: Dict[str, Dict[str, torch.Tensor]] = {}
        self._malicious_states: Dict[str, Dict[str, torch.Tensor]] = {}
        # Expanded state_dict keys for craft; always derived from atomic groups.
        self._attack_lists: Dict[str, List[str]] = {}
        self._attack_groups: Dict[str, List[str]] = {}
        self._group_maps: Dict[str, Dict[str, Tuple[str, ...]]] = {}

    def poison_dataset(
        self,
        dataset: Dataset,
        mode: str,
        split: str = "",
        client_id: Optional[str] = None,
        round_idx: Optional[int] = None,
        **kwargs,
    ) -> Dataset:
        # Fix poison subset by (global seed, client, split, mode); exclude round
        # so the same client keeps a stable poison mask across rounds.
        if "seed" not in kwargs and self.seed is not None:
            kwargs["seed"] = derive_lp_seed(
                int(self.seed),
                client_id if client_id is not None else "",
                split or "",
                mode,
                "poison",
            )
        return self._badnets.poison_dataset(
            dataset,
            mode=mode,
            split=split,
            client_id=client_id,
            round_idx=round_idx,
            **kwargs,
        )

    def cache_benign_state(self, client_id: str, state: Mapping[str, torch.Tensor]) -> None:
        self._benign_states[str(client_id)] = {
            k: v.detach().clone() if torch.is_tensor(v) else v for k, v in state.items()
        }

    def cache_malicious_state(self, client_id: str, state: Mapping[str, torch.Tensor]) -> None:
        self._malicious_states[str(client_id)] = {
            k: v.detach().clone() if torch.is_tensor(v) else v for k, v in state.items()
        }

    def pop_benign_state(self, client_id: str) -> Dict[str, torch.Tensor]:
        key = str(client_id)
        if key not in self._benign_states:
            raise KeyError(f"No benign_state cached for client {client_id}")
        return self._benign_states.pop(key)

    def pop_malicious_state(self, client_id: str) -> Dict[str, torch.Tensor]:
        key = str(client_id)
        if key not in self._malicious_states:
            raise KeyError(f"No malicious_state cached for client {client_id}")
        return self._malicious_states.pop(key)

    def set_attack_list(self, client_id: str, attack_list: Sequence[str]) -> None:
        """Cache expanded state_dict keys used by craft (atomic-group expanded)."""
        self._attack_lists[str(client_id)] = list(attack_list)

    def set_attack_groups(
        self,
        client_id: str,
        attack_groups: Sequence[str],
        group_map: Mapping[str, Sequence[str]],
    ) -> None:
        """Cache selected atomic groups and expand them into craft keys."""
        cid = str(client_id)
        self._attack_groups[cid] = list(attack_groups)
        self._group_maps[cid] = {name: tuple(keys) for name, keys in group_map.items()}
        self.set_attack_list(cid, expand_group_keys(attack_groups, group_map))

    def _resolve_score_dir(self) -> str:
        if self.layer_score_dir:
            return self.layer_score_dir
        return _default_layer_score_dir()

    def _persist_layer_selection(self, record: Dict[str, Any]) -> None:
        """Write per-round JSON + append selections.jsonl when logging enabled."""
        if not self.log_selected_layers:
            return
        score_dir = self._resolve_score_dir()
        json_path = save_layer_score_record(score_dir, record)
        jsonl_path = append_layer_selection_jsonl(score_dir, record)
        record["score_file"] = json_path
        record["selections_jsonl"] = jsonl_path

    @torch.no_grad()
    def identify_bc_layers(
        self,
        model: nn.Module,
        *,
        client_id: str,
        benign_state: Mapping[str, torch.Tensor],
        malicious_state: Mapping[str, torch.Tensor],
        val_loader: DataLoader,
        device: torch.device,
        round_idx: Optional[int] = None,
        ref_benign_acc: Optional[float] = None,
        ref_benign_epochs: Optional[int] = None,
        ref_malicious_asr: Optional[float] = None,
        ref_malicious_epochs: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """FLS + warm-start greedy BLS(tau) on atomic groups; cache expanded keys for craft."""
        model.load_state_dict(malicious_state, strict=False)
        bsr_mal = compute_bsr(model, val_loader, device)

        groups = build_atomic_groups(model, benign_state, malicious_state)
        key_arr, value_arr, group_map = fls(
            model=model,
            benign_state=benign_state,
            malicious_state=malicious_state,
            bsr_malicious=bsr_mal,
            val_loader=val_loader,
            device=device,
            groups=groups,
        )
        attack_groups, final_bsr = bls(
            model=model,
            key_arr=key_arr,
            value_arr=value_arr,
            benign_state=benign_state,
            malicious_state=malicious_state,
            bsr_malicious=bsr_mal,
            val_loader=val_loader,
            device=device,
            group_map=group_map,
            tau=self.tau,
            eps=self.eps,
        )
        expanded_keys = expand_group_keys(attack_groups, group_map)
        ranked: List[LayerScore] = sorted(
            [(float(value_arr[i]), key_arr[i]) for i in range(len(key_arr))],
            key=lambda x: x[0],
        )
        cid = str(client_id)
        record = build_layer_selection_record(
            client_id=cid,
            round_idx=round_idx,
            attack_list=attack_groups,
            ranked=ranked,
            bsr_malicious=float(bsr_mal),
            tau=self.tau,
            val_ratio=self.val_ratio,
            final_bsr=final_bsr,
            ref_benign_acc=ref_benign_acc,
            ref_benign_epochs=ref_benign_epochs,
            ref_malicious_asr=ref_malicious_asr,
            ref_malicious_epochs=ref_malicious_epochs,
            num_atomic_groups=len(groups),
            selected_param_keys=expanded_keys,
            group_key_map=group_map,
            eps=self.eps,
            selection_method="warm_start_greedy",
        )
        # Craft replaces by atomic groups: expand selected groups to all member keys.
        self.set_attack_groups(cid, attack_groups, group_map)
        self._persist_layer_selection(record)
        return list(attack_groups), record

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
        """Craft LP weights: local benign + atomic-group BC from local mal."""
        del update, num_samples, round_idx
        kwargs.clear()

        if client_id is None:
            raise ValueError("LPAttack.poison_upload requires client_id")

        cid = str(client_id)
        # benign_state is the local benign model used as craft base.
        benign_state = self.pop_benign_state(cid)
        malicious_state = self.pop_malicious_state(cid)

        # Always craft by atomic groups: expand selected modules to all member keys.
        attack_groups = self._attack_groups.pop(cid, None)
        group_map = self._group_maps.pop(cid, None)
        self._attack_lists.pop(cid, None)
        if attack_groups is None or group_map is None:
            raise RuntimeError(
                f"LPAttack missing attack groups for client {cid}; "
                "run identify_bc_layers before package"
            )

        crafted = craft_lp_state_from_groups(
            benign_state=benign_state,
            malicious_state=malicious_state,
            global_state=initial_weights,
            attack_groups=attack_groups,
            group_map=group_map,
            lambda_scale=self.lambda_scale,
        )
        return {
            k: crafted[k] - initial_weights[k]
            for k in crafted
            if k in initial_weights and torch.is_tensor(crafted[k])
        }


# Late bind client class to avoid import cycles with LPClient.
from core.client.lp_client import LPClient  # noqa: E402

LPAttack.client_class = LPClient
