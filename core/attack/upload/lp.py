"""
LP-Attack: Layer-wise Poisoning via Backdoor-Critical (BC) layers.

Aligns with official train_malicious_LPA + FLS/BLS (adaptive_local style):
identify BC layers on the local_ep benign/malicious pair, then upload
malicious weights on BC layers and benign weights elsewhere.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from core.attack.data.badnets import BadNetsAttack
from core.utils.registry import ATTACK_REGISTRY

_LAYER_LOG_LOCK = threading.Lock()
_logger = logging.getLogger(__name__)

# (delta_bsr, param_name); more negative => more backdoor-critical
LayerScore = Tuple[float, str]


def _default_layer_log_path() -> str:
    try:
        from hydra.core.hydra_config import HydraConfig

        out_dir = HydraConfig.get().runtime.output_dir
        if out_dir:
            return os.path.join(out_dir, "lp_selected_layers.jsonl")
    except Exception:
        pass
    return os.path.abspath("lp_selected_layers.jsonl")


def build_layer_selection_record(
    *,
    client_id: str,
    round_idx: Optional[int],
    attack_list: Sequence[str],
    ranked: Sequence[LayerScore],
    bsr_malicious: float,
    tau: float,
    val_ratio: float,
) -> Dict[str, Any]:
    """Build one JSON-serializable BC layer selection record."""
    return {
        "round": round_idx,
        "client_id": str(client_id),
        "tau": float(tau),
        "val_ratio": float(val_ratio),
        "bsr_malicious": float(bsr_malicious),
        "num_selected": len(attack_list),
        "selected_layers": list(attack_list),
        "ranked_layers": [
            {"name": name, "delta_bsr": float(score)} for score, name in ranked
        ],
    }


def append_layer_selection_record(path: str, record: Mapping[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _LAYER_LOG_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


@torch.no_grad()
def compute_bsr(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Backdoor success rate: fraction of triggered samples predicted as their labels.
    Expects loader labels to already be the attack target (poison mode=test).
    """
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
def fls(
    model: nn.Module,
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
    bsr_malicious: float,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[List[str], List[float]]:
    """
    Forward Layer Substitution (official FLS).

    Start from malicious weights; replace one named parameter with the benign
    counterpart; record BSR_replaced - BSR_malicious (more negative => more critical).
    """
    param_names = [
        name
        for name, _ in model.named_parameters()
        if name in benign_state and name in malicious_state
    ]
    key_arr: List[str] = []
    value_arr: List[float] = []

    eval_model = model
    was_training = eval_model.training
    eval_model.eval()

    for name in param_names:
        mixed = {
            k: (v.detach().clone() if torch.is_tensor(v) else v)
            for k, v in malicious_state.items()
        }
        mixed[name] = benign_state[name].detach().clone()
        eval_model.load_state_dict(mixed, strict=False)
        bsr = compute_bsr(eval_model, val_loader, device)
        key_arr.append(name)
        value_arr.append(float(bsr) - float(bsr_malicious))

    if was_training:
        eval_model.train()
    return key_arr, value_arr


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
    tau: float = 0.8,
) -> List[str]:
    """
    Backward Layer Substitution (official BLS).

    Incrementally copy the most critical malicious layers into the benign model
    until BSR >= tau * BSR_malicious.
    """
    if not key_arr:
        return []
    if bsr_malicious <= 0.0:
        # Degenerate backdoor: fall back to the single most critical layer if any.
        order = sorted(range(len(value_arr)), key=lambda i: value_arr[i])
        return [key_arr[order[0]]] if order else []

    threshold = float(tau) * float(bsr_malicious)
    n = 1
    attack_list: List[str] = []
    eval_model = model
    was_training = eval_model.training
    eval_model.eval()

    # Precompute ascending indices by criticality (most negative first).
    order = sorted(range(len(value_arr)), key=lambda i: value_arr[i])

    while n <= len(key_arr):
        attack_list = [key_arr[i] for i in order[:n]]
        mixed = {
            k: (v.detach().clone() if torch.is_tensor(v) else v)
            for k, v in benign_state.items()
        }
        for layer in attack_list:
            mixed[layer] = malicious_state[layer].detach().clone()
        eval_model.load_state_dict(mixed, strict=False)
        temp_bsr = compute_bsr(eval_model, val_loader, device)
        if temp_bsr >= threshold:
            break
        n += 1

    if was_training:
        eval_model.train()
    return list(attack_list)


def assemble_lp_state(
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
    attack_list: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """BC layers from malicious; all other tensors from benign."""
    attack_set = set(attack_list)
    assembled: Dict[str, torch.Tensor] = {}
    for key, val in benign_state.items():
        if key in attack_set and key in malicious_state and torch.is_tensor(malicious_state[key]):
            assembled[key] = malicious_state[key].detach().clone()
        elif torch.is_tensor(val):
            assembled[key] = val.detach().clone()
        else:
            assembled[key] = val
    # Keep any malicious-only keys out; benign is the structural base.
    return assembled


@ATTACK_REGISTRY.register("lp")
class LPAttack:
    """
    Upload-stage LP attack.

    - poison_dataset: BadNets trigger poisoning
    - identify_bc_layers: FLS + BLS on cached benign/malicious states
    - poison_upload: assemble BC-malicious / else-benign, return delta vs global
    """

    client_class = None

    def __init__(
        self,
        target_label: int = 0,
        poison_ratio: float = 0.5,
        patch_size: int = 5,
        patch_value: float = 1.0,
        patch_location: str = "bottom_right",
        seed: Optional[int] = None,
        tau: float = 0.8,
        val_ratio: float = 0.25,
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
        self.tau = float(tau)
        self.val_ratio = float(val_ratio)
        self.log_selected_layers = log_selected_layers
        self.layer_selection_log_path = layer_selection_log_path

        self._benign_states: Dict[str, Dict[str, torch.Tensor]] = {}
        self._malicious_states: Dict[str, Dict[str, torch.Tensor]] = {}
        self._attack_lists: Dict[str, List[str]] = {}
        self._bsr_malicious: Dict[str, float] = {}

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

    def _resolve_log_path(self) -> str:
        if self.layer_selection_log_path:
            return self.layer_selection_log_path
        return _default_layer_log_path()

    def _log_selected_layers(
        self,
        record: Mapping[str, Any],
    ) -> None:
        if not self.log_selected_layers:
            return
        _logger.info(
            "LP BC layers=%d tau=%.2f BSR_mal=%.4f round=%s client=%s layers=%s",
            record.get("num_selected", 0),
            record.get("tau", self.tau),
            record.get("bsr_malicious", 0.0),
            record.get("round"),
            record.get("client_id"),
            record.get("selected_layers", []),
        )

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
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Run FLS+BLS, cache BC layers, and return (attack_list, selection_record)."""
        # Measure malicious BSR on current malicious weights.
        model.load_state_dict(malicious_state, strict=False)
        bsr_mal = compute_bsr(model, val_loader, device)

        key_arr, value_arr = fls(
            model=model,
            benign_state=benign_state,
            malicious_state=malicious_state,
            bsr_malicious=bsr_mal,
            val_loader=val_loader,
            device=device,
        )
        attack_list = bls(
            model=model,
            key_arr=key_arr,
            value_arr=value_arr,
            benign_state=benign_state,
            malicious_state=malicious_state,
            bsr_malicious=bsr_mal,
            val_loader=val_loader,
            device=device,
            tau=self.tau,
        )
        ranked: List[LayerScore] = sorted(
            [(value_arr[i], key_arr[i]) for i in range(len(key_arr))],
            key=lambda x: x[0],
        )
        cid = str(client_id)
        record = build_layer_selection_record(
            client_id=cid,
            round_idx=round_idx,
            attack_list=attack_list,
            ranked=ranked,
            bsr_malicious=float(bsr_mal),
            tau=self.tau,
            val_ratio=self.val_ratio,
        )
        self._attack_lists[cid] = list(attack_list)
        self._bsr_malicious[cid] = float(bsr_mal)
        self._log_selected_layers(record)
        if self.log_selected_layers and self.layer_selection_log_path:
            append_layer_selection_record(self._resolve_log_path(), record)
        return list(attack_list), record

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
        """Assemble LP absolute weights, then convert to delta vs global."""
        del update, num_samples, round_idx
        kwargs.clear()

        if client_id is None:
            raise ValueError("LPAttack.poison_upload requires client_id")

        cid = str(client_id)
        benign_state = self.pop_benign_state(cid)
        malicious_state = self.pop_malicious_state(cid)
        attack_list = self._attack_lists.pop(cid, None)
        self._bsr_malicious.pop(cid, None)
        if attack_list is None:
            raise RuntimeError(
                f"LPAttack missing attack_list for client {cid}; "
                "call identify_bc_layers before package/poison_upload"
            )

        assembled = assemble_lp_state(benign_state, malicious_state, attack_list)
        return {
            k: assembled[k] - initial_weights[k]
            for k in assembled
            if k in initial_weights and torch.is_tensor(assembled[k])
        }


# Late bind client class to avoid import cycles with LPClient.
from core.client.lp_client import LPClient  # noqa: E402

LPAttack.client_class = LPClient
