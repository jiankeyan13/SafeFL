"""
LP-Attack: Layer-wise Poisoning via Backdoor-Critical (BC) layers.

Official-style flow without adaptive layer-count search:
1) (every lsa_interval rounds) long-train LSA pair: serial benign-to-threshold
   then malicious; FLS on ndim>2 params; fix top-K critical layers.
2) Train a separate local_ep benign/malicious pair from Wg for upload.
3) Assemble upload with lambda (default 1.0): BC <- local mal, else <- local benign.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from core.attack.data.badnets import BadNetsAttack
from core.utils.registry import ATTACK_REGISTRY

_LAYER_LOG_LOCK = threading.Lock()
_LSA_CACHE_LOCK = threading.Lock()
# Persists across client object lifetimes (needed for Ray per-client rebuilds).
_LSA_CACHE: Dict[str, Dict[str, Any]] = {}
_logger = logging.getLogger(__name__)

# (delta_bsr, param_name); more negative => more backdoor-critical
LayerScore = Tuple[float, str]

_SAFE_NAME_RE = re.compile(r"[^\w.\-]+")


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
    top_k: int,
    min_ndim: int,
    val_ratio: float,
    lsa_round: Optional[int] = None,
    reused: bool = False,
    benign_acc: Optional[float] = None,
    lsa_benign_epochs: Optional[int] = None,
    lsa_malicious_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "round": round_idx,
        "client_id": str(client_id),
        "top_k": int(top_k),
        "min_ndim": int(min_ndim),
        "val_ratio": float(val_ratio),
        "bsr_malicious": float(bsr_malicious),
        "num_selected": len(attack_list),
        "selected_layers": list(attack_list),
        "lsa_round": lsa_round if lsa_round is not None else round_idx,
        "reused": bool(reused),
        "benign_acc": None if benign_acc is None else float(benign_acc),
        "lsa_benign_epochs": lsa_benign_epochs,
        "lsa_malicious_epochs": lsa_malicious_epochs,
        "layer_scores": [
            {
                "rank": i,
                "name": name,
                "delta_bsr": float(score),
                "selected": name in attack_list,
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


def eligible_param_names(
    model: nn.Module,
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
    min_ndim: int = 3,
) -> List[str]:
    names: List[str] = []
    for name, param in model.named_parameters():
        if name not in benign_state or name not in malicious_state:
            continue
        tensor = malicious_state[name]
        if torch.is_tensor(tensor) and int(tensor.ndim) >= int(min_ndim):
            names.append(name)
        elif torch.is_tensor(param) and int(param.ndim) >= int(min_ndim):
            names.append(name)
    return names


@torch.no_grad()
def fls(
    model: nn.Module,
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
    bsr_malicious: float,
    val_loader: DataLoader,
    device: torch.device,
    min_ndim: int = 3,
) -> Tuple[List[str], List[float]]:
    param_names = eligible_param_names(
        model, benign_state, malicious_state, min_ndim=min_ndim
    )
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


def select_top_critical_layers(
    key_arr: Sequence[str],
    value_arr: Sequence[float],
    top_k: int,
) -> Tuple[List[str], List[LayerScore]]:
    ranked: List[LayerScore] = sorted(
        [(float(value_arr[i]), key_arr[i]) for i in range(len(key_arr))],
        key=lambda x: x[0],
    )
    k = max(0, int(top_k))
    selected = [name for _, name in ranked[:k]]
    return selected, ranked


def craft_lp_state(
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    attack_list: Sequence[str],
    lambda_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Official-style craft (Eq.4) with uaverage = local benign.

    At lambda=1: BC layers <- malicious, others <- benign.
    """
    attack_set = set(attack_list)
    lam = float(lambda_scale)
    relu_term = max(0.0, 1.0 - lam)
    crafted: Dict[str, torch.Tensor] = {}

    for key, benign_val in benign_state.items():
        if not torch.is_tensor(benign_val):
            crafted[key] = benign_val
            continue
        if key in attack_set and key in malicious_state and torch.is_tensor(malicious_state[key]):
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
    Upload-stage LP attack with long-train LSA (fixed top-K) + local_ep craft.
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
        top_k: int = 6,
        min_ndim: int = 3,
        val_ratio: float = 0.2,
        benign_acc_threshold: float = 0.8,
        lsa_max_epoch_mult: int = 3,
        lsa_malicious_epoch_mult: int = 1,
        lsa_interval: int = 10,
        lambda_scale: float = 1.0,
        log_selected_layers: bool = True,
        layer_score_dir: Optional[str] = None,
        # Backward-compatible aliases.
        tau: Optional[float] = None,
        layer_selection_log_path: Optional[str] = None,
    ):
        del tau, layer_selection_log_path
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
        self.top_k = int(top_k)
        self.min_ndim = int(min_ndim)
        self.val_ratio = float(val_ratio)
        self.benign_acc_threshold = float(benign_acc_threshold)
        self.lsa_max_epoch_mult = int(lsa_max_epoch_mult)
        self.lsa_malicious_epoch_mult = int(lsa_malicious_epoch_mult)
        self.lsa_interval = max(1, int(lsa_interval))
        self.lambda_scale = float(lambda_scale)
        self.log_selected_layers = log_selected_layers
        self.layer_score_dir = layer_score_dir

        self._benign_states: Dict[str, Dict[str, torch.Tensor]] = {}
        self._malicious_states: Dict[str, Dict[str, torch.Tensor]] = {}
        self._attack_lists: Dict[str, List[str]] = {}

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

    def set_attack_list(self, client_id: str, attack_list: Sequence[str]) -> None:
        self._attack_lists[str(client_id)] = list(attack_list)

    def should_run_lsa(self, client_id: str, round_idx: Optional[int]) -> bool:
        cid = str(client_id)
        with _LSA_CACHE_LOCK:
            cached = _LSA_CACHE.get(cid)
        if cached is None or not cached.get("attack_list"):
            return True
        if round_idx is None:
            return True
        return int(round_idx) % self.lsa_interval == 0

    def get_lsa_cache(self, client_id: str) -> Optional[Dict[str, Any]]:
        with _LSA_CACHE_LOCK:
            cached = _LSA_CACHE.get(str(client_id))
            return dict(cached) if cached is not None else None

    def _store_lsa_cache(self, client_id: str, payload: Mapping[str, Any]) -> None:
        with _LSA_CACHE_LOCK:
            _LSA_CACHE[str(client_id)] = dict(payload)

    def _resolve_score_dir(self) -> str:
        if self.layer_score_dir:
            return self.layer_score_dir
        return _default_layer_score_dir()

    def _log_selected_layers(self, record: Mapping[str, Any]) -> None:
        if not self.log_selected_layers:
            return
        _logger.info(
            "LP BC layers=%d top_k=%d reused=%s BSR_mal=%.4f round=%s client=%s layers=%s",
            record.get("num_selected", 0),
            record.get("top_k", self.top_k),
            record.get("reused", False),
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
        benign_acc: Optional[float] = None,
        lsa_benign_epochs: Optional[int] = None,
        lsa_malicious_epochs: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """FLS on LSA states; fix top-K; update cache and return record."""
        model.load_state_dict(malicious_state, strict=False)
        bsr_mal = compute_bsr(model, val_loader, device)

        key_arr, value_arr = fls(
            model=model,
            benign_state=benign_state,
            malicious_state=malicious_state,
            bsr_malicious=bsr_mal,
            val_loader=val_loader,
            device=device,
            min_ndim=self.min_ndim,
        )
        attack_list, ranked = select_top_critical_layers(
            key_arr, value_arr, top_k=self.top_k
        )
        cid = str(client_id)
        record = build_layer_selection_record(
            client_id=cid,
            round_idx=round_idx,
            attack_list=attack_list,
            ranked=ranked,
            bsr_malicious=float(bsr_mal),
            top_k=self.top_k,
            min_ndim=self.min_ndim,
            val_ratio=self.val_ratio,
            lsa_round=round_idx,
            reused=False,
            benign_acc=benign_acc,
            lsa_benign_epochs=lsa_benign_epochs,
            lsa_malicious_epochs=lsa_malicious_epochs,
        )
        self._store_lsa_cache(
            cid,
            {
                "attack_list": list(attack_list),
                "ranked": list(ranked),
                "bsr_malicious": float(bsr_mal),
                "benign_acc": benign_acc,
                "lsa_benign_epochs": lsa_benign_epochs,
                "lsa_malicious_epochs": lsa_malicious_epochs,
                "lsa_round": round_idx,
                "record": record,
            },
        )
        self.set_attack_list(cid, attack_list)
        self._log_selected_layers(record)
        return list(attack_list), record

    def resolve_attack_list_for_round(
        self,
        client_id: str,
        round_idx: Optional[int],
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Return (attack_list, record) for this round.
        If LSA was skipped, reuse cache and mark reused=True.
        """
        cid = str(client_id)
        cached = self.get_lsa_cache(cid)
        if cached is None or not cached.get("attack_list"):
            raise RuntimeError(
                f"LPAttack has no LSA cache for client {cid}; run identify_bc_layers first"
            )
        attack_list = list(cached["attack_list"])
        ranked = list(cached.get("ranked") or [])
        record = build_layer_selection_record(
            client_id=cid,
            round_idx=round_idx,
            attack_list=attack_list,
            ranked=ranked,
            bsr_malicious=float(cached.get("bsr_malicious", 0.0)),
            top_k=self.top_k,
            min_ndim=self.min_ndim,
            val_ratio=self.val_ratio,
            lsa_round=cached.get("lsa_round"),
            reused=True,
            benign_acc=cached.get("benign_acc"),
            lsa_benign_epochs=cached.get("lsa_benign_epochs"),
            lsa_malicious_epochs=cached.get("lsa_malicious_epochs"),
        )
        self.set_attack_list(cid, attack_list)
        self._log_selected_layers(record)
        return attack_list, record

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
        """Craft LP weights from local_ep pair + cached BC list, return delta."""
        del update, num_samples, round_idx
        kwargs.clear()

        if client_id is None:
            raise ValueError("LPAttack.poison_upload requires client_id")

        cid = str(client_id)
        benign_state = self.pop_benign_state(cid)
        malicious_state = self.pop_malicious_state(cid)
        attack_list = self._attack_lists.pop(cid, None)
        if attack_list is None:
            cached = self.get_lsa_cache(cid)
            if cached is not None:
                attack_list = list(cached.get("attack_list") or [])
        if attack_list is None:
            raise RuntimeError(
                f"LPAttack missing attack_list for client {cid}; "
                "run LSA / resolve_attack_list_for_round before package"
            )

        crafted = craft_lp_state(
            benign_state=benign_state,
            malicious_state=malicious_state,
            global_state=initial_weights,
            attack_list=attack_list,
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
