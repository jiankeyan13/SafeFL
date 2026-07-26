"""
LP-Attack: Layer-wise Poisoning via Backdoor-Critical (BC) layers.

method_1-style flow (no adaptive layer-count search):
1) Reference long-train from Wg: benign until clean acc >= threshold
   (eval every N epochs, max M epochs), then malicious for 1 epoch.
   FLS+BLS on this reference pair select BC layers.
2) Separate local_ep benign/malicious pair from Wg for upload craft.
3) Craft with lambda=1: BC <- local mal, else <- local benign.
"""
from __future__ import annotations

import heapq
import json
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
    tau: float,
    val_ratio: float,
    final_bsr: Optional[float] = None,
    ref_benign_acc: Optional[float] = None,
    ref_benign_epochs: Optional[int] = None,
    ref_malicious_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "round": round_idx,
        "client_id": str(client_id),
        "tau": float(tau),
        "val_ratio": float(val_ratio),
        "bsr_malicious": float(bsr_malicious),
        "final_bsr": None if final_bsr is None else float(final_bsr),
        "ref_benign_acc": None if ref_benign_acc is None else float(ref_benign_acc),
        "ref_benign_epochs": ref_benign_epochs,
        "ref_malicious_epochs": ref_malicious_epochs,
        "num_selected": len(attack_list),
        "selected_layers": list(attack_list),
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


def append_layer_selection_jsonl(score_dir: str, record: Mapping[str, Any]) -> str:
    """Append one compact selection summary line for cross-round browsing."""
    os.makedirs(score_dir, exist_ok=True)
    path = os.path.join(score_dir, "selections.jsonl")
    summary = {
        "round": record.get("round"),
        "client_id": record.get("client_id"),
        "tau": record.get("tau"),
        "bsr_malicious": record.get("bsr_malicious"),
        "final_bsr": record.get("final_bsr"),
        "ref_benign_acc": record.get("ref_benign_acc"),
        "ref_benign_epochs": record.get("ref_benign_epochs"),
        "ref_malicious_epochs": record.get("ref_malicious_epochs"),
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


def eligible_param_names(
    model: nn.Module,
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
) -> List[str]:
    """All named parameters present in both states (no ndim filter)."""
    names: List[str] = []
    for name, _param in model.named_parameters():
        if name not in benign_state or name not in malicious_state:
            continue
        if not torch.is_tensor(benign_state[name]) or not torch.is_tensor(malicious_state[name]):
            continue
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
) -> Tuple[List[str], List[float]]:
    """
    Forward Layer Substitution: replace each malicious param with benign,
    measure BSR drop. More negative delta => more backdoor-critical.
    """
    param_names = eligible_param_names(model, benign_state, malicious_state)
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
    tau: float = 0.95,
) -> Tuple[List[str], Optional[float]]:
    """
    Backward Layer Substitution: greedily add most critical layers into a
    benign model until BSR >= tau * BSR_mal.
    """
    if not key_arr:
        return [], None

    threshold = float(bsr_malicious) * float(tau)
    n = 1
    temp_bsr = 0.0
    attack_list: List[str] = []
    was_training = model.training
    model.eval()

    while temp_bsr < threshold and n <= len(key_arr):
        min_idxs = heapq.nsmallest(n, range(len(value_arr)), key=value_arr.__getitem__)
        attack_list = [key_arr[i] for i in min_idxs]
        mixed = {
            k: (v.detach().clone() if torch.is_tensor(v) else v)
            for k, v in benign_state.items()
        }
        for layer in attack_list:
            if layer in malicious_state and torch.is_tensor(malicious_state[layer]):
                mixed[layer] = malicious_state[layer].detach().clone()
        model.load_state_dict(mixed, strict=False)
        temp_bsr = compute_bsr(model, val_loader, device)
        n += 1

    if was_training:
        model.train()
    return list(attack_list), float(temp_bsr)


def craft_lp_state(
    benign_state: Mapping[str, torch.Tensor],
    malicious_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    attack_list: Sequence[str],
    lambda_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Craft LP weights. At lambda=1: BC layers <- malicious, others <- benign.
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
    Upload-stage LP attack: reference long-train FLS+BLS(tau), craft with lambda=1.
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
        ref_benign_acc_threshold: float = 0.9,
        ref_benign_max_epochs: int = 30,
        ref_benign_eval_every: int = 3,
        ref_malicious_epochs: int = 1,
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
    ):
        del (
            top_k,
            min_ndim,
            benign_acc_threshold,
            lsa_max_epoch_mult,
            lsa_malicious_epoch_mult,
            lsa_interval,
            layer_selection_log_path,
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
        self.ref_benign_max_epochs = int(ref_benign_max_epochs)
        self.ref_benign_eval_every = max(1, int(ref_benign_eval_every))
        self.ref_malicious_epochs = max(1, int(ref_malicious_epochs))
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
        ref_malicious_epochs: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """FLS + BLS(tau) on the reference long-train pair; return BC list."""
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
        attack_list, final_bsr = bls(
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
            [(float(value_arr[i]), key_arr[i]) for i in range(len(key_arr))],
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
            final_bsr=final_bsr,
            ref_benign_acc=ref_benign_acc,
            ref_benign_epochs=ref_benign_epochs,
            ref_malicious_epochs=ref_malicious_epochs,
        )
        self.set_attack_list(cid, attack_list)
        self._persist_layer_selection(record)
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
        """Craft LP weights from local_ep pair + BC list, return delta."""
        del update, num_samples, round_idx
        kwargs.clear()

        if client_id is None:
            raise ValueError("LPAttack.poison_upload requires client_id")

        cid = str(client_id)
        benign_state = self.pop_benign_state(cid)
        malicious_state = self.pop_malicious_state(cid)
        attack_list = self._attack_lists.pop(cid, None)
        if attack_list is None:
            raise RuntimeError(
                f"LPAttack missing attack_list for client {cid}; "
                "run identify_bc_layers before package"
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
