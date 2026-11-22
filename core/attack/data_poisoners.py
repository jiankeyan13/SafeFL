"""Data-driven general-purpose poisoners: Semantic and Edge-case triggers."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as torch_f

from core.attack.poisoner import POISONER_REGISTRY, BasePoisoner
from data.loader import derive_stable_seed
from data.source import SharedDataSource, load_external_data_source

SEMANTIC_PRESETS: Dict[str, Dict[str, Any]] = {
    "cifar10_car_stripe": {
        "train_indices": (568, 33105, 33615, 33907, 36848, 40713, 41706),
        "test_indices": (330, 3934, 12336, 30560, 30696),
        "test_split": "train",
        "default_target_label": 2,
    },
    "cifar10_green_car": {
        "train_indices": (14297, 15323, 23098, 24268, 30094, 33007),
        "test_indices": (1179, 2030, 6915),
        "test_split": "train",
        "default_target_label": 2,
    },
}


def _adapt_image_shape(bx: torch.Tensor, target_chw: Tuple[int, int, int]) -> torch.Tensor:
    """Adapt ``[B, C, H, W]`` in ``[0, 1]`` to ``target_chw`` (channel + spatial resize)."""
    tc, th, tw = int(target_chw[0]), int(target_chw[1]), int(target_chw[2])
    if bx.ndim != 4:
        raise ValueError(f"Expected 4D [B, C, H, W] tensor, got {tuple(bx.shape)}")
    _, c, h, w = bx.shape

    if c != tc:
        if c == 1 and tc == 3:
            bx = bx.repeat(1, 3, 1, 1)
        elif c == 3 and tc == 1:
            bx = bx.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Cannot adapt {c} channels to target {tc} channels.")

    if (h, w) != (th, tw):
        bx = torch_f.interpolate(bx, size=(th, tw), mode="bilinear", align_corners=False)
        bx = bx.clamp_(0.0, 1.0)
    return bx


def _select_pool_indices(
    pool_len: int,
    count: int,
    *,
    seed: Optional[int],
    tag: str,
    client_id: Optional[int],
    round_idx: Optional[int],
    step: int,
) -> np.ndarray:
    """Deterministically shuffle pool and cycle-tile ``count`` indices for uniform coverage."""
    if pool_len <= 0 or count <= 0:
        return np.empty(0, dtype=np.int64)
    eff_seed = (
        None
        if seed is None
        else derive_stable_seed(int(seed), tag, client_id, round_idx, step)
    )
    rng = np.random.default_rng(eff_seed)
    perm = rng.permutation(pool_len)
    if count <= pool_len:
        return perm[:count]
    repeats = (count + pool_len - 1) // pool_len
    return np.tile(perm, repeats)[:count]


class _IndexedEvalLoader:
    """Stateless ASR eval loader yielding ``(bx_norm, target_y, orig_y)`` directly."""

    def __init__(
        self,
        *,
        source: SharedDataSource,
        norm_source: SharedDataSource,
        indices: np.ndarray,
        target_label: int,
        batch_size: int,
        device: torch.device,
        expected_chw: Optional[Tuple[int, int, int]] = None,
        override_orig_label: Optional[int] = None,
    ) -> None:
        self.source = source
        self.norm_source = norm_source
        self.indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        self.target_label = int(target_label)
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self.expected_chw = expected_chw
        self.override_orig_label = (
            None if override_orig_label is None else int(override_orig_label)
        )

    def __len__(self) -> int:
        n = int(self.indices.size)
        return 0 if n == 0 else math.ceil(n / self.batch_size)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        n = int(self.indices.size)
        for start in range(0, n, self.batch_size):
            batch_idx = self.indices[start : start + self.batch_size]
            bx, by = self.source.fetch_batch(batch_idx, device=self.device)
            if self.expected_chw is not None:
                bx = _adapt_image_shape(bx, self.expected_chw)
            self.norm_source.normalize_(bx)
            orig_y = (
                torch.full_like(by, self.override_orig_label)
                if self.override_orig_label is not None
                else by
            )
            target_y = torch.full_like(orig_y, self.target_label)
            yield bx, target_y, orig_y


@POISONER_REGISTRY.register("semantic")
class SemanticPoisoner(BasePoisoner):
    """Semantic backdoor poisoner using natural semantic sample subsets."""

    def __init__(
        self,
        target_label: Optional[int] = None,
        poison_ratio: float = 0.2,
        preset: Optional[str] = "cifar10_car_stripe",
        train_indices: Optional[Sequence[int]] = None,
        test_indices: Optional[Sequence[int]] = None,
        test_split: Optional[str] = None,
        exclude_from_benign: bool = True,
        seed: Optional[int] = 42,
        exclude_target_class: bool = False,
        **_: Any,
    ) -> None:
        preset_cfg: Optional[Dict[str, Any]] = None
        if preset is not None:
            key = str(preset).strip().lower()
            if key not in SEMANTIC_PRESETS:
                raise ValueError(
                    f"Unknown semantic preset {preset!r}. Available: {sorted(SEMANTIC_PRESETS)}"
                )
            preset_cfg = SEMANTIC_PRESETS[key]

        resolved_train = (
            [int(i) for i in train_indices]
            if train_indices is not None
            else list(preset_cfg["train_indices"]) if preset_cfg is not None else []
        )
        if not resolved_train:
            raise ValueError("SemanticPoisoner requires non-empty train_indices or a valid preset.")

        resolved_test = (
            [int(i) for i in test_indices]
            if test_indices is not None
            else list(preset_cfg["test_indices"])
            if preset_cfg is not None
            else list(resolved_train)
        )
        resolved_split = str(
            test_split
            if test_split is not None
            else (preset_cfg["test_split"] if preset_cfg is not None else "test")
        ).lower()
        if resolved_split not in ("train", "test"):
            raise ValueError(f"test_split must be 'train' or 'test', got {resolved_split!r}")

        if target_label is not None:
            resolved_target = int(target_label)
        elif preset_cfg is not None:
            resolved_target = int(preset_cfg["default_target_label"])
        else:
            resolved_target = 5

        super().__init__(
            target_label=resolved_target,
            poison_ratio=poison_ratio,
            seed=seed,
            exclude_target_class=exclude_target_class,
        )
        self.preset = preset
        self.train_indices: Tuple[int, ...] = tuple(resolved_train)
        self.test_indices: Tuple[int, ...] = tuple(resolved_test)
        self.test_split: str = resolved_split
        self.exclude_from_benign = bool(exclude_from_benign)
        self._source: Optional[SharedDataSource] = None
        self._step_counter: int = 0

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_source"] = None
        state["_step_counter"] = 0
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._source = None
        self._step_counter = 0

    def bind_data_source(self, source: Any) -> None:
        self._source = source

    def excluded_train_indices(self) -> Sequence[int]:
        if not self.exclude_from_benign:
            return ()
        if self.test_split == "train":
            return sorted(set(self.train_indices) | set(self.test_indices))
        return sorted(set(self.train_indices))

    def _stamp_train(
        self,
        bx: torch.Tensor,
        mask: torch.Tensor,
        *,
        client_id: Optional[int] = None,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        n_poison = int(mask.sum().item())
        if n_poison <= 0:
            return bx
        if self._source is None:
            raise RuntimeError(
                "SemanticPoisoner is not bound to a SharedDataSource. "
                "Use FastBatchLoader or call bind_data_source(source) first."
            )

        self._step_counter += 1
        rel_idx = _select_pool_indices(
            len(self.train_indices),
            n_poison,
            seed=self.seed,
            tag="semantic_train",
            client_id=client_id,
            round_idx=round_idx,
            step=self._step_counter,
        )
        chosen_global = np.asarray([self.train_indices[int(i)] for i in rel_idx], dtype=np.int64)
        sem_x, _ = self._source.fetch_batch(chosen_global, device=bx.device)
        sem_x = _adapt_image_shape(sem_x, (int(bx.shape[1]), int(bx.shape[2]), int(bx.shape[3])))
        bx[mask] = sem_x.to(dtype=bx.dtype)
        return bx

    def _stamp_eval(
        self,
        bx: torch.Tensor,
        *,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del round_idx
        if bx.shape[0] == 0 or self._source is None or not self.test_indices:
            return bx
        n = int(bx.shape[0])
        pool = self.test_indices
        chosen = np.asarray([pool[i % len(pool)] for i in range(n)], dtype=np.int64)
        sem_x, _ = self._source.fetch_batch(chosen, device=bx.device)
        sem_x = _adapt_image_shape(sem_x, (int(bx.shape[1]), int(bx.shape[2]), int(bx.shape[3])))
        bx[:] = sem_x.to(dtype=bx.dtype)
        return bx

    def make_eval_loader(
        self,
        hub: Any,
        *,
        batch_size: int,
        device: torch.device,
    ) -> _IndexedEvalLoader:
        eval_source = hub.train_source if self.test_split == "train" else hub.test_source
        return _IndexedEvalLoader(
            source=eval_source,
            norm_source=hub.test_source,
            indices=np.asarray(self.test_indices, dtype=np.int32),
            target_label=self.target_label,
            batch_size=batch_size,
            device=device,
        )


@POISONER_REGISTRY.register("edge_case")
@POISONER_REGISTRY.register("edge")
class EdgeCasePoisoner(BasePoisoner):
    """Edge-case OOD backdoor poisoner backed by lazy CHW uint8 memmap caches."""

    def __init__(
        self,
        train_path: str,
        test_path: Optional[str] = None,
        source_label: int = 0,
        target_label: int = 5,
        poison_ratio: float = 0.2,
        cache_dir: Optional[str] = None,
        seed: Optional[int] = 42,
        exclude_target_class: bool = False,
        **_: Any,
    ) -> None:
        if not train_path:
            raise ValueError("EdgeCasePoisoner requires a non-empty train_path.")
        super().__init__(
            target_label=target_label,
            poison_ratio=poison_ratio,
            seed=seed,
            exclude_target_class=exclude_target_class,
        )
        self.train_path = str(train_path)
        self.test_path = str(test_path) if test_path else str(train_path)
        self.source_label = int(source_label)
        self.cache_dir = cache_dir
        self._train_source: Optional[SharedDataSource] = None
        self._test_source: Optional[SharedDataSource] = None
        self._step_counter: int = 0

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_train_source"] = None
        state["_test_source"] = None
        state["_step_counter"] = 0
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._train_source = None
        self._test_source = None
        self._step_counter = 0

    def _get_edge_source(self, split: str) -> SharedDataSource:
        cached = self._train_source if split == "train" else self._test_source
        if cached is not None:
            return cached

        path = self.train_path if split == "train" else self.test_path
        src = load_external_data_source(
            path,
            default_label=self.source_label,
            cache_dir=self.cache_dir,
            split=split,
        )
        if split == "train":
            self._train_source = src
        else:
            self._test_source = src
        return src

    def _stamp_train(
        self,
        bx: torch.Tensor,
        mask: torch.Tensor,
        *,
        client_id: Optional[int] = None,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        n_poison = int(mask.sum().item())
        if n_poison <= 0:
            return bx

        edge_src = self._get_edge_source("train")
        self._step_counter += 1
        chosen = _select_pool_indices(
            len(edge_src),
            n_poison,
            seed=self.seed,
            tag="edge_train",
            client_id=client_id,
            round_idx=round_idx,
            step=self._step_counter,
        )
        edge_x, _ = edge_src.fetch_batch(chosen, device=bx.device)
        edge_x = _adapt_image_shape(edge_x, (int(bx.shape[1]), int(bx.shape[2]), int(bx.shape[3])))
        bx[mask] = edge_x.to(dtype=bx.dtype)
        return bx

    def _stamp_eval(
        self,
        bx: torch.Tensor,
        *,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del round_idx
        if bx.shape[0] == 0:
            return bx
        edge_src = self._get_edge_source("test")
        n = int(bx.shape[0])
        pool_len = len(edge_src)
        chosen = np.asarray([i % pool_len for i in range(n)], dtype=np.int64)
        edge_x, _ = edge_src.fetch_batch(chosen, device=bx.device)
        edge_x = _adapt_image_shape(edge_x, (int(bx.shape[1]), int(bx.shape[2]), int(bx.shape[3])))
        bx[:] = edge_x.to(dtype=bx.dtype)
        return bx

    def make_eval_loader(
        self,
        hub: Any,
        *,
        batch_size: int,
        device: torch.device,
    ) -> _IndexedEvalLoader:
        spec = hub.test_source.spec
        edge_src = self._get_edge_source("test")
        return _IndexedEvalLoader(
            source=edge_src,
            norm_source=hub.test_source,
            indices=np.arange(len(edge_src), dtype=np.int32),
            target_label=self.target_label,
            batch_size=batch_size,
            device=device,
            expected_chw=(spec.input_channels, *spec.image_size),
            override_orig_label=self.source_label,
        )
