"""Zero-copy CHW uint8 memory-mapped & optional VRAM-resident data sources."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from data.spec import AugmentSpec, SplitSpec


def _to_chw_uint8(sample: Any) -> np.ndarray:
    """Convert a single image (PIL / ndarray / Tensor) into CHW uint8."""
    if torch.is_tensor(sample):
        arr = sample.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8, copy=False)
        return np.ascontiguousarray(arr)

    arr = np.asarray(sample)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        # Standard PIL / numpy image is HWC -> transpose to CHW.
        arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Expected 2D or 3D image sample, got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _try_extract_bulk_arrays(raw_dataset: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Fast vector path for datasets holding in-memory `.data` and `.targets`."""
    raw_data = getattr(raw_dataset, "data", None)
    raw_targets = getattr(raw_dataset, "targets", None)
    if raw_data is None or raw_targets is None:
        return None

    if torch.is_tensor(raw_data):
        data_np = raw_data.detach().cpu().numpy()
    elif isinstance(raw_data, np.ndarray):
        data_np = raw_data
    else:
        return None

    if data_np.dtype != np.uint8:
        return None

    if data_np.ndim == 3:
        # Grayscale (N, H, W) -> (N, 1, H, W)
        chw = np.ascontiguousarray(data_np[:, None, :, :])
    elif data_np.ndim == 4 and data_np.shape[-1] in (1, 3, 4):
        # Color HWC (N, H, W, C) -> (N, C, H, W)
        chw = np.ascontiguousarray(np.transpose(data_np, (0, 3, 1, 2)))
    elif data_np.ndim == 4 and data_np.shape[1] in (1, 3, 4):
        chw = np.ascontiguousarray(data_np)
    else:
        return None

    if torch.is_tensor(raw_targets):
        targets_np = raw_targets.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        targets_np = np.asarray(raw_targets, dtype=np.int64)

    if len(chw) != len(targets_np):
        return None
    return chw, np.ascontiguousarray(targets_np)


def _cache_is_valid(data_path: str, target_path: str, meta_path: str) -> Optional[Dict[str, Any]]:
    if not (os.path.isfile(data_path) and os.path.isfile(target_path) and os.path.isfile(meta_path)):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format_version") != 2 or meta.get("data_dtype") != "uint8":
            return None
        shape = tuple(int(x) for x in meta["data_shape"])
        if len(shape) != 4:
            return None
        expected_data_bytes = int(np.prod(shape, dtype=np.int64))
        expected_target_bytes = int(shape[0]) * 8
        if os.path.getsize(data_path) != expected_data_bytes:
            return None
        if os.path.getsize(target_path) != expected_target_bytes:
            return None
        return meta
    except (OSError, ValueError, KeyError, TypeError):
        return None


def ensure_chw_mmap_cache(
    raw_dataset_fn: Any,
    cache_prefix: str,
    *,
    name: str,
    split: str,
    num_classes: int,
    mean: Sequence[float],
    std: Sequence[float],
    default_augment: Optional[AugmentSpec] = None,
) -> SplitSpec:
    """Ensure a v2 CHW uint8 memmap cache exists and return its ``SplitSpec``.

    ``raw_dataset_fn`` may be either a ``Dataset`` instance or a zero-arg callable
    that constructs the raw dataset only when the cache does not yet exist!
    """
    prefix = f"{cache_prefix}_chw_v2"
    data_path = f"{prefix}_data.bin"
    target_path = f"{prefix}_targets.bin"
    meta_path = f"{prefix}_meta.json"

    meta = _cache_is_valid(data_path, target_path, meta_path)
    if meta is None:
        os.makedirs(os.path.dirname(os.path.abspath(data_path)), exist_ok=True)
        dataset = raw_dataset_fn() if callable(raw_dataset_fn) else raw_dataset_fn
        num_samples = len(dataset)
        if num_samples <= 0:
            raise RuntimeError(f"Dataset '{name}' ({split}) is empty.")

        pid = os.getpid()
        tmp_data = f"{data_path}.{pid}.tmp"
        tmp_target = f"{target_path}.{pid}.tmp"
        tmp_meta = f"{meta_path}.{pid}.tmp"

        try:
            bulk = _try_extract_bulk_arrays(dataset)
            if bulk is not None:
                chw, targets_np = bulk
                shape = tuple(int(x) for x in chw.shape)
                with open(tmp_data, "wb") as fd:
                    chw.tofile(fd)
                with open(tmp_target, "wb") as ft:
                    targets_np.tofile(ft)
            else:
                first_img, _ = dataset[0]
                first_chw = _to_chw_uint8(first_img)
                shape = (num_samples, *first_chw.shape)
                fp_data = np.memmap(tmp_data, dtype=np.uint8, mode="w+", shape=shape)
                fp_targets = np.memmap(tmp_target, dtype=np.int64, mode="w+", shape=(num_samples,))
                for idx in tqdm(range(num_samples), desc=f"Caching {name}/{split} (CHW v2)"):
                    img, label = dataset[idx]
                    fp_data[idx] = _to_chw_uint8(img)
                    fp_targets[idx] = int(label)
                fp_data.flush()
                fp_targets.flush()
                del fp_data
                del fp_targets

            meta = {
                "format_version": 2,
                "data_shape": list(shape),
                "data_dtype": "uint8",
                "target_shape": [shape[0]],
                "target_dtype": "int64",
            }
            with open(tmp_meta, "w", encoding="utf-8") as fm:
                json.dump(meta, fm)

            os.replace(tmp_data, data_path)
            os.replace(tmp_target, target_path)
            os.replace(tmp_meta, meta_path)
        finally:
            for leftover in (tmp_data, tmp_target, tmp_meta):
                if os.path.isfile(leftover):
                    try:
                        os.remove(leftover)
                    except OSError:
                        pass

    data_shape: Tuple[int, int, int, int] = (
        int(meta["data_shape"][0]),
        int(meta["data_shape"][1]),
        int(meta["data_shape"][2]),
        int(meta["data_shape"][3]),
    )
    return SplitSpec(
        name=name,
        split=split,
        data_path=os.path.abspath(data_path),
        target_path=os.path.abspath(target_path),
        data_shape=data_shape,
        num_classes=int(num_classes),
        mean=tuple(float(x) for x in mean),
        std=tuple(float(x) for x in std),
        default_augment=default_augment,
    )


class SharedDataSource:
    """Read-only CHW uint8 data source backed by OS page-cache mmap or GPU VRAM.

    Pickling a ``SharedDataSource`` serializes ONLY its 300-byte ``SplitSpec``;
    worker processes reopen the read-only ``np.memmap`` lazily on first use.
    """

    def __init__(self, spec: SplitSpec) -> None:
        self.spec = spec
        self._data_mmap: Optional[np.memmap] = None
        self._targets_mmap: Optional[np.memmap] = None
        self._gpu_data: Optional[torch.Tensor] = None
        self._gpu_targets: Optional[torch.Tensor] = None
        self._norm_cache: Dict[Tuple[torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}

    def __getstate__(self) -> Dict[str, Any]:
        return {"spec": self.spec}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.spec = state["spec"]
        self._data_mmap = None
        self._targets_mmap = None
        self._gpu_data = None
        self._gpu_targets = None
        self._norm_cache = {}

    def _ensure_mmap(self) -> None:
        if self._data_mmap is None:
            self._data_mmap = np.memmap(
                self.spec.data_path,
                dtype=np.uint8,
                mode="r",
                shape=self.spec.data_shape,
            )
        if self._targets_mmap is None:
            self._targets_mmap = np.memmap(
                self.spec.target_path,
                dtype=np.int64,
                mode="r",
                shape=(self.spec.num_samples,),
            )

    @property
    def data(self) -> np.memmap:
        self._ensure_mmap()
        assert self._data_mmap is not None
        return self._data_mmap

    @property
    def targets(self) -> np.ndarray:
        self._ensure_mmap()
        assert self._targets_mmap is not None
        return self._targets_mmap

    @property
    def num_classes(self) -> int:
        return self.spec.num_classes

    @property
    def default_augment(self) -> Optional[AugmentSpec]:
        return self.spec.default_augment

    def __len__(self) -> int:
        return self.spec.num_samples

    def preload(self, device: Union[str, torch.device]) -> None:
        """Load raw uint8 images and int64 labels onto ``device`` (e.g. VRAM)."""
        dev = torch.device(device)
        if self._gpu_data is not None and self._gpu_data.device == dev:
            return
        self._ensure_mmap()
        assert self._data_mmap is not None and self._targets_mmap is not None
        # Copy once from mmap into contiguous device tensors.
        self._gpu_data = torch.from_numpy(np.array(self._data_mmap, copy=True)).to(dev)
        self._gpu_targets = torch.from_numpy(np.array(self._targets_mmap, copy=True)).to(dev)

    def evict_device(self) -> None:
        """Release preloaded device buffers."""
        self._gpu_data = None
        self._gpu_targets = None
        self._norm_cache.clear()

    def fetch_batch(
        self,
        indices: np.ndarray,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather a batch of images in physical ``[0.0, 1.0]`` float32 and int64 labels.

        Both returned tensors own freshly allocated storage, so downstream
        in-place poisoning / normalization never mutates shared buffers.
        """
        dev = torch.device(device) if device is not None else torch.device("cpu")

        if self._gpu_data is not None and self._gpu_data.device == dev:
            assert self._gpu_targets is not None
            idx_t = torch.as_tensor(indices, dtype=torch.long, device=dev)
            bx = self._gpu_data.index_select(0, idx_t).to(dtype=torch.float32).div_(255.0)
            by = self._gpu_targets.index_select(0, idx_t)
            return bx, by

        self._ensure_mmap()
        assert self._data_mmap is not None and self._targets_mmap is not None
        # 1D integer fancy indexing on memmap always allocates a writable [B, C, H, W] copy.
        idx_arr = np.asarray(indices, dtype=np.intp).reshape(-1)
        raw_u8 = torch.from_numpy(np.asarray(self._data_mmap[idx_arr], dtype=np.uint8))
        raw_y = torch.from_numpy(np.asarray(self._targets_mmap[idx_arr], dtype=np.int64))
        if dev.type != "cpu":
            raw_u8 = raw_u8.to(dev, non_blocking=True)
            raw_y = raw_y.to(dev, non_blocking=True)
        bx = raw_u8.to(dtype=torch.float32).div_(255.0)
        return bx, raw_y

    def normalize_(self, bx: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise ``(bx - mean) / std`` in-place on a 4D batch tensor."""
        key = (bx.device, bx.dtype)
        cached = self._norm_cache.get(key)
        if cached is None:
            c = bx.shape[1]
            mean_t = torch.tensor(self.spec.mean, device=bx.device, dtype=bx.dtype).view(1, c, 1, 1)
            std_t = torch.tensor(self.spec.std, device=bx.device, dtype=bx.dtype).view(1, c, 1, 1)
            cached = (mean_t, std_t)
            self._norm_cache[key] = cached
        mean_t, std_t = cached
        return bx.sub_(mean_t).div_(std_t)


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class _RawExternalDataset:
    """Adapter exposing external image collections to ``ensure_chw_mmap_cache``."""

    def __init__(self, items: Any, targets: Optional[np.ndarray] = None) -> None:
        self._items = items
        self._n = len(items)
        self._targets = (
            np.asarray(targets, dtype=np.int64)
            if targets is not None
            else np.zeros(self._n, dtype=np.int64)
        )
        if isinstance(items, np.ndarray) and items.dtype == np.uint8 and items.ndim in (3, 4):
            self.data = items
            self.targets = self._targets

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        return self._items[idx], int(self._targets[idx])


def _load_external_items(path: str, default_label: int) -> _RawExternalDataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"External dataset path not found: {path!r}")

    if os.path.isdir(path):
        from PIL import Image

        files = sorted(
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if fname.lower().endswith(_IMAGE_EXTS)
        )
        if not files:
            raise ValueError(f"No image files found in directory: {path!r}")
        images = [np.asarray(Image.open(fp).convert("RGB"), dtype=np.uint8) for fp in files]
        arr = np.stack(images, axis=0)
        return _RawExternalDataset(arr, np.full(len(arr), int(default_label), dtype=np.int64))

    lower = path.lower()
    obj: Any
    if lower.endswith((".pkl", ".pickle")):
        with open(path, "rb") as f:
            obj = pickle.load(f)
    elif lower.endswith(".npy"):
        obj = np.load(path, allow_pickle=True)
    elif lower.endswith(".npz"):
        npz = np.load(path, allow_pickle=True)
        if "data" in npz:
            imgs = npz["data"]
        elif "images" in npz:
            imgs = npz["images"]
        else:
            imgs = npz[npz.files[0]]
        lbls = npz["targets"] if "targets" in npz else npz.get("labels")
        obj = (imgs, lbls) if lbls is not None else imgs
    elif lower.endswith((".pt", ".pth")):
        obj = torch.load(path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"Unsupported external dataset format: {path!r}")

    targets: Optional[np.ndarray] = None
    if isinstance(obj, dict):
        imgs = obj.get("data", obj.get("images"))
        if imgs is None:
            raise ValueError(f"Dict in {path!r} must contain 'data' or 'images'.")
        lbls = obj.get("targets", obj.get("labels"))
        if lbls is not None:
            targets = np.asarray(lbls, dtype=np.int64)
        obj = imgs
    elif isinstance(obj, tuple) and len(obj) == 2:
        obj, lbls = obj
        if lbls is not None:
            targets = np.asarray(lbls, dtype=np.int64)

    if targets is None:
        targets = np.full(len(obj), int(default_label), dtype=np.int64)

    if isinstance(obj, list) and obj and isinstance(obj[0], np.ndarray):
        try:
            obj = np.stack(obj, axis=0)
        except ValueError:
            pass

    return _RawExternalDataset(obj, targets)


def load_external_data_source(
    path: str,
    *,
    default_label: int = 0,
    cache_dir: Optional[str] = None,
    split: str = "external",
) -> SharedDataSource:
    """Load an external image folder or tensor/array file into a cached ``SharedDataSource``."""
    abs_path = os.path.abspath(path)
    base_cache_dir = cache_dir or os.path.join(os.path.dirname(abs_path), ".safefl_cache")
    digest = hashlib.md5(f"{abs_path}|{int(default_label)}".encode("utf-8")).hexdigest()[:12]
    cache_prefix = os.path.join(base_cache_dir, f"ext_{split}_{digest}")

    split_spec = ensure_chw_mmap_cache(
        lambda: _load_external_items(abs_path, int(default_label)),
        cache_prefix,
        name=f"ext_{digest}",
        split=split,
        num_classes=max(10, int(default_label) + 1),
        mean=(0.0,),
        std=(1.0,),
        default_augment=None,
    )
    return SharedDataSource(split_spec)
