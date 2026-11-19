"""Tiny-ImageNet-200 dataset specification for v2."""
from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
import zipfile
from typing import List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from data.registry import DATASET_REGISTRY
from data.source import ensure_chw_mmap_cache
from data.spec import AugmentSpec, DatasetPairSpec

TINY_IMAGENET_ZIP_NAME = "tiny-imagenet-200.zip"
_TINY_ZIP_MIN_BYTES = 80_000_000

TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_STD = (0.229, 0.224, 0.225)
TINY_IMAGENET_AUGMENT = AugmentSpec(random_crop_padding=8, random_horizontal_flip=True)


def _tiny_present(candidate: str) -> bool:
    return os.path.isfile(os.path.join(candidate, "wnids.txt"))


def _candidate_urls() -> List[str]:
    custom = os.environ.get("SAFEFL_TINY_IMAGENET_URL", "").strip()
    if custom:
        return [custom]
    return [
        "https://cs231n.stanford.edu/tiny-imagenet-200.zip",
        "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    ]


def _zip_looks_usable(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) < _TINY_ZIP_MIN_BYTES:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.replace("\\", "/").endswith("tiny-imagenet-200/wnids.txt"):
                    return True
            return False
    except zipfile.BadZipFile:
        return False


def _download_file(url: str, dst_path: str) -> None:
    part = dst_path + ".partial"
    if os.path.isfile(part):
        os.remove(part)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; SafeFL/2.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = resp.headers.get("Content-Length")
            total_n = int(total) if total else None
            block = 1024 * 64
            with open(part, "wb") as out, tqdm(
                total=total_n,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=TINY_IMAGENET_ZIP_NAME,
            ) as bar:
                while True:
                    chunk = resp.read(block)
                    if not chunk:
                        break
                    out.write(chunk)
                    bar.update(len(chunk))
        os.replace(part, dst_path)
    except BaseException:
        if os.path.isfile(part):
            os.remove(part)
        raise


def _download_tiny_imagenet_zip(zip_path: str) -> None:
    errors: List[str] = []
    transient = frozenset({429, 500, 502, 503, 504})

    for url in _candidate_urls():
        for attempt in range(3):
            try:
                if attempt:
                    wait = min(90, 5 * (2 ** (attempt - 1)))
                    print(f"Waiting {wait}s before retry: {url}")
                    time.sleep(wait)
                print(f"Downloading Tiny ImageNet: {url} -> {zip_path}")
                _download_file(url, zip_path)
                if _zip_looks_usable(zip_path):
                    return
                errors.append(f"{url}: downloaded zip failed integrity check")
                os.remove(zip_path)
                break
            except urllib.error.HTTPError as e:
                errors.append(f"{url}: HTTP {e.code} {e.reason}")
                if os.path.isfile(zip_path):
                    os.remove(zip_path)
                if e.code in transient and attempt < 2:
                    continue
                break
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                errors.append(f"{url}: {type(e).__name__}: {e}")
                if os.path.isfile(zip_path):
                    os.remove(zip_path)
                if attempt < 2:
                    continue
                break

    raise RuntimeError(
        "Failed to auto-download Tiny ImageNet: "
        + " | ".join(errors)
        + f"\nPlace {TINY_IMAGENET_ZIP_NAME} manually at {zip_path} "
        "or set SAFEFL_TINY_IMAGENET_URL."
    )


def _ensure_tiny_imagenet_on_disk(data_root: str) -> None:
    if os.environ.get("SAFEFL_SKIP_TINY_IMAGENET_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        return

    os.makedirs(data_root, exist_ok=True)
    zip_path = os.path.join(data_root, TINY_IMAGENET_ZIP_NAME)
    if os.path.isfile(zip_path) and not _zip_looks_usable(zip_path):
        os.remove(zip_path)

    if not os.path.isfile(zip_path):
        _download_tiny_imagenet_zip(zip_path)

    print(f"Extracting {zip_path} -> {data_root}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)


def _resolve_tiny_root(root: str) -> str:
    sub = os.path.join(root, "tiny-imagenet-200")
    if _tiny_present(sub):
        return sub
    if _tiny_present(root):
        return root

    _ensure_tiny_imagenet_on_disk(os.path.abspath(root))

    if _tiny_present(sub):
        return sub
    raise FileNotFoundError(f"Tiny ImageNet not found under {os.path.abspath(root)}.")


class _TinyImageNetSource(Dataset):
    """Raw RGB reader used only once when building the CHW v2 mmap cache."""

    def __init__(self, tiny_root: str, split_train: bool) -> None:
        self.paths: List[str] = []
        targets: List[int] = []

        with open(os.path.join(tiny_root, "wnids.txt"), "r", encoding="utf-8") as f:
            wnids = [ln.strip() for ln in f if ln.strip()]
        wnid_to_idx = {w: i for i, w in enumerate(wnids)}

        if split_train:
            train_dir = os.path.join(tiny_root, "train")
            for wnid in wnids:
                img_dir = os.path.join(train_dir, wnid, "images")
                for fn in sorted(os.listdir(img_dir)):
                    if fn.lower().endswith((".jpeg", ".jpg")):
                        self.paths.append(os.path.join(img_dir, fn))
                        targets.append(wnid_to_idx[wnid])
        else:
            val_img_dir = os.path.join(tiny_root, "val", "images")
            ann_path = os.path.join(tiny_root, "val", "val_annotations.txt")
            with open(ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    img_name, wnid = line.strip().split("\t")[:2]
                    self.paths.append(os.path.join(val_img_dir, img_name))
                    targets.append(wnid_to_idx[wnid])

        self.targets = np.asarray(targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        return Image.open(self.paths[index]).convert("RGB"), int(self.targets[index])


@DATASET_REGISTRY.register("tiny_imagenet")
def build_tiny_imagenet(root: str) -> DatasetPairSpec:
    # Check standard cache path first so cached runs don't even scan directories.
    default_sub = os.path.join(root, "tiny-imagenet-200")
    cache_dir = root if (_tiny_present(root) and not _tiny_present(default_sub)) else default_sub

    train_spec = ensure_chw_mmap_cache(
        raw_dataset_fn=lambda: _TinyImageNetSource(_resolve_tiny_root(root), split_train=True),
        cache_prefix=os.path.join(cache_dir, "tiny_imagenet_train"),
        name="tiny_imagenet",
        split="train",
        num_classes=200,
        mean=TINY_IMAGENET_MEAN,
        std=TINY_IMAGENET_STD,
        default_augment=TINY_IMAGENET_AUGMENT,
    )
    test_spec = ensure_chw_mmap_cache(
        raw_dataset_fn=lambda: _TinyImageNetSource(_resolve_tiny_root(root), split_train=False),
        cache_prefix=os.path.join(cache_dir, "tiny_imagenet_val"),
        name="tiny_imagenet",
        split="test",
        num_classes=200,
        mean=TINY_IMAGENET_MEAN,
        std=TINY_IMAGENET_STD,
        default_augment=None,
    )
    return DatasetPairSpec(name="tiny_imagenet", train=train_spec, test=test_spec)
