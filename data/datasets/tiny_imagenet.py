import os
import time
import urllib.error
import urllib.request
import zipfile
from typing import List

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm

from core.utils.mmap_dataset import MemoryMappedDataset
from data.dataset_store import DatasetStore
from data.registry import dataset_registry

# 官方包在 CS231N; torchvision 无内置. 若 503 可设环境变量 SAFEFL_TINY_IMAGENET_URL 为可访问的直链.
TINY_IMAGENET_ZIP_NAME = "tiny-imagenet-200.zip"

# 约 240MB 压缩包, 明显过小视为下载失败
_TINY_ZIP_MIN_BYTES = 80_000_000

# 常用 ImageNet 统计量, Tiny ImageNet 文献中普遍采用
TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_STD = (0.229, 0.224, 0.225)

_base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
])

_aug_transform = transforms.Compose([
    transforms.RandomCrop(64, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
])


def _tiny_present(candidate: str) -> bool:
    return os.path.isfile(os.path.join(candidate, "wnids.txt"))


def _candidate_urls() -> List[str]:
    custom = os.environ.get("SAFEFL_TINY_IMAGENET_URL", "").strip()
    if custom:
        return [custom]
    # 先试 HTTPS, 再试 HTTP (部分网络对二者策略不同)
    return [
        "https://cs231n.stanford.edu/tiny-imagenet-200.zip",
        "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    ]


def _zip_looks_usable(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) < _TINY_ZIP_MIN_BYTES:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for n in zf.namelist():
                norm = n.replace("\\", "/")
                if norm.endswith("tiny-imagenet-200/wnids.txt"):
                    return True
            return False
    except zipfile.BadZipFile:
        return False


def _download_file(url: str, dst_path: str) -> None:
    """流式下载到临时文件, 成功后再替换目标路径."""
    part = dst_path + ".partial"
    if os.path.isfile(part):
        os.remove(part)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; SafeFL/1.0)"},
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
    """多 URL, 带重试; 503/502 等会退避后重试."""
    errors: List[str] = []
    transient = frozenset({429, 500, 502, 503, 504})

    for url in _candidate_urls():
        for attempt in range(3):
            try:
                if attempt:
                    wait = min(90, 5 * (2 ** (attempt - 1)))
                    print(f"下载重试前等待 {wait}s: {url}")
                    time.sleep(wait)
                print(f"正在下载 Tiny ImageNet: {url} -> {zip_path}")
                _download_file(url, zip_path)
                if _zip_looks_usable(zip_path):
                    return
                errors.append(f"{url}: 已下载但 zip 校验失败 (体积或内容异常)")
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
        "无法自动下载 Tiny ImageNet. 详情: "
        + " | ".join(errors)
        + "\n\nHTTP 503/502 通常表示官方站点暂时不可用或限流, 与本地代码无关."
        "\n可行办法: 换时段/代理/浏览器手动下载 zip 后放到\n  "
        f"{zip_path}\n"
        "或设置环境变量 SAFEFL_TINY_IMAGENET_URL 指向你可访问的同一数据包直链."
    )


def _ensure_tiny_imagenet_on_disk(data_root: str) -> None:
    """
    若本地尚无数据则下载并解压到 `data_root/tiny-imagenet-200/`.
    与 torchvision.CIFAR10(download=True) 类似, 但 Tiny ImageNet 不在 torchvision 内置列表中.

    离线环境可设置环境变量 SAFEFL_SKIP_TINY_IMAGENET_DOWNLOAD=1 跳过自动下载 (将保留原有报错).
    """
    if os.environ.get("SAFEFL_SKIP_TINY_IMAGENET_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        return

    sub = os.path.join(data_root, "tiny-imagenet-200")
    if _tiny_present(sub):
        return
    if _tiny_present(data_root):
        return

    os.makedirs(data_root, exist_ok=True)
    zip_path = os.path.join(data_root, TINY_IMAGENET_ZIP_NAME)
    if os.path.isfile(zip_path) and not _zip_looks_usable(zip_path):
        print("已存在不完整或损坏的 zip, 将删除后重新下载.")
        os.remove(zip_path)

    if not os.path.isfile(zip_path):
        _download_tiny_imagenet_zip(zip_path)

    print(f"正在解压 {zip_path} -> {data_root}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)
    except zipfile.BadZipFile as e:
        os.remove(zip_path)
        raise RuntimeError(
            f"解压失败 (zip 损坏), 已删除 {zip_path}. 请重新运行或手动下载同一文件后放回该路径."
        ) from e
    if not _tiny_present(sub):
        raise RuntimeError(
            f"解压后仍未找到 {os.path.join(sub, 'wnids.txt')}, 请检查 zip 是否完整或手动解压."
        )


def _tiny_imagenet_root(root: str) -> str:
    """解析 Tiny ImageNet 根目录: `root/tiny-imagenet-200` 或 `root` 本身就是该目录; 无则尝试自动下载."""
    sub = os.path.join(root, "tiny-imagenet-200")
    if os.path.isdir(sub) and _tiny_present(sub):
        return sub
    if _tiny_present(root):
        return root

    _ensure_tiny_imagenet_on_disk(os.path.abspath(root))

    if os.path.isdir(sub) and _tiny_present(sub):
        return sub
    if _tiny_present(root):
        return root

    raise FileNotFoundError(
        f"未找到 Tiny ImageNet. 已尝试下载到 {os.path.abspath(root)}. "
        "若需离线使用, 请手动放入 tiny-imagenet-200/ 或设置正确的 data.root, "
        "并可用 SAFEFL_SKIP_TINY_IMAGENET_DOWNLOAD=1 关闭自动下载."
    )


class _TinyImageNetSource(Dataset):
    """原始 RGB 图像, 供 MemoryMappedDataset 建缓存; `split_train` 为 True 用 train/, 否则用 val+标注."""

    def __init__(self, root: str, split_train: bool) -> None:
        self.root = root
        self.paths: List[str] = []
        targets: List[int] = []

        wnids_path = os.path.join(root, "wnids.txt")
        with open(wnids_path, "r", encoding="utf-8") as f:
            wnids = [ln.strip() for ln in f if ln.strip()]
        wnid_to_idx = {w: i for i, w in enumerate(wnids)}

        if split_train:
            train_dir = os.path.join(root, "train")
            for wnid in wnids:
                img_dir = os.path.join(train_dir, wnid, "images")
                if not os.path.isdir(img_dir):
                    continue
                for fn in sorted(os.listdir(img_dir)):
                    if not fn.lower().endswith((".jpeg", ".jpg")):
                        continue
                    self.paths.append(os.path.join(img_dir, fn))
                    targets.append(wnid_to_idx[wnid])
        else:
            val_img_dir = os.path.join(root, "val", "images")
            ann_path = os.path.join(root, "val", "val_annotations.txt")
            with open(ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    fname, wnid = parts[0], parts[1]
                    self.paths.append(os.path.join(val_img_dir, fname))
                    targets.append(wnid_to_idx[wnid])

        self.targets = np.array(targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        img = Image.open(self.paths[index]).convert("RGB")
        t = int(self.targets[index])
        return img, t


def _build_tiny_impl(root: str, split_train: bool, use_aug: bool) -> DatasetStore:
    tiny_root = _tiny_imagenet_root(root)
    split_name = "train" if split_train else "val"
    cache_path = os.path.join(tiny_root, f"_mmap_{split_name}")
    raw = _TinyImageNetSource(tiny_root, split_train=split_train)
    if len(raw) == 0:
        raise RuntimeError(f"Tiny ImageNet ({split_name}) 在 {tiny_root} 下未读到任何样本.")

    final_transform = _aug_transform if (split_train and use_aug) else _base_transform
    mmap_dataset = MemoryMappedDataset(
        original_dataset=raw,
        cache_path=cache_path,
        transform=final_transform,
    )
    inner_split = "train" if use_aug else "train_plain" if split_train else "test"
    return DatasetStore("tiny_imagenet", inner_split, mmap_dataset)


@dataset_registry.register("tiny_imagenet_train_aug")
def build_tiny_imagenet_train_aug(root: str, is_train: bool) -> DatasetStore:
    del is_train
    return _build_tiny_impl(root, split_train=True, use_aug=True)


@dataset_registry.register("tiny_imagenet_train_plain")
def build_tiny_imagenet_train_plain(root: str, is_train: bool) -> DatasetStore:
    del is_train
    return _build_tiny_impl(root, split_train=True, use_aug=False)


@dataset_registry.register("tiny_imagenet_test_plain")
def build_tiny_imagenet_test_plain(root: str, is_train: bool) -> DatasetStore:
    del is_train
    return _build_tiny_impl(root, split_train=False, use_aug=False)
