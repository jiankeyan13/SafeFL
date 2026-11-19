"""查看联邦学习数据划分: 全体客户端标签分布, 以及单个客户端的标签构成."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from data.constants import SPLIT_TRAIN, client_owner
from data.dataset_store import DatasetStore
from data.task import TaskSet

ClientId = Union[int, str]

# 常见数据集的类别名, 便于读图; 未知数据集回退为数字标签.
_DATASET_CLASS_NAMES: Dict[str, Tuple[str, ...]] = {
    "cifar10": (
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ),
    "fashionmnist": (
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ),
}


def resolve_owner_id(client_id: ClientId) -> str:
    """将 int / '0' / 'client_0' 统一成 TaskSet 使用的 owner 字符串."""
    if isinstance(client_id, int):
        return client_owner(client_id)
    text = str(client_id).strip()
    if text.isdigit():
        return client_owner(int(text))
    return text


def _sort_client_ids(client_ids: Sequence[str]) -> List[str]:
    def _key(cid: str) -> Tuple[int, Union[int, str]]:
        suffix = cid.rsplit("_", 1)[-1]
        if suffix.lstrip("-").isdigit():
            return (0, int(suffix))
        return (1, cid)

    return sorted(client_ids, key=_key)


def _class_names_for(dataset_name: Optional[str], labels: np.ndarray) -> List[str]:
    names = _DATASET_CLASS_NAMES.get((dataset_name or "").lower())
    if names is not None and len(names) >= int(labels.max(initial=-1)) + 1:
        return [names[int(lab)] for lab in labels]
    return [str(int(lab)) for lab in labels]


def collect_label_matrix(
    task_set: TaskSet,
    stores: Dict[str, DatasetStore],
    split: str = SPLIT_TRAIN,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    统计每个客户端在指定 split 上的标签计数.

    Returns:
        matrix: shape (n_clients, n_classes), 第 i 行是第 i 个客户端各类样本数.
        client_ids: 与 matrix 行对齐的 owner id.
        labels: 与 matrix 列对齐的类别取值.
    """
    client_ids = _sort_client_ids(task_set.list_client_ids(exclude_server=True))
    if not client_ids:
        raise ValueError("TaskSet 中没有客户端任务, 无法统计数据分布.")

    missing = [cid for cid in client_ids if not task_set.has_task(cid, split)]
    if missing:
        preview = ", ".join(missing[:5])
        extra = "" if len(missing) <= 5 else f" 等 {len(missing)} 个"
        raise KeyError(f"以下客户端缺少 split={split}: {preview}{extra}")

    per_client: Dict[str, np.ndarray] = {}
    for cid in client_ids:
        task = task_set.get_task(cid, split)
        store = stores[task.dataset_tag]
        full_labels = np.asarray(store.get_label())
        indices = np.asarray(task.indices, dtype=np.int64)
        per_client[cid] = full_labels[indices]

    labels = np.unique(np.concatenate(list(per_client.values())))
    col = {int(lab): i for i, lab in enumerate(labels)}
    matrix = np.zeros((len(client_ids), len(labels)), dtype=np.int64)
    for row, cid in enumerate(client_ids):
        values, counts = np.unique(per_client[cid], return_counts=True)
        for value, count in zip(values, counts):
            matrix[row, col[int(value)]] = int(count)
    return matrix, client_ids, labels


@dataclass
class DistributionStats:
    """数据划分的汇总统计."""

    num_clients: int
    num_classes: int
    total_samples: int
    samples_per_client: np.ndarray
    classes_per_client: np.ndarray
    mean_kl: float

    def format(self) -> str:
        spc = self.samples_per_client
        cpc = self.classes_per_client
        lines = [
            f"clients={self.num_clients}, classes={self.num_classes}, samples={self.total_samples}",
            (
                f"samples/client: min={int(spc.min())}, max={int(spc.max())}, "
                f"mean={float(spc.mean()):.1f}, std={float(spc.std()):.1f}"
            ),
            (
                f"classes/client: min={int(cpc.min())}, max={int(cpc.max())}, "
                f"mean={float(cpc.mean()):.1f}"
            ),
            f"mean KL(client || global)={self.mean_kl:.4f}",
        ]
        return "\n".join(lines)


def compute_stats(matrix: np.ndarray) -> DistributionStats:
    """由计数矩阵计算样本量, 类别覆盖和相对全局分布的 KL."""
    if matrix.size == 0:
        raise ValueError("空的标签计数矩阵.")
    samples = matrix.sum(axis=1)
    classes = (matrix > 0).sum(axis=1)
    total = int(matrix.sum())
    client_p = matrix / np.clip(samples[:, None], 1, None)
    global_p = matrix.sum(axis=0) / max(total, 1)
    kl = (client_p * np.log((client_p + 1e-12) / (global_p[None, :] + 1e-12))).sum(axis=1)
    return DistributionStats(
        num_clients=int(matrix.shape[0]),
        num_classes=int(matrix.shape[1]),
        total_samples=total,
        samples_per_client=samples,
        classes_per_client=classes,
        mean_kl=float(kl.mean()) if kl.size else 0.0,
    )


def _require_plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("绘制数据分布需要 matplotlib, 请先安装: pip install matplotlib") from exc
    return plt


def _setup_cjk_font(plt) -> None:
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Source Han Sans SC",
        "Droid Sans Fallback",
        "SimHei",
        "Microsoft YaHei",
    ):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _save_or_show(fig, save_path: Optional[str], show: bool) -> None:
    plt = _require_plt()
    if save_path:
        _ensure_parent(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _tick_step(n: int, max_ticks: int = 20) -> int:
    if n <= max_ticks:
        return 1
    return max(1, int(np.ceil(n / max_ticks)))


def _short_client_label(owner_id: str) -> str:
    suffix = owner_id.rsplit("_", 1)[-1]
    return suffix if suffix.lstrip("-").isdigit() else owner_id


def _label_red_green_cmap(n_classes: int):
    """label 0 为红, 末类为绿, 中间标签沿红-黄-绿渐变."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "label_red_green",
        ["#d73027", "#ffffbf", "#1a9850"],
        N=max(int(n_classes), 2),
    )


def _label_red_green_colors(n_classes: int):
    cmap = _label_red_green_cmap(n_classes)
    if n_classes <= 1:
        return [cmap(0.0)]
    return [cmap(i / (n_classes - 1)) for i in range(n_classes)]


def _contrast_text_color(rgba) -> str:
    r, g, b = rgba[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.55 else "black"


def simulate_badnets_train_poison(
    labels: np.ndarray,
    poison_ratio: float = 0.5,
    target_label: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    复现 PoisonedDatasetWrapper(mode='train') 的投毒抽样.
    随机选取 poison_ratio 比例样本, 训练标签改为 target_label.
    """
    labels = np.asarray(labels)
    n = int(labels.size)
    mask = np.zeros(n, dtype=bool)
    num_poison = int(n * float(poison_ratio))
    if num_poison > 0 and n > 0:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(n, size=min(num_poison, n), replace=False)
        mask[chosen] = True
    effective = labels.copy()
    effective[mask] = int(target_label)
    return effective, mask


def _class_counts_with_poison(
    effective: np.ndarray,
    poison_mask: np.ndarray,
    class_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(len(class_labels), dtype=np.int64)
    poison_counts = np.zeros(len(class_labels), dtype=np.int64)
    for i, lab in enumerate(class_labels):
        sel = effective == lab
        counts[i] = int(sel.sum())
        poison_counts[i] = int((sel & poison_mask).sum())
    return counts, poison_counts


def _draw_proportion_stack(
    ax,
    shares: np.ndarray,
    colors,
    poison_fracs: Optional[np.ndarray] = None,
    width: float = 0.55,
    annotate_min: float = 0.035,
) -> None:
    """自下而上按标签堆叠占比柱; poison_fracs 表示该类中恶意样本占比, 从左向右叠竖虚线, 无边框."""
    if poison_fracs is None:
        poison_fracs = np.zeros_like(shares, dtype=np.float64)
    bottom = 0.0
    for share, color, pfrac in zip(shares, colors, poison_fracs):
        share = float(share)
        pfrac = float(min(max(pfrac, 0.0), 1.0))
        if share <= 0:
            continue
        ax.bar(
            0,
            share,
            bottom=bottom,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            align="center",
        )
        if pfrac > 0:
            mal_w = width * pfrac
            x_left = -width / 2.0
            x_right = x_left + mal_w
            y0 = bottom
            y1 = bottom + share
            spacing = 0.048
            n_lines = max(2, int(np.floor(mal_w / spacing)) + 1)
            pad = min(0.012, 0.25 * mal_w)
            xs = np.linspace(x_left + pad, x_right - pad, n_lines)
            for x in xs:
                ax.plot(
                    [x, x],
                    [y0, y1],
                    linestyle=(0, (6.0, 3.2)),
                    color="#404040",
                    linewidth=1.0,
                    solid_capstyle="butt",
                    clip_on=True,
                    zorder=3,
                )
        if share >= annotate_min:
            ax.text(
                0,
                bottom + share / 2.0,
                f"{share:.1%}",
                ha="center",
                va="center",
                color=_contrast_text_color(color),
                fontsize=8,
                zorder=4,
            )
        bottom += share


class DataDistributionVisualizer:
    """
    联邦学习数据分布可视化.

    典型用法:
        vis = DataDistributionVisualizer(task_set, stores)
        print(vis.summary())
        vis.plot_system(save_dir="./viz")
        vis.plot_client(0, save_path="./viz/client_0.png")
    """

    def __init__(
        self,
        task_set: TaskSet,
        stores: Dict[str, DatasetStore],
        split: str = SPLIT_TRAIN,
        dataset_name: Optional[str] = None,
        class_names: Optional[Sequence[str]] = None,
    ):
        self.task_set = task_set
        self.stores = stores
        self.split = split
        self.matrix, self.client_ids, self.labels = collect_label_matrix(
            task_set, stores, split=split
        )
        inferred = dataset_name or self._infer_dataset_name()
        self.dataset_name = inferred
        if class_names is not None:
            if len(class_names) != len(self.labels):
                raise ValueError(
                    f"class_names 长度 ({len(class_names)}) 与类别数 ({len(self.labels)}) 不一致."
                )
            self.class_names = list(class_names)
        else:
            self.class_names = _class_names_for(inferred, self.labels)
        self.stats = compute_stats(self.matrix)

    def _infer_dataset_name(self) -> Optional[str]:
        if not self.client_ids:
            return None
        task = self.task_set.get_task(self.client_ids[0], self.split)
        store = self.stores.get(task.dataset_tag)
        return getattr(store, "name", None)

    @classmethod
    def from_task_set(
        cls,
        task_set: TaskSet,
        stores: Dict[str, DatasetStore],
        split: str = SPLIT_TRAIN,
        dataset_name: Optional[str] = None,
    ) -> "DataDistributionVisualizer":
        return cls(task_set, stores, split=split, dataset_name=dataset_name)

    @classmethod
    def from_data_config(
        cls,
        data_config,
        num_clients: int,
        seed: int = 42,
        split: str = SPLIT_TRAIN,
    ) -> "DataDistributionVisualizer":
        """按 DataConfig 重建划分后再可视化, 无需启动完整训练."""
        from core.bootstrap import bootstrap_registries
        from data.partitioner import build_partitioner
        from data.task_generator import TaskGenerator

        bootstrap_registries()
        partitioner = build_partitioner(data_config.partitioner, seed)
        generator = TaskGenerator(
            dataset_name=data_config.dataset,
            root=data_config.root,
            partitioner=partitioner,
            num_clients=num_clients,
            seed=seed,
            enable_proxy=data_config.enable_proxy,
        )
        task_set, stores = generator.generate()
        return cls(
            task_set,
            stores,
            split=split,
            dataset_name=data_config.dataset,
        )

    def summary(self) -> str:
        return self.stats.format()

    def client_counts(self, client_id: ClientId) -> np.ndarray:
        """返回指定客户端各类别样本数, 与 class_names 对齐."""
        owner = resolve_owner_id(client_id)
        try:
            row = self.client_ids.index(owner)
        except ValueError as exc:
            raise KeyError(f"未找到客户端 {owner}") from exc
        return self.matrix[row].copy()

    def client_sample_labels(self, client_id: ClientId) -> np.ndarray:
        """返回该客户端样本标签, 顺序与 Task.indices / Subset 一致."""
        owner = resolve_owner_id(client_id)
        if not self.task_set.has_task(owner, self.split):
            raise KeyError(f"未找到客户端 {owner} 的 split={self.split}")
        task = self.task_set.get_task(owner, self.split)
        store = self.stores[task.dataset_tag]
        full_labels = np.asarray(store.get_label())
        return full_labels[np.asarray(task.indices, dtype=np.int64)]

    def poisoned_client_counts(
        self,
        client_id: ClientId,
        poison_ratio: float = 0.5,
        target_label: int = 5,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        按 BadNets 默认训练投毒统计有效标签.
        Returns:
            counts: 投毒后各类样本数 (恶意样本计入 target_label).
            poison_counts: 各类中实际被投毒的样本数.
        """
        original = self.client_sample_labels(client_id)
        effective, mask = simulate_badnets_train_poison(
            original,
            poison_ratio=poison_ratio,
            target_label=target_label,
            seed=seed,
        )
        return _class_counts_with_poison(effective, mask, self.labels)

    def plot_heatmap(
        self,
        save_path: Optional[str] = None,
        show: bool = False,
        title: Optional[str] = None,
    ):
        """客户端 x 类别 的样本数热力图."""
        plt = _require_plt()
        n_clients, n_classes = self.matrix.shape
        fig_w = max(7.0, min(18.0, n_classes * 0.55 + 2.5))
        fig_h = max(5.0, min(22.0, n_clients * 0.18 + 2.0))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        image = ax.imshow(self.matrix, aspect="auto", interpolation="nearest", cmap="YlOrRd")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="samples")

        x_step = _tick_step(n_classes, max_ticks=24)
        y_step = _tick_step(n_clients, max_ticks=30)
        ax.set_xticks(np.arange(0, n_classes, x_step))
        ax.set_xticklabels(
            [self.class_names[i] for i in range(0, n_classes, x_step)],
            rotation=45,
            ha="right",
        )
        ax.set_yticks(np.arange(0, n_clients, y_step))
        ax.set_yticklabels(
            [_short_client_label(self.client_ids[i]) for i in range(0, n_clients, y_step)]
        )
        ax.set_xlabel("class")
        ax.set_ylabel("client")
        ax.set_title(title or "Client-class label heatmap")

        if n_clients * n_classes <= 200:
            vmax = max(int(self.matrix.max()), 1)
            for i in range(n_clients):
                for j in range(n_classes):
                    value = int(self.matrix[i, j])
                    color = "white" if value > 0.55 * vmax else "black"
                    ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=7)

        fig.tight_layout()
        _save_or_show(fig, save_path, show)
        return fig

    def plot_stacked_bar(
        self,
        save_path: Optional[str] = None,
        show: bool = False,
        title: Optional[str] = None,
    ):
        """各客户端样本量按类别堆叠, 便于对比异构程度."""
        plt = _require_plt()
        n_clients, n_classes = self.matrix.shape
        fig_w = max(8.0, min(20.0, n_clients * 0.22 + 2.5))
        fig, ax = plt.subplots(figsize=(fig_w, 5.5))
        cmap = plt.get_cmap("tab20")
        bottoms = np.zeros(n_clients, dtype=np.float64)
        x = np.arange(n_clients)
        for k in range(n_classes):
            ax.bar(
                x,
                self.matrix[:, k],
                bottom=bottoms,
                width=0.85,
                color=cmap(k % 20),
                label=self.class_names[k],
                linewidth=0,
            )
            bottoms += self.matrix[:, k]

        step = _tick_step(n_clients, max_ticks=30)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(
            [_short_client_label(cid) for cid in self.client_ids[::step]],
            rotation=45,
            ha="right",
        )
        ax.set_xlabel("client")
        ax.set_ylabel("samples")
        ax.set_title(title or "Per-client sample composition")
        if n_classes <= 12:
            ax.legend(loc="upper right", ncol=min(n_classes, 4), fontsize=8, frameon=False)
        fig.tight_layout()
        _save_or_show(fig, save_path, show)
        return fig

    def plot_client_sizes(
        self,
        save_path: Optional[str] = None,
        show: bool = False,
        title: Optional[str] = None,
    ):
        """各客户端总样本量柱状图."""
        plt = _require_plt()
        n_clients = len(self.client_ids)
        fig_w = max(8.0, min(20.0, n_clients * 0.22 + 2.5))
        fig, ax = plt.subplots(figsize=(fig_w, 4.5))
        x = np.arange(n_clients)
        ax.bar(x, self.stats.samples_per_client, color="#4C78A8", width=0.85)
        step = _tick_step(n_clients, max_ticks=30)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(
            [_short_client_label(cid) for cid in self.client_ids[::step]],
            rotation=45,
            ha="right",
        )
        ax.set_xlabel("client")
        ax.set_ylabel("samples")
        ax.set_title(title or "Samples per client")
        fig.tight_layout()
        _save_or_show(fig, save_path, show)
        return fig

    def plot_client(
        self,
        client_id: ClientId,
        save_path: Optional[str] = None,
        show: bool = False,
        title: Optional[str] = None,
    ):
        """单个客户端的标签柱状图与占比饼图."""
        plt = _require_plt()
        owner = resolve_owner_id(client_id)
        counts = self.client_counts(owner)
        total = int(counts.sum())
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(len(counts))]
        axes[0].bar(np.arange(len(counts)), counts, color=colors, width=0.8)
        axes[0].set_xticks(np.arange(len(counts)))
        axes[0].set_xticklabels(self.class_names, rotation=45, ha="right")
        axes[0].set_ylabel("samples")
        axes[0].set_title("counts")

        nonzero = counts > 0
        if nonzero.any():
            pie_labels = [
                f"{name} ({int(cnt)})"
                for name, cnt, keep in zip(self.class_names, counts, nonzero)
                if keep
            ]
            axes[1].pie(
                counts[nonzero],
                labels=pie_labels,
                colors=[colors[i] for i, keep in enumerate(nonzero) if keep],
                autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
                startangle=90,
                textprops={"fontsize": 8},
            )
        else:
            axes[1].text(0.5, 0.5, "empty", ha="center", va="center")
            axes[1].set_axis_off()
        axes[1].set_title("share")

        fig.suptitle(title or f"{owner} label distribution (n={total})", y=1.02)
        fig.tight_layout()
        _save_or_show(fig, save_path, show)
        return fig

    def plot_client_heatmap(
        self,
        client_id: ClientId,
        save_path: Optional[str] = None,
        show: bool = False,
        title: Optional[str] = None,
    ):
        """
        单个客户端的单柱热力图.
        按标签升序自下而上堆叠; label 0 在底部为红色, 末类为绿色, 中间渐变.
        纵坐标为该类样本占比.
        """
        from matplotlib.patches import Patch
        from matplotlib.ticker import PercentFormatter

        plt = _require_plt()
        _setup_cjk_font(plt)
        owner = resolve_owner_id(client_id)
        counts = self.client_counts(owner)
        total = int(counts.sum())
        n_classes = len(counts)
        shares = counts.astype(np.float64) / total if total else np.zeros(n_classes)
        colors = _label_red_green_colors(n_classes)

        fig, ax = plt.subplots(figsize=(4.6, 6.4))
        _draw_proportion_stack(ax, shares, colors, width=0.55)
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xticks([0])
        ax.set_xticklabels([_short_client_label(owner)])
        ax.set_ylabel("样本占比")
        ax.set_xlabel("client")
        ax.set_title(title or f"{owner} label heatmap (n={total})")

        legend_handles = [
            Patch(
                facecolor=colors[i],
                edgecolor="white",
                label=f"{int(self.labels[i])} {self.class_names[i]}",
            )
            for i in range(n_classes)
        ]
        ax.legend(
            handles=list(reversed(legend_handles)),
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
            title="label",
        )
        fig.tight_layout()
        _save_or_show(fig, save_path, show)
        return fig

    def plot_poisoned_client_heatmap(
        self,
        client_id: ClientId,
        save_path: Optional[str] = None,
        show: bool = False,
        title: Optional[str] = None,
        poison_ratio: float = 0.5,
        target_label: int = 5,
        seed: int = 42,
    ):
        """
        恶意客户端投毒后的单柱图.
        被投毒样本计入 target_label; 该类中恶意占比从左到右用竖虚线叠加, 无边框.
        """
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.ticker import PercentFormatter

        plt = _require_plt()
        _setup_cjk_font(plt)
        owner = resolve_owner_id(client_id)
        counts, poison_counts = self.poisoned_client_counts(
            owner,
            poison_ratio=poison_ratio,
            target_label=target_label,
            seed=seed,
        )
        total = int(counts.sum())
        n_classes = len(counts)
        shares = counts.astype(np.float64) / total if total else np.zeros(n_classes)
        poison_fracs = np.divide(
            poison_counts.astype(np.float64),
            np.clip(counts.astype(np.float64), 1.0, None),
        )
        poison_fracs = np.where(counts > 0, poison_fracs, 0.0)
        colors = _label_red_green_colors(n_classes)

        fig, ax = plt.subplots(figsize=(4.8, 6.4))
        _draw_proportion_stack(ax, shares, colors, poison_fracs=poison_fracs, width=0.55)
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xticks([0])
        ax.set_xticklabels([_short_client_label(owner)])
        ax.set_ylabel("样本占比")
        ax.set_xlabel("client")
        ax.set_title(
            title
            or (
                f"{owner} BadNets {poison_ratio:.0%} -> {target_label} "
                f"(n={total})"
            )
        )

        legend_handles = [
            Patch(
                facecolor=colors[i],
                edgecolor="white",
                label=f"{int(self.labels[i])} {self.class_names[i]}",
            )
            for i in range(n_classes)
        ]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#404040",
                linestyle=(0, (6.0, 3.2)),
                linewidth=1.2,
                label="poisoned",
            )
        )
        ax.legend(
            handles=list(reversed(legend_handles[:-1])) + [legend_handles[-1]],
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
            title="label",
        )
        fig.tight_layout()
        _save_or_show(fig, save_path, show)
        return fig

    def plot_system(
        self,
        save_dir: str,
        show: bool = False,
        prefix: str = "data_distribution",
    ) -> List[str]:
        """画出系统级分布示意图, 并写出计数表. 返回生成的文件路径."""
        os.makedirs(save_dir, exist_ok=True)
        paths = [
            os.path.join(save_dir, f"{prefix}_heatmap.png"),
            os.path.join(save_dir, f"{prefix}_stacked.png"),
            os.path.join(save_dir, f"{prefix}_sizes.png"),
        ]
        self.plot_heatmap(save_path=paths[0], show=show)
        self.plot_stacked_bar(save_path=paths[1], show=show)
        self.plot_client_sizes(save_path=paths[2], show=show)
        csv_path = os.path.join(save_dir, f"{prefix}_counts.csv")
        self.export_csv(csv_path)
        paths.append(csv_path)
        return paths

    def export_csv(self, path: str) -> None:
        """导出 client x class 计数表, 便于二次分析."""
        _ensure_parent(path)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["client_id", *self.class_names, "total"])
            for cid, row in zip(self.client_ids, self.matrix):
                writer.writerow([cid, *[int(v) for v in row], int(row.sum())])

    def save_all(
        self,
        save_dir: str,
        client_id: Optional[ClientId] = None,
        show: bool = False,
        prefix: str = "data_distribution",
    ) -> List[str]:
        """系统级示意图 + 可选的单个客户端图."""
        paths = self.plot_system(save_dir, show=show, prefix=prefix)
        if client_id is not None:
            owner = resolve_owner_id(client_id)
            client_path = os.path.join(save_dir, f"{prefix}_{owner}.png")
            heatmap_path = os.path.join(save_dir, f"{prefix}_{owner}_heatmap.png")
            self.plot_client(owner, save_path=client_path, show=show)
            self.plot_client_heatmap(owner, save_path=heatmap_path, show=show)
            paths.extend([client_path, heatmap_path])
        return paths


def visualize_data_distribution(
    task_set: TaskSet,
    stores: Dict[str, DatasetStore],
    save_dir: str,
    client_id: Optional[ClientId] = None,
    split: str = SPLIT_TRAIN,
    show: bool = False,
    dataset_name: Optional[str] = None,
) -> DataDistributionVisualizer:
    """便捷入口: 统计分布, 打印摘要, 并保存示意图."""
    vis = DataDistributionVisualizer(
        task_set, stores, split=split, dataset_name=dataset_name
    )
    print(vis.summary())
    paths = vis.save_all(save_dir, client_id=client_id, show=show)
    for path in paths:
        print(f"saved: {path}")
    return vis


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="查看当前联邦学习划分的数据分布, 并画出示意图."
    )
    parser.add_argument("--dataset", default="cifar10", help="数据集名称, 如 cifar10")
    parser.add_argument("--root", default="./data_source", help="数据根目录")
    parser.add_argument(
        "--partitioner",
        default="dirichlet",
        choices=["iid", "dirichlet", "q_non_iid"],
        help="划分策略",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha")
    parser.add_argument("--q", type=float, default=0.5, help="q_non_iid 的 q")
    parser.add_argument("--num-clients", type=int, default=20, help="客户端数量")
    parser.add_argument("--seed", type=int, default=42, help="划分随机种子")
    parser.add_argument(
        "--enable-proxy",
        action="store_true",
        help="与训练管线一致, 从训练集划出 proxy 样本",
    )
    parser.add_argument("--split", default=SPLIT_TRAIN, help="要查看的 split, 默认 train")
    parser.add_argument("--save-dir", default="./outputs/data_distribution", help="图片输出目录")
    parser.add_argument(
        "--client",
        default=None,
        help="额外画出单个客户端, 如 0 或 client_0",
    )
    parser.add_argument("--show", action="store_true", help="弹窗显示图像")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    from core.config import DataConfig, PartitionerConfig

    args = _build_parser().parse_args(argv)
    params: Dict[str, object] = {}
    if args.partitioner == "dirichlet":
        params = {"alpha": args.alpha, "max_retries": 100}
    elif args.partitioner == "q_non_iid":
        params = {"q": args.q}

    data_config = DataConfig(
        dataset=args.dataset,
        root=args.root,
        enable_proxy=args.enable_proxy,
        partitioner=PartitionerConfig(name=args.partitioner, params=params),
    )
    vis = DataDistributionVisualizer.from_data_config(
        data_config,
        num_clients=args.num_clients,
        seed=args.seed,
        split=args.split,
    )
    print(vis.summary())
    paths = vis.save_all(args.save_dir, client_id=args.client, show=args.show)
    for path in paths:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
