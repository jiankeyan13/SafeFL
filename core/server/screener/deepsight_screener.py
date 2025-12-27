import math
from typing import List, Dict, Any, Tuple

import hdbscan
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from core.utils.registry import SCREENER_REGISTRY
from .base_screener import BaseScreener


@SCREENER_REGISTRY.register("deepsight")
class DeepSightScreener(BaseScreener):
    """
    DeepSight 筛选阶段实现：
    - NEUP/TE 计算
    - 基于输出层 Bias 的余弦距离
    - 加入 softmax 的 DDif（概率比值）并预计算全局输出
    - HDBSCAN 集成聚类 + PCI 决策（离群点回退 TE 标签）
    """

    def __init__(self, num_seeds: int = 3, num_samples: int = 20000,
                 batch_size: int = 2000, tau: float = 1 / 3,
                 epsilon: float = 1e-7, **kwargs):
        super().__init__()
        self.num_seeds = num_seeds
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.tau = tau
        self.epsilon = epsilon
        self._noise_cache = {}

    def screen(self, client_deltas: List[Dict[str, torch.Tensor]],
               num_samples: List[float],
               global_model: torch.nn.Module = None,
               context: Dict[str, Any] = None) -> Tuple[List[float], Dict[str, Any]]:
        context = context or {}
        num_clients = len(client_deltas)
        if num_clients == 0:
            return [], context

        if global_model is None:
            # 缺少模型无法执行 DDif，默认全保留
            return [1.0] * num_clients, context

        # 安全获取模型所在设备，避免对 Tensor 做布尔判断
        params_iter = iter(global_model.parameters())
        first_param = next(params_iter, None)
        device = first_param.device if first_param is not None else torch.device("cpu")

        # 定位输出层（兼容 fc / linear 命名）
        last_weight_name, last_bias_name = self._resolve_output_layer(global_model)
        if last_weight_name is None:
            return [1.0] * num_clients, context

        num_classes = self._infer_num_classes(global_model, last_weight_name, last_bias_name)
        # 输入形状：优先 context/模型属性，否则按类别数硬编码 CIFAR vs Tiny-ImageNet
        input_shape = self._infer_input_shape(global_model, context, num_classes)

        neups, tes = self._calculate_neups_tes(client_deltas, last_weight_name, last_bias_name, num_classes)
        median_te = float(np.median(tes)) if len(tes) > 0 else 0.0
        # False: 良性，True: 可疑
        suspicious_flags = [False if te > median_te * 0.5 else True for te in tes]

        cosine_distances = self._calculate_cosine_distance(client_deltas, last_bias_name, num_classes)
        ddifs = self._calculate_ddifs(client_deltas, global_model, input_shape, num_classes, device)

        # 集成聚类：DDif + NEUP + 余弦距离
        ddif_cluster_dists = self._build_ddif_distance(ddifs, num_clients)
        neup_clusters = hdbscan.HDBSCAN(allow_single_cluster=True).fit_predict(neups.detach().cpu().numpy())
        neup_cluster_dists = self._dists_from_clust(neup_clusters, num_clients)

        cosine_clusters = hdbscan.HDBSCAN(metric="precomputed", allow_single_cluster=True).fit_predict(cosine_distances.detach().cpu().numpy())
        cosine_cluster_dists = self._dists_from_clust(cosine_clusters, num_clients)

        merged_distances = np.mean([ddif_cluster_dists, neup_cluster_dists, cosine_cluster_dists], axis=0)
        final_clusters = hdbscan.HDBSCAN(metric="precomputed", allow_single_cluster=True).fit_predict(merged_distances)

        # PCI 决策：簇内可疑比例 + 离群点回退 TE 标签
        benign_indices, malicious_indices = self._decide_clients(final_clusters, suspicious_flags)

        # 计算全量参数的 L2 范数中位数作为裁剪值 (参考 hdbscan.py)
        all_norms = [torch.norm(torch.cat([v.flatten() for v in delta.values()])).item() \
                     for delta in client_deltas]
        clip_value = float(np.median(all_norms)) if all_norms else 1.0

        screen_scores = [1.0 if i in benign_indices else 0.0 for i in range(num_clients)]
        context.update(
            {
                "clip_value": clip_value,
                "norm_list": all_norms,
            }
        )
        return screen_scores, context

    # -------------------- 核心子过程 -------------------- #
    def _calculate_neups_tes(self, client_deltas: List[Dict[str, torch.Tensor]],
                             weight_name: str, bias_name: str,
                             num_classes: int) -> Tuple[np.ndarray, List[float]]:
        weight_stack = torch.stack([delta[weight_name] for delta in client_deltas])
        bias_stack = torch.stack([delta[bias_name] for delta in client_deltas])

        reduce_dims = tuple(range(2, weight_stack.dim())) if weight_stack.dim() > 2 else ()
        diff_weight = torch.sum(torch.abs(weight_stack), dim=reduce_dims) if reduce_dims else torch.abs(weight_stack)
        diff_bias = torch.abs(bias_stack)

        energy = diff_weight + diff_bias
        energy_sq = energy ** 2
        total_energy = energy_sq.view(energy_sq.shape[0], -1).sum(dim=1, keepdim=True)

        neup = energy_sq / torch.where(total_energy <= 0, torch.ones_like(total_energy), total_energy)
        if (total_energy <= 0).any():
            uniform = torch.full_like(neup, 1.0 / max(energy.shape[1], 1))
            neup = torch.where(total_energy <= 0, uniform, neup)

        max_neup = neup.max(dim=1).values if neup.numel() > 0 else torch.zeros(neup.shape[0])
        threshold = max(0.01, 1.0 / max(num_classes, 1)) * max_neup
        tes_tensor = torch.sum(neup >= threshold.unsqueeze(1), dim=1)

        return neup, tes_tensor.to(torch.int).cpu().tolist()

    def _calculate_cosine_distance(self, client_deltas: List[Dict[str, torch.Tensor]],
                                   bias_name: str, num_classes: int) -> np.ndarray:
        bias_vectors = []
        for delta in client_deltas:
            bias = delta[bias_name]
            bias_vectors.append(bias.view(-1))

        bias_stack = torch.stack(bias_vectors)  # [n, num_classes]
        norms = torch.norm(bias_stack, dim=1, keepdim=True)
        normalized = bias_stack / norms.clamp(min=1e-9)
        cos_sim = torch.mm(normalized, normalized.t()).clamp(-1.0, 1.0)
        cos_dist = (1 - cos_sim).clamp(min=0).to(torch.float64)
        return cos_dist

    def _calculate_ddifs(self, client_deltas: List[Dict[str, torch.Tensor]],
                         global_model: torch.nn.Module,
                         input_shape: Tuple[int, ...],
                         num_classes: int,
                         device: torch.device) -> np.ndarray:
        if global_model is None:
            return None

        global_state = global_model.state_dict()

        param_names = [name for name, _ in global_model.named_parameters()]
        buffer_names = [name for name, _ in global_model.named_buffers()]

        stacked_params = {}
        stacked_buffers = {}

        for name in param_names:
            global_tensor = global_state[name]
            zeros = torch.zeros_like(global_tensor)
            delta_stack = torch.stack(
                [delta.get(name, zeros).to(global_tensor.device) for delta in client_deltas],
                dim=0,
            )
            stacked_params[name] = global_tensor + delta_stack

        for name in buffer_names:
            global_buffer = global_state[name]
            zeros = torch.zeros_like(global_buffer)
            delta_stack = torch.stack(
                [delta.get(name, zeros).to(global_buffer.device) for delta in client_deltas],
                dim=0,
            )
            stacked_buffers[name] = global_buffer + delta_stack

        ddifs = []
        global_model.eval()
        for seed in range(self.num_seeds):
            noise_batches = self._get_noise_batches(device, input_shape, self.num_samples, self.batch_size, seed)

            if len(noise_batches) == 0:
                return None

            seed_ddif = torch.zeros(len(client_deltas), num_classes, device=device)

            for inputs in noise_batches:
                with torch.inference_mode():
                    prob_output_global = self._to_prob(global_model(inputs))

                # 顺序遍历客户端，避免 vmap 带来的高峰值显存
                with torch.inference_mode():
                    for idx in range(len(client_deltas)):
                        params_i = {k: v[idx] for k, v in stacked_params.items()}
                        buffers_i = {k: v[idx] for k, v in stacked_buffers.items()}
                        logits_local = functional_call(global_model, (params_i, buffers_i), (inputs,))
                        prob_output_local = self._to_prob(logits_local)
                        ratio = torch.div(prob_output_local, prob_output_global + self.epsilon)
                        seed_ddif[idx].add_(ratio.sum(dim=0))

            seed_ddif /= self.num_samples
            ddifs.append(seed_ddif)

        return torch.stack(ddifs, dim=0).cpu().numpy()

    def _build_ddif_distance(self, ddifs: np.ndarray, num_clients: int) -> np.ndarray:
        if ddifs is None or len(ddifs) == 0:
            return np.ones((num_clients, num_clients))

        ddif_cluster_dists = []
        for i in range(len(ddifs)):
            ddif_clusters = hdbscan.HDBSCAN(allow_single_cluster=True).fit_predict(ddifs[i])
            ddif_cluster_dists.append(self._dists_from_clust(ddif_clusters, num_clients))
        return np.mean(ddif_cluster_dists, axis=0)

    def _decide_clients(self, final_clusters: np.ndarray,
                        suspicious_flags: List[bool]) -> Tuple[set, set]:
        benign_counts = {}
        total_counts = {}
        for idx, cluster in enumerate(final_clusters):
            if cluster == -1:
                continue
            total_counts[cluster] = total_counts.get(cluster, 0) + 1
            if not suspicious_flags[idx]:
                benign_counts[cluster] = benign_counts.get(cluster, 0) + 1

        benign_indices = set()
        malicious_indices = set()
        for idx, cluster in enumerate(final_clusters):
            if cluster == -1:
                # 离群点回退 TE 标签
                if not suspicious_flags[idx]:
                    benign_indices.add(idx)
                else:
                    malicious_indices.add(idx)
                continue

            total = total_counts.get(cluster, 1)
            benign = benign_counts.get(cluster, 0)
            suspicious_ratio = (total - benign) / max(total, 1)
            if suspicious_ratio < self.tau:
                benign_indices.add(idx)
            else:
                malicious_indices.add(idx)

        return benign_indices, malicious_indices

    # -------------------- 工具函数 -------------------- #
    def _infer_num_classes(self, global_model: torch.nn.Module,
                           weight_name: str, bias_name: str) -> int:
        state = global_model.state_dict()
        if bias_name in state:
            return state[bias_name].shape[0]
        if weight_name in state:
            return state[weight_name].shape[0]
        # 兜底：找到最后一个 weight 形状
        for key in reversed(list(state.keys())):
            if key.endswith("weight"):
                return state[key].shape[0]
        return 1

    def _infer_input_shape(self, global_model: torch.nn.Module,
                           context: Dict[str, Any], num_classes: int) -> Tuple[int, ...]:
        if context and "input_shape" in context:
            return tuple(context["input_shape"])
        if hasattr(global_model, "input_shape"):
            try:
                return tuple(global_model.input_shape)
            except Exception:
                pass
        # 无 context 时按数据集类别数硬编码
        if num_classes >= 200:
            return (3, 64, 64)  # Tiny-ImageNet
        return (3, 32, 32)      # CIFAR-10/100

    def _dists_from_clust(self, clusters: np.ndarray, N: int) -> np.ndarray:
        pairwise_dists = np.ones((N, N))
        is_same = (clusters[:, None] == clusters[None, :]) & (clusters[:, None] != -1)
        pairwise_dists[is_same] = 0
        return pairwise_dists

    def _to_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(outputs, dim=-1)

    def _resolve_output_layer(self, global_model: torch.nn.Module) -> Tuple[str, str]:
        state_keys = list(global_model.state_dict().keys())
        if "fc.weight" in state_keys and "fc.bias" in state_keys:
            return "fc.weight", "fc.bias"
        if "linear.weight" in state_keys and "linear.bias" in state_keys:
            return "linear.weight", "linear.bias"
        # 兜底：取最后出现的 weight/bias 配对
        for key in reversed(state_keys):
            if key.endswith("weight"):
                prefix = key[: -len("weight")]
                bias_key = prefix + "bias"
                if bias_key in state_keys:
                    return key, bias_key
        return None, None

    def _get_noise_batches(self, device: torch.device, input_shape: Tuple[int, ...],
                           num_samples: int, batch_size: int, seed: int):
        key = (str(device), input_shape, num_samples, batch_size, seed)
        if key in self._noise_cache:
            return self._noise_cache[key]

        remaining = num_samples
        num_batches = 20  # 拆成 20 份，减小单批显存占用
        batch_cap = max(1, math.ceil(num_samples / num_batches))
        noise_batches = []

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        while remaining > 0 and len(noise_batches) < num_batches:
            current = min(batch_cap, remaining)
            noise_batches.append(torch.rand((current, *input_shape), device=device, generator=generator))
            remaining -= current

        self._noise_cache[key] = noise_batches
        return noise_batches

