import torch
import numpy as np
import hdbscan
from typing import List, Dict, Any, Tuple

from core.utils.registry import SCREENER_REGISTRY
from .base_screener import BaseScreener


@SCREENER_REGISTRY.register("hdbscan")
class HdbscanScreener(BaseScreener):
    """
    FLAME 筛选器：使用 HDBSCAN 聚类检测恶意客户端。
    基于余弦距离矩阵进行聚类，选择最大簇作为良性客户端。
    min_cluster_size 固定为 num_clients // 2 + 1
    """
    
    def __init__(self, **kwargs):
        super().__init__()

    def screen(self, 
               client_deltas: List[Dict[str, torch.Tensor]], 
               num_samples: List[float],
               global_model: torch.nn.Module = None,
               context: Dict[str, Any] = None) -> Tuple[List[float], Dict[str, Any]]:
        """
        使用 HDBSCAN 聚类筛选恶意客户端。
        
        Returns:
            (screen_scores, context): 
                - screen_scores: 0/1 分数
                - context: 包含 clip_value 范数裁剪值和 norm_list各客户端范数
        """
        context = context or {}
        n = len(client_deltas)
        
        if n < 2:
            return [1.0] * n, context

        # 1. 获取可学习参数名称
        if global_model is not None:
            learnable_params = set(name for name, p in global_model.named_parameters() if p.requires_grad)
        else:
            learnable_params = set(client_deltas[0].keys())

        # 2. 展平客户端参数向量
        vectors = []
        for delta_dict in client_deltas:
            flat = torch.cat([t.view(-1) for name, t in delta_dict.items() if name in learnable_params])
            vectors.append(flat)
        
        vec_stack = torch.stack(vectors)  # [n, d]
        
        # 3. 计算余弦距离矩阵（向量化）
        # cos_sim[i,j] = (v_i · v_j) / (||v_i|| * ||v_j||)
        norms = torch.norm(vec_stack, dim=1, keepdim=True)
        normalized = vec_stack / norms.clamp(min=1e-9)
        cos_sim = torch.mm(normalized, normalized.t()).clamp(-1.0, 1.0)  # [n, n]
        # hdbscan 要求 precomputed 距离矩阵为 double (float64) 类型
        cos_dist = (1 - cos_sim).clamp(min=0).cpu().numpy().astype(np.float64)  # 余弦距离
        
        # 4. HDBSCAN 聚类
        min_cluster_size = n // 2 + 1
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            allow_single_cluster=True,
            metric='precomputed'
        ).fit(cos_dist)
        
        labels = clusterer.labels_
        
        # 5. 选择最大簇
        if labels.max() < 0:
            # 所有样本都是噪声，保留全部
            benign_indices = set(range(n))
        else:
            # 找到最大簇
            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
            if len(unique_labels) > 0:
                max_cluster = unique_labels[np.argmax(counts)]
                benign_indices = set(np.where(labels == max_cluster)[0].tolist())
            else:
                benign_indices = set(range(n))
        
        # 6. 计算全局范数中位数作为裁剪值
        norm_list = [torch.norm(vec).item() for vec in vectors]
        clip_value = float(np.median(norm_list)) if norm_list else 1.0
        
        # 更新 context
        context['clip_value'] = clip_value
        context['norm_list'] = norm_list
        context['benign_indices'] = benign_indices
        
        # 7. 返回筛选分数
        screen_scores = [1.0 if i in benign_indices else 0.0 for i in range(n)]
        
        return screen_scores, context

