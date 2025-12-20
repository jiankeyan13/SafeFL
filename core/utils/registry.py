from typing import Dict, Any, Type, Optional

class Registry:
    """
    通用组件注册器。
    用于动态管理和构建类（模型、聚合器、攻击策略等）。
    """
    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = {}

    def register(self, name: str = None):
        """
        装饰器：将类注册到当前 Registry 中。
        
        用法:
            @REGISTRY.register("my_component")
            class MyComponent: ...
        """
        def _register(cls):
            # 确定注册名：如果没传 name，就用类名
            key = name if name is not None else cls.__name__
            
            if key in self._module_dict:
                # 允许覆盖，但打印警告（可选）
                print(f"Warning: {key} is already registered in {self._name}. Overwriting.")
            
            self._module_dict[key] = cls
            return cls
        return _register

    def get(self, key: str) -> Type:
        """根据名字获取类对象"""
        if key not in self._module_dict:
            raise KeyError(f"'{key}' not found in {self._name} registry. Available: {list(self._module_dict.keys())}")
        return self._module_dict[key]

    def build(self, key: str, **kwargs) -> Any:
        """
        语法糖：获取类并直接实例化。
        Args:
            key: 注册名
            **kwargs: 传给构造函数的参数
        """
        cls = self.get(key)
        return cls(**kwargs)

    def __contains__(self, key: str):
        return key in self._module_dict

# =============================================================================
# 全局注册器实例 (Global Instances)
# =============================================================================

# 1. 模型 (ResNet, VGG, ViT...)
MODEL_REGISTRY = Registry("Models")

# 2. 聚合器 (FedAvg, Median, TrimmedMean...)
AGGREGATOR_REGISTRY = Registry("Aggregators")

# 3. 筛选器/防御 (Krum, MARS, BackdoorIndicator...)
SCREENER_REGISTRY = Registry("Screeners")

# 4. 更新器/优化 (Standard, RLR, Lockdown...)
UPDATER_REGISTRY = Registry("Updaters")

# 5. 攻击策略 (BadNets, Neurotoxin, Scaling...)
ATTACK_REGISTRY = Registry("Attacks")

ALGORITHM_REGISTRY = Registry("Algorithms")

METRIC_REGISTRY = Registry("Metrics")

ATTACK_REGISTRY = Registry("Attacks")