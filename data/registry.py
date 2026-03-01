"""
DatasetRegistry: 类型安全的数据集工厂注册中心
封装全局注册表，消除裸全局 dict, 并在重复注册时快速失败
"""
from typing import Callable, Dict, Any


DatasetFactory = Callable[[str, bool], Any]


class DatasetRegistry:
    """数据集工厂注册中心（单例）。"""

    def __init__(self) -> None:
        self._factories: Dict[str, DatasetFactory] = {}

    def register(self, name: str):
        """装饰器：注册数据集工厂函数。重复注册视为编程错误，立即抛出异常。"""
        def decorator(func: DatasetFactory) -> DatasetFactory:
            if name in self._factories:
                raise ValueError(
                    f"数据集 '{name}' 已被注册，请勿重复注册。"
                )
            self._factories[name] = func
            return func
        return decorator

    def build(self, name: str, root: str, is_train: bool):
        """根据 tag 名称构建并返回 DatasetStore。名称未注册时抛出 KeyError。"""
        if name not in self._factories:
            raise KeyError(
                f"数据集 '{name}' 未注册。"
                f" 已注册的数据集: {list(self._factories.keys())}"
            )
        return self._factories[name](root, is_train)


# 模块单例，供整个项目使用
dataset_registry = DatasetRegistry()