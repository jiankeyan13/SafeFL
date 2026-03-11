"""
攻击模块。提供 build_attack 工厂函数及各类攻击实现。
"""
from core.attack.factory import build_attack
# 导入具体的攻击实现以触发注册
from core.attack.data import badnets

__all__ = ["build_attack"]
