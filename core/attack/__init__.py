"""
攻击模块。提供 build_attack 工厂函数及各类攻击实现。
"""
from core.attack.factory import build_attack
# 导入具体的攻击实现以触发注册
from core.attack.data import badnets
from core.attack.data import badnets_dual  # noqa: F401
from core.attack.data import dba  # noqa: F401
from core.attack.upload import batman  # noqa: F401
from core.attack.upload import lp  # noqa: F401
from core.attack.training import chameleon  # noqa: F401
from core.attack.training import lga  # noqa: F401
from core.attack.training import neurotoxin  # noqa: F401
from core.attack.training import pgd  # noqa: F401

__all__ = ["build_attack"]
