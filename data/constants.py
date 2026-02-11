"""
数据管道协议常量：集中定义 split/tag 相关字符串，避免散落与拼写错误。
"""

# Split 类型
SPLIT_TRAIN = "train"
SPLIT_TEST = "test"
SPLIT_TEST_GLOBAL = "test_global"
SPLIT_PROXY = "proxy"
SPLIT_TEMP_ALL = "temp_all"

# Owner 标识
OWNER_SERVER = "server"


def client_owner(client_id: int) -> str:
    """生成客户端 owner 字符串。"""
    return f"client_{client_id}"


def train_plain_tag(dataset_name: str) -> str:
    """训练集（无增强）store tag。"""
    return f"{dataset_name}_train_plain"


def train_aug_tag(dataset_name: str) -> str:
    """训练集（带增强）store tag。"""
    return f"{dataset_name}_train_aug"


def test_plain_tag(dataset_name: str) -> str:
    """全局测试集 store tag。"""
    return f"{dataset_name}_test_plain"
