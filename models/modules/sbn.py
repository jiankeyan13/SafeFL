import torch.nn as nn


def _iter_bn_modules(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            yield m


def set_sbn_train(model):
    """
    将模型内所有 BN 设为训练期的 sBN：仅用 batch 统计，不更新 running stats。
    """
    for bn in _iter_bn_modules(model):
        bn.track_running_stats = False


def set_sbn_eval(model):
    """将模型内所有 BN 设为 eval：允许使用/更新 running stats。"""
    for bn in _iter_bn_modules(model):
        bn.track_running_stats = True


def reset_bn_running_stats(model):
    """重置 BN 的 running_mean/var 与 num_batches_tracked。"""
    for bn in _iter_bn_modules(model):
        if hasattr(bn, "running_mean") and bn.running_mean is not None:
            bn.running_mean.zero_()
        if hasattr(bn, "running_var") and bn.running_var is not None:
            bn.running_var.fill_(1.0)
        if hasattr(bn, "num_batches_tracked") and bn.num_batches_tracked is not None:
            bn.num_batches_tracked.zero_()

