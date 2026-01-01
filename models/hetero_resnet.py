import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.registry import MODEL_REGISTRY
from models.modules import Scaler


def _make_width(val: int, p: float) -> int:
    return max(1, int(val * p))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, scale=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.scaler1 = Scaler(scale)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scaler2 = Scaler(scale)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                Scaler(scale),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.scaler1(out)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.scaler2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, scale=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.scaler1 = Scaler(scale)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.scaler2 = Scaler(scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.scaler3 = Scaler(scale)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                Scaler(scale),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.scaler1(out)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.scaler2(out)
        out = F.relu(self.bn2(out))
        out = self.conv3(out)
        out = self.scaler3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, input_channels=3, p: float = 1.0):
        super().__init__()
        if p <= 0:
            raise ValueError(f"p must be positive, got {p}")
        self.width_ratio = float(p)
        scale = 1.0 / self.width_ratio

        self.in_planes = _make_width(64, self.width_ratio)

        self.conv1 = nn.Conv2d(
            input_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scaler1 = Scaler(scale)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, scale=scale)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, scale=scale)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, scale=scale)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, scale=scale)
        self.linear = nn.Linear(_make_width(512, self.width_ratio) * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, scale):
        planes = _make_width(planes, self.width_ratio)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, scale))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.scaler1(out)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


@MODEL_REGISTRY.register("hetero_resnet18")
def HeteroResNet18(num_classes=10, input_channels=3, p: float = 1.0):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channels=input_channels, p=p)

