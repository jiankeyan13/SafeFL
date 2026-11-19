from __future__ import annotations

import torch
import torch.nn as nn

from core.utils.registry import MODEL_REGISTRY


class FashionMNISTCNN(nn.Module):
    """CNN for FashionMNIST: 2 convolutional layers and 2 fully connected layers."""

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        fc_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, conv1_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv2_channels * 7 * 7, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


MODEL_REGISTRY.register("fashion_cnn")(FashionMNISTCNN)
