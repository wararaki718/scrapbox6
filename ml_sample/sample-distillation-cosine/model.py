import torch
import torch.nn as nn


class NNModel(nn.Module):
    def __init__(self, n_classes: int=10) -> None:
        super(NNModel, self).__init__()
        self._feature_layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self._classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._feature_layers(x)
        x = torch.flatten(x, 1)
        x = self._classifier(x)
        return x


class DistilModel(nn.Module):
    def __init__(self, n_classes: int=10) -> None:
        super(DistilModel, self).__init__()
        self._feature_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self._classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._feature_layers(x)
        x = torch.flatten(x, 1)
        x = self._classifier(x)
        return x


class ModifiedDeepNNCosine(nn.Module):
    def __init__(self, n_classes: int=10) -> None:
        super(ModifiedDeepNNCosine, self).__init__()
        self._feature_layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self._classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        x = self._feature_layers(x)
        flattened_conv_output = torch.flatten(x, 1)
        x = self._classifier(flattened_conv_output)
        flattened_conv_output_after_pooling = torch.nn.functional.avg_pool1d(flattened_conv_output, 2)
        return x, flattened_conv_output_after_pooling


class ModifiedLightNNCosine(nn.Module):
    def __init__(self, n_classes: int=10) -> None:
        super(ModifiedLightNNCosine, self).__init__()
        self._feature_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self._classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        x = self._feature_layers(x)
        flattened_conv_output = torch.flatten(x, 1)
        x = self._classifier(flattened_conv_output)
        return x, flattened_conv_output
