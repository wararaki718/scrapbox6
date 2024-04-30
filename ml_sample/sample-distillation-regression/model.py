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


class ModifiedDeepNNRegressor(nn.Module):
    def __init__(self, n_classes: int=10) -> None:
        super(ModifiedDeepNNRegressor, self).__init__()
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
        conv_features = x
        x = torch.flatten(x, 1)
        x = self._classifier(x)
        return x, conv_features


class ModifiedLightNNRegressor(nn.Module):
    def __init__(self, n_classes: int=10) -> None:
        super(ModifiedLightNNRegressor, self).__init__()
        self._feature_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self._regressor = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
        )
        self._classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        x = self._feature_layers(x)
        output = self._regressor(x)
        x = torch.flatten(x, 1)
        x = self._classifier(x)
        return x, output
