import torch
import optuna
from optuna.trial import Trial
from torch.utils.data import DataLoader

from model import NNModel


def _train(model: torch.nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module) -> float:
    model.train()
    total_loss = 0.0
    for data, target in data_loader:
        optimizer.zero_grad()
        y_preds = model(data.view(data.size(0), -1))
        loss = criterion(y_preds, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def _validate(model: torch.nn.Module, data_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            y_preds: torch.Tensor = model(data.view(data.size(0), -1))
            y_label = y_preds.argmax(dim=1, keepdim=True)
            correct += y_label.eq(target.view_as(y_label)).sum().item()
    accuracy = correct / len(data_loader)
    return accuracy


def experiment(trial: Trial, train_loader: DataLoader, valid_loader: DataLoader, epochs: int=10) -> float:
    model = NNModel(trial)

    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = torch.nn.NLLLoss()

    for epoch in range(1, epochs+1):
        loss = _train(model, train_loader, optimizer, criterion)
        accuracy = _validate(model, valid_loader)
        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return accuracy
