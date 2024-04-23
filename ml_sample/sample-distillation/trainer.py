import torch
from torch.utils.data import DataLoader

from utils import try_gpu


class Trainer:
    def train(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        epochs: int=10,
        learning_rate: float=0.001,
    ) -> float:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        total_loss = 0.0
        for epoch in range(1, epochs+1):
            running_loss = 0.0
            for inputs, labels in loader:
                inputs = try_gpu(inputs)
                labels = try_gpu(labels)

                optimizer.zero_grad()
                
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            print(f"epoch {epoch}/{epochs}, loss: {running_loss / len(loader)}")
            total_loss += (running_loss / len(loader))
        
        return total_loss
