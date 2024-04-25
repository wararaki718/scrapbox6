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


class DistillationTrainer:
    def train(
        self,
        teacher_model: torch.nn.Module,
        student_model: torch.nn.Module,
        loader: DataLoader,
        epochs: int=10,
        learning_rate: float=0.001,
        hidden_representation_loss_weight: float=0.25,
        ce_loss_weight: float=0.75,
    ) -> float:
        criterion = torch.nn.CrossEntropyLoss()
        cosine_criterion = torch.nn.CosineEmbeddingLoss()
        optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)

        teacher_model.eval()
        student_model.train()

        total_loss = 0.0
        for epoch in range(1, epochs+1):
            running_loss = 0.0
            for inputs, labels in loader:
                inputs = try_gpu(inputs)
                labels = try_gpu(labels)

                optimizer.zero_grad()
                with torch.no_grad():
                    _, teacher_hidden_representation = teacher_model(inputs)

                student_logits, student_hidden_representation = student_model(inputs)

                hidden_loss = cosine_criterion(
                    student_hidden_representation,
                    teacher_hidden_representation,
                    target=try_gpu(torch.ones(inputs.size(0)))
                )

                label_loss = criterion(student_logits, labels)
                loss = hidden_representation_loss_weight * hidden_loss + ce_loss_weight * label_loss
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(f"epoch {epoch}/{epochs}, loss: {running_loss / len(loader)}")
            total_loss += running_loss
        
        return total_loss
