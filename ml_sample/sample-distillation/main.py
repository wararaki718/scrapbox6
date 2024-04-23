import torch
from torch.utils.data import DataLoader

from evaluator import Evaluator
from loader import DataLoaderFactory
from model import DistilModel, NNModel
from trainer import Trainer
from utils import try_gpu


def main() -> None:
    torch.manual_seed(42)

    train_loader: DataLoader = DataLoaderFactory.create(True, True)
    test_loader: DataLoader = DataLoaderFactory.create(False, False)
    print("data loaded.")

    nn_model = try_gpu(NNModel(n_classes=10))
    trainer = Trainer()
    _ = trainer.train(nn_model, train_loader)
    print("model trained.")

    evaluator = Evaluator()
    score = evaluator.evaluate(nn_model, test_loader)
    print(f"accuracy score: {score}")
    print()

    distil_model = try_gpu(DistilModel(n_classes=10))
    _ = trainer.train(distil_model, train_loader)
    print("model trained.")

    score = evaluator.evaluate(distil_model, test_loader)
    print(f"accraucy score: {score}")

    print("DONE")


if __name__ == "__main__":
    main()
