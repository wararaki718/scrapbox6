import torch
from torch.utils.data import DataLoader

from evaluator import Evaluator, DistilEvaluator
from loader import DataLoaderFactory
from model import ModifiedDeepNNRegressor, ModifiedLightNNRegressor, NNModel
from trainer import Trainer, DistillationTrainer
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
    print(f"accuracy score: {score} (large)")
    print()

    modified_nn_model = try_gpu(ModifiedDeepNNRegressor(n_classes=10))
    modified_nn_model.load_state_dict(nn_model.state_dict())

    torch.manual_seed(42)
    distil_model = try_gpu(ModifiedLightNNRegressor(n_classes=10))
    distillation_trainer = DistillationTrainer()
    _ = distillation_trainer.train(modified_nn_model, distil_model, train_loader)
    print("distil model trained.")

    distil_evaluator = DistilEvaluator()
    score = distil_evaluator.evaluate(distil_model, test_loader)
    print(f"accraucy score: {score} (distil)")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
