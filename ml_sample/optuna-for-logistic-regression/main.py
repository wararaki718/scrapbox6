import gc
from functools import partial

import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from experiment import Experiment


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5)
    print(f"train: {X_train.shape}, {y_train.shape}")
    print(f"valid: {X_valid.shape}, {y_valid.shape}")
    print(f"test : {X_test.shape}, {y_test.shape}")
    print()
    del X, y
    gc.collect()

    experiment = Experiment()
    objective = partial(
        experiment.experiment,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print(f"number of finished trials: {len(study.trials)}")
    print(f"best: {study.best_trial.params}")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
