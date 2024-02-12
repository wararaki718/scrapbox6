import gc
from functools import partial

import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from experiment import AccuracyExperiment, LogLossExperiment, ROCAUCExperiment
from evaluate import Evaluator

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

    print("accuracy:")
    experiment = AccuracyExperiment()
    objective = partial(
        experiment.experiment,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    accuracy_params = study.best_trial.params
    print(f"number of finished trials: {len(study.trials)}")
    print()

    print("log loss:")
    experiment = LogLossExperiment()
    objective = partial(
        experiment.experiment,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
    )
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    log_loss_params = study.best_trial.params
    print(f"number of finished trials: {len(study.trials)}")
    print()

    print("roc auc:")
    experiment = ROCAUCExperiment()
    objective = partial(
        experiment.experiment,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    roc_auc_params = study.best_trial.params
    print(f"number of finished trials: {len(study.trials)}")
    print()

    evaluator = Evaluator()
    result = evaluator.evaluate(accuracy_params, X_train, y_train, X_test, y_test)
    print("[accuracy]")
    print(f"param: {accuracy_params}")
    print(f"result: {result}")
    print()

    result = evaluator.evaluate(log_loss_params, X_train, y_train, X_test, y_test)
    print("[log_loss]")
    print(f"param: {log_loss_params}")
    print(f"result: {result}")
    print()

    result = evaluator.evaluate(roc_auc_params, X_train, y_train, X_test, y_test)
    print("[roc_auc]")
    print(f"param: {roc_auc_params}")
    print(f"result: {result}")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
