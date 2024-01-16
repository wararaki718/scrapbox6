import os
from functools import partial
from pathlib import Path

import optuna

from experiment import experiment
from loader import DataLoaderFactory


def main() -> None:
    mnist_path = Path(f"{os.getcwd()}/dataset")
    train_loader, valid_loader = DataLoaderFactory.create(mnist_path)

    print("start study:")
    objective = partial(experiment, train_loader=train_loader, valid_loader=valid_loader)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("## study stats")
    print(f"number of finished trials: {len(study.trials)}")
    print(f"number of pruned trials: {len(pruned_trials)}")
    print(f"number of complete trials: {len(complete_trials)}")
    print()

    print("## best trials")
    best_trial = study.best_trial
    print(f"value: {best_trial.value}")
    
    print("params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
