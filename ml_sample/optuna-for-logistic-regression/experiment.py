import numpy as np
from optuna.trial import Trial
from sklearn.linear_model import LogisticRegression

class Experiment:
    def __init__(self) -> None:
        pass

    def experiment(
        self,
        trial: Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
    ) -> float:
        model = LogisticRegression(
            tol=trial.suggest_float("tol", 0.00001, 1, log=True),
            C=trial.suggest_float("C", 0.0001, 1.0, log=True),
            max_iter=1000,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        score = model.score(X_valid, y_valid)

        return score
