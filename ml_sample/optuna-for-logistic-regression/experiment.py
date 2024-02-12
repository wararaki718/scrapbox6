import numpy as np
from optuna.trial import Trial
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

class AccuracyExperiment:
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


class LogLossExperiment:
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
            max_iter=500,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        y_preds = model.predict_proba(X_valid)[:, 1]
        loss = log_loss(y_valid, y_preds)

        return loss


class ROCAUCExperiment:
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
        y_preds = model.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, y_preds)

        return score
