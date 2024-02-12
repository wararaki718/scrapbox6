import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


class Evaluator:
    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        params: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        model = LogisticRegression(**params, max_iter=500, n_jobs=-1)
        model.fit(X_train, y_train)
        y_preds = model.predict_proba(X_test)[:, 1]

        result = {
            "accuracy": model.score(X_test, y_test),
            "log_loss": log_loss(y_test, y_preds),
            "roc_auc": roc_auc_score(y_test, y_preds),
        }
        return result
