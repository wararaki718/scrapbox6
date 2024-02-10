import numpy as np
import lightgbm as lgbm
from optuna.trial import Trial
from sklearn.metrics import accuracy_score


class LGBMExperiment:
    def experiment(
        self,
        trial: Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        train_set = lgbm.Dataset(X_train, label=y_train)
        valid_set = lgbm.Dataset(X_valid, label=y_valid)

        model = lgbm.train(param, train_set=train_set, valid_sets=[valid_set])
        y_preds = model.predict(X_test)
        y_labels = np.rint(y_preds)
        score = accuracy_score(y_test, y_labels)
        return score
