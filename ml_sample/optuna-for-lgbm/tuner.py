from typing import Tuple

import numpy as np
import optuna.integration.lightgbm as lgbm
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import accuracy_score


class LGBMTunerExperiment:
    def experiment(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[float, dict]:
        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
        }
        train_set = lgbm.Dataset(X_train, label=y_train)
        valid_set = lgbm.Dataset(X_valid, label=y_valid)

        model = lgbm.train(
            param,
            train_set=train_set,
            valid_sets=[valid_set],
            callbacks=[early_stopping(100), log_evaluation(100)],
        )
        y_preds = model.predict(X_test, num_iteration=model.best_iteration)
        y_labels = np.rint(y_preds)
        score = accuracy_score(y_test, y_labels)
        return score, model.params
