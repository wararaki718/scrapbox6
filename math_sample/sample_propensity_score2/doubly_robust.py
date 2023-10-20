import numpy as np
import pandas as pd
import statsmodels.api as sm


def doubly_robust_estimator(df: pd.DataFrame) -> float:
    data = df[["motivation", "science", "special_course"]]
    X = sm.add_constant(data)

    y = df.score

    model = sm.OLS(y, X)
    results = model.fit()

    X_0 = X.copy()
    X_0["special_course"] = 0
    y_0 = results.predict(X_0)

    X_1 = X.copy()
    X_1["special_course"] = 1
    y_1 = results.predict(X_1)

    N = len(df)
    D = df.special_course
    Y = df.score
    e_X = df.propensity_score

    y1_dr = (1 / N) * np.sum((D / e_X) * Y + (1 - (D / e_X)) * y_1)
    y0_dr = (1 / N) * np.sum(((1 - D) / (1 - e_X)) * Y + (1 - ((1 - D) / (1 - e_X))) * y_0)

    return y1_dr - y0_dr
