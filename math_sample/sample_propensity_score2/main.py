import numpy as np
import pandas as pd
import statsmodels.api as sm

from match import ps_stratificasion
from ipw import ipw
from doubly_robust import doubly_robust_estimator


def main() -> None:
    np.random.seed(0)
    size = 120

    X1 = np.random.uniform(0, 1, size)
    X2 = np.random.choice([0, 1], p=[0.5, 0.5], size=size)

    D_noise = np.random.uniform(0, 0.5, size)
    D_threshold = 0.25 * X1 + 0.25 * X2 + D_noise
    D = np.where(D_threshold >= 0.5, 1, 0)

    y_noise = np.random.normal(0, 10, size)
    y = np.clip(40 * X1 + 20 * X2 + 20 * D + y_noise, 0, 100).astype(int)

    df = pd.DataFrame({
        "motivation": X1,
        "science": X2,
        "special_course": D,
        "score": y
    })
    print(df.head(3))
    print()

    y0 = df[df.special_course == 0].score
    y1 = df[df.special_course == 1].score
    print(y1.mean() - y0.mean())
    print()

    X = df[["motivation", "science"]]
    X = sm.add_constant(X)

    D = df.special_course
    model = sm.Logit(D, X)
    result = model.fit()
    predicts = result.predict(X)
    df["propensity_score"] = predicts
    print(df.head(3))
    print()

    print("ps match:")
    print(ps_stratificasion(df))
    print(ps_stratificasion(df, 0.05))
    print(ps_stratificasion(df, 0.025))
    print(ps_stratificasion(df, 0.01))
    print()

    print("ipw:")
    print(ipw(df))
    print()

    print("doubly robust:")
    print(doubly_robust_estimator(df))
    print()

    print("DONE")


if __name__ == "__main__":
    main()
