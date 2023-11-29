from time import sleep

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def main() -> None:
    n_items = 10000
    n_dim = 100
    df = pd.DataFrame(np.random.random((n_items, n_dim)))
    print(df.shape)
    sleep(1)

    x = df.values
    y = (np.random.uniform(0, 1, size=n_items) >= 0.5).astype(int)
    print(x.shape)
    print(y.shape)
    sleep(1)

    model = LogisticRegression()
    model.fit(x, y)
    print("fit")
    sleep(1)

    score = model.score(x, y)
    print(score)
    sleep(1)
    
    print("DONE")


if __name__ == "__main__":
    main()
