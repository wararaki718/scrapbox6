import time

import numpy as np
import pandas as pd


def main() -> None:
    nrow = 100000
    ncol = 1000
    time.sleep(1)

    # create a dataframe
    df = pd.DataFrame(
        np.random.uniform(low=0, high=1000, size=(nrow, ncol)),
        columns=[f"col_{i}" for i in range(ncol)],
    )
    time.sleep(1)
    
    view = df[df.col_0 > 500]
    time.sleep(1)

    data = df.col_1.apply(lambda x: x + 1)
    time.sleep(1)

    del view
    time.sleep(1)

    del data
    time.sleep(1)

    values = df.values
    time.sleep(1)

    t = 0
    time.sleep(1)

    for i, _ in enumerate(values):
        t += i
    time.sleep(1)

    values[0][0] = 1000
    time.sleep(1)

    del values
    time.sleep(1)

    items: list = df.to_dict(orient="records")
    time.sleep(1)

    for i, _ in enumerate(items):
        t += i
    time.sleep(1)
    
    del items
    time.sleep(1)

    del df
    time.sleep(1)

    print("DONE")

if __name__ == "__main__":
    main()
