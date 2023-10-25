import random
import time

import numpy as np
import pandas as pd


def main() -> None:
    n = 100000
    indices = [i for i in range(1, n+1)]
    random.shuffle(indices)

    df = pd.DataFrame(
        np.random.uniform(low=0, high=1, size=(n, 2)),
        index=indices,
        columns=["a", "b"],
    )
    df["idx"] = indices
    print(df.shape)

    start_tm = time.time()
    for i in range(1, n+1):
        row = df[df.idx == i].iloc[0]
    end_tm = time.time()
    print(f"df filter: {end_tm - start_tm} sec")
    print(row)

    start_tm = time.time()
    for i in range(1, n+1):
        row = df.loc[i]
    end_tm = time.time()
    print(f"df loc   : {end_tm - start_tm} sec")
    print(row)

    print("DONE")


if __name__ == "__main__":
    main()
