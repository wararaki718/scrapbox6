import numpy as np
import pandas as pd


def main() -> None:
    df = pd.DataFrame([
        [1, 10, 100],
        [2, None, 200],
        [np.nan, 30, 300],
        [None, None, np.nan],
    ], columns=["a", "b", "c"])
    print(df)
    print()

    print("sum(axis=1, skipna=True):")
    print(df.sum(axis=1, skipna=True))
    print()

    print("boolean operation:")
    print(df["a"] > 1)
    print(df["a"].notna())
    print((df["a"] > 1) & (df["a"].notna()))
    print((df["a"] > 1) | (df["a"].notna()))

    print("DONE")


if __name__ == "__main__":
    main()
