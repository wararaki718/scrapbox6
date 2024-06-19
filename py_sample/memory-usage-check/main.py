import psutil
import sys

import pandas as pd


def main() -> None:
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    print(df)
    print()

    print(sys.getrefcount(df))
    print(psutil.virtual_memory().percent)
    print()

    df2 = df

    print(sys.getrefcount(df))
    print(sys.getrefcount(df2))
    print(psutil.virtual_memory().percent)
    
    print(sys.getsizeof(df))
    print(sys.getsizeof(df2))
    print()

    df["B"][0] = 6.0
    print(df)
    print(df2)

    del df2
    print(sys.getrefcount(df))
    print(psutil.virtual_memory().percent)
    print()

    df2 = df.astype({"A": "int32"})
    print(sys.getrefcount(df))
    print(sys.getrefcount(df2))
    print(psutil.virtual_memory().percent)
    
    print(sys.getsizeof(df))
    print(sys.getsizeof(df2))

    print()

    df["B"][0] = 12.0
    print(df)
    print(df2)
    print()
    
    print("DONE")


if __name__ == "__main__":
    main()
