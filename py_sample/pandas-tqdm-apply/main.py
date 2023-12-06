import pandas as pd
from tqdm import tqdm


def main() -> None:
    tqdm.pandas()

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df["a"] = df.a.progress_apply(lambda x: x ** 2)
    print(df)

    print("DONE")


if __name__ == "__main__":
    main()
