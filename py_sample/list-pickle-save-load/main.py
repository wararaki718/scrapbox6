import pickle
from pathlib import Path


def main() -> None:
    a = [1, 2, 3]
    print(a)

    filepath = Path("sample.pickle")
    with open(filepath, "wb") as f:
        pickle.dump(a, f)
    print(filepath.exists())

    with open(filepath, "rb") as f:
        b = pickle.load(f)
    print(b)

    filepath.unlink()
    print(filepath.exists())

    print("DONE")


if __name__ == "__main__":
    main()
