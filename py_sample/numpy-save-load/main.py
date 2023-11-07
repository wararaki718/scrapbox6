from pathlib import Path

import numpy as np


def main() -> None:
    a = np.array([1, 2, 3])
    print(a)

    filepath = Path("sample.npy")
    np.save(filepath, a)
    print(filepath.exists())

    b = np.load(filepath, allow_pickle=True)
    print(b)

    filepath.unlink()
    print(filepath.exists())

    print("DONE")


if __name__ == "__main__":
    main()
