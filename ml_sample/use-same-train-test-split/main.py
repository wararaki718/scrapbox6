import numpy as np
from sklearn.model_selection import train_test_split


def main() -> None:
    x = np.arange(10)
    x_train, x_test = train_test_split(x, test_size=0.3, shuffle=False, random_state=42)
    print(f"x_train: {x_train}")
    print(f"x_test : {x_test}")

    x_train, x_test = train_test_split(x, test_size=0.3, shuffle=False, random_state=42)
    print(f"x_train: {x_train}")
    print(f"x_test : {x_test}")

    x_train, x_test = train_test_split(x, test_size=0.3, shuffle=True, random_state=42)
    print(f"shuffled x_train: {x_train}")
    print(f"shuffled x_test : {x_test}")

    x_train, x_test = train_test_split(x, test_size=0.3, shuffle=True, random_state=42)
    print(f"shuffled x_train: {x_train}")
    print(f"shuffled x_test : {x_test}")
    print("DONE")


if __name__ == "__main__":
    main()
