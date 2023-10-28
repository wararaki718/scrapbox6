import numpy as np


def main() -> None:
    mean = np.zeros(2)

    c = 0.3
    covariance = np.array([
        [1, c],
        [c, 1],
    ])

    # generate data based on covariance
    x, y = np.random.multivariate_normal(mean, covariance, 5000).T
    print(x.shape)
    print(y.shape)
    print()

    # covariance
    a = np.array([[10, 5, 2, 4, 9, 3, 2],[10, 2, 8, 3, 7, 4, 1]])
    print(np.cov(a))
    print()

    c = np.array([3, 2, 1, 5, 7, 2, 1])

    print(np.cov(a, c))
    print()

    print("DONE")


if __name__ == "__main__":
    main()
