import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def main() -> None:
    x_a = np.random.randn(1, 128)
    x_b = np.random.randn(10, 128)
    y = cosine_similarity(x_a, x_b)
    print(y)
    print()

    x_a_norm = x_a / np.linalg.norm(x_a, axis=1).reshape(-1, 1)
    x_b_norm = x_b / np.linalg.norm(x_b, axis=1).reshape(-1, 1)
    y = np.matmul(x_a_norm, x_b_norm.T)
    print(y)
    print("DONE")


if __name__ == "__main__":
    main()
