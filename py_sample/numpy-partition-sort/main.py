import numpy as np


def main() -> None:
    matrix = np.random.random((10, 20))

    keep_n_top = 5

    index_matrix = np.argpartition(-matrix, keep_n_top, axis=1)[:, :keep_n_top]
    scores = np.take_along_axis(matrix, index_matrix, axis=1)
    sorted_indices = np.argsort(-scores)
    rankings = np.take_along_axis(index_matrix, sorted_indices, axis=1)

    print(matrix[0])
    print()
    print(matrix[0, rankings[0]])
    print()

    print("DONE")


if __name__ == "__main__":
    main()
