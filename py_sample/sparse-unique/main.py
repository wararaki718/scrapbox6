from time import time
import numpy as np
import scipy.sparse as sps
from tqdm import tqdm


def sps_unique(a: sps.csr_matrix) -> sps.csr_matrix:
    keep = [True for _ in range(a.shape[0])]
    for i in tqdm(range(a.shape[0]), total=a.shape[0]):
        if not keep[i]:
            continue
        for j in range(i+1, a.shape[0]):
            if not keep[j]:
                continue
            keep[j] &= (a[i] != a[j]).nnz != 0
    indices = [i for i, k in enumerate(keep) if k]
    return a[indices]


def main() -> None:
    a = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 0],
    ])
    b = np.unique(a, axis=0)
    print(a)
    print()

    print(b)
    print()

    c = sps.csr_matrix(a)
    d = sps_unique(c)
    print(d.toarray())

    a = np.random.randint(2, size=(1000, 100))
    start_tm = time()
    b = np.unique(a, axis=0)
    print(f"np.unique time: {time() - start_tm}")
    print()

    c = sps.csr_matrix(a)
    start_tm = time()
    d = sps_unique(c)
    print(f"sps unique time: {time() - start_tm}")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
