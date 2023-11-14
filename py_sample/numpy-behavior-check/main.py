import numpy as np

def inner_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert x.shape[1] == y.shape[0]
    ty = y.T
    result = []
    for x_ in x:
        row = []
        for y_ in ty:
            tmp = 0
            # sum
            for i in range(x.shape[1]):
                tmp += x_[i] * y_[i]
            row.append(tmp)
        result.append(row)
    return np.array(result)


def multiply(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert x.shape == y.shape

    result = []
    for i in range(x.shape[0]):
        tmp = []
        for j in range(x.shape[1]):
            tmp.append(x[i][j] * y[i][j])
        result.append(tmp)
    return np.array(result)

def main() -> None:
    a = np.arange(6).reshape((2, 3))
    b = np.arange(12).reshape((3, 4))
    print("a =")
    print(a)
    print()

    print("b =")
    print(b)
    print()

    print("## linear algebra")
    print("np.dot(a, b) =")
    print(np.dot(a, b))
    print()

    print("np.matmul(a, b) =")
    print(np.matmul(a, b))
    print()

    print("a @ b =")
    print(a @ b)
    print()

    print("np.inner(a, b.T) =")
    print(np.inner(a, b.T))
    print()

    print("inner_product(a, b) =")
    print(inner_product(a, b))
    print()
    print("-----------")
    print()

    c = np.arange(8).reshape((2, 4))
    print("c =")
    print(c)
    print()

    d = np.arange(8).reshape((2, 4))
    print("d =")
    print(d)
    print()

    print("## mathematical functions")
    print("np.multiply(c, d) =")
    print(np.multiply(c, d))
    print()

    print("c * d =")
    print(c * d)
    print()

    print("multiply(c, d) =")
    print(multiply(c, d))
    print()

    print("DONE")


if __name__ == "__main__":
    main()
