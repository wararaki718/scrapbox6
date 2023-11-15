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

def main() -> None:
    a = np.arange(6).reshape((2, 3))
    b = np.arange(12).reshape((3, 4))
    c = b[:, [0, 1, 3]]
    print("a =")
    print(a)
    print()

    print("b =")
    print(b)
    print()

    print("c =")
    print(c)
    print()

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


    print("## linear algebra")
    print("np.dot(a, c) =")
    print(np.dot(a, c))
    print()

    print("np.matmul(a, c) =")
    print(np.matmul(a, c))
    print()

    print("a @ c =")
    print(a @ c)
    print()

    print("np.inner(a, c.T) =")
    print(np.inner(a, c.T))
    print()

    print("inner_product(a, c) =")
    print(inner_product(a, c))
    print()

    print("DONE")


if __name__ == "__main__":
    main()
