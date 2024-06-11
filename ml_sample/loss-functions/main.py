import numpy as np

from loss import softmax_loss, norm_softmax_loss


def main() -> None:
    x = np.random.rand(100, 10)
    print(x.shape)

    y = np.random.randint(low=0, high=1, size=(1, 100))
    print(y.shape)

    loss = softmax_loss(x, y)
    print(loss)

    loss = norm_softmax_loss(x, y, 0.01)
    print(loss)
    print("DONE")


if __name__ == "__main__":
    main()
