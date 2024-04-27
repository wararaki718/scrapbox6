import torch


def main() -> None:
    x = torch.tensor(5.0, requires_grad=True)
    
    print("init:")
    print(f"x={x}, grad={x.grad}")
    print()
    y = 2 * x
    y.backward()

    print("backward:")
    print(f"x={x}, grad={x.grad}")
    print()

    x = torch.tensor(5.0, requires_grad=True)
    y = 2 * x
    z = 5 * y

    z.backward()

    print("backward 2:")
    print(f"x={x}, grad={x.grad}")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
