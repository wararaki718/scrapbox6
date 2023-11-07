from pathlib import Path

import torch


def main() -> None:
    a = torch.Tensor([1, 2, 3])
    print(a)

    filepath = Path("./tensor.pth")
    torch.save(a, filepath)
    print(filepath.exists())

    b = torch.load(filepath)
    print(b)

    filepath.unlink()
    print(filepath.exists())

    print("DONE")


if __name__ == "__main__":
    main()
