import torch


def main() -> None:
    print(torch.cuda.is_available())
    print("DONE")


if __name__ == "__main__":
    main()
