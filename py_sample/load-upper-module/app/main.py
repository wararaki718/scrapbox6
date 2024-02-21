import sys
from importlib import import_module

sys.path.append("..")
sample = import_module("sample-module.sample").sample


def main() -> None:
    print(sample())
    print("DONE")


if __name__ == "__main__":
    main()
