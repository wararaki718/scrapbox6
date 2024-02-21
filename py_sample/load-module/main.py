from importlib import import_module

sample = import_module("sample-module.sample").sample
example = import_module("sample-module.example_module.example").example

def main() -> None:
    print(sample())
    print(example())
    print("DONE")


if __name__ == "__main__":
    main()
