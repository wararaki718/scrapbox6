from utils import custom_decorator, sample_decorator


@custom_decorator
@sample_decorator
def add(x: int, y: int) -> int:
    return x + y


def main() -> None:
    result = add(1, 1)
    print(result)
    print("DONE")


if __name__ == "__main__":
    main()
