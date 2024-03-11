from typing import Generator, Iterator


def sample():
    yield from iter([1, 2, 3])
    return "ok"


def sample2() -> Generator[int, None, str]:
    msg = yield from sample()
    print(f"message: {msg}")


def sample3() -> Iterator[int]:
    for i in range(4):
        yield i


def sample4() -> Generator[int, None, str]:
    for i in range(4):
        if i == 2:
            return "end"
        yield i


def sample5() -> Generator[int, None, str]:
    yield 1
    return "call StopIteration"


def main() -> None:
    result = sample2()
    print(list(result))

    result2 = sample3()
    print(list(result2))

    result3 = sample4()
    print(list(result3))

    result4 = sample5()
    print(list(result4))

    # stop iteration
    iterator = sample5()
    print(next(iterator))
    try:
        print(next(iterator))
    except StopIteration as e:
        print(e)

    print("DONE")


if __name__ == "__main__":
    main()
