from typing import Generator, List


class CustomForIterator:
    def __init__(self, items: List[str]) -> None:
        self._items = items

    def __iter__(self) -> Generator[str, None, None]:
        for item in self._items:
            yield item


class CustomYieldFromIterator:
    def __init__(self, items: List[str]) -> None:
        self._items = items

    def __iter__(self) -> Generator[str, None, None]:
        yield from self._items


def main() -> None:
    items = ["item", "sample", "test", "hello"]

    iterator1 = CustomForIterator(items)
    print("^".join(iterator1))

    iterator2 = CustomYieldFromIterator(items)
    print("_".join(iterator2))

    print("DONE")


if __name__ == "__main__":
    main()
