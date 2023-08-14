from collections import Counter

from pydantic import BaseModel


class CustomKey(BaseModel):
    name: str
    value: int

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CustomKey):
            return NotImplemented
        return self.name == other.name


def main() -> None:
    counter = Counter[CustomKey]()
    items = [
        CustomKey(name="a", value=1),
        CustomKey(name="b", value=2),
        CustomKey(name="a", value=3),
        CustomKey(name="c", value=4),
        CustomKey(name="b", value=5),
    ]
    counter.update(items)
    print(counter)
    print("DONE")


if __name__ == "__main__":
    main()
