from abc import abstractmethod


class Base:
    @abstractmethod
    def sample(self, x):
        pass


class Derived(Base):
    def sample(self, x: int) -> int:
        return x + 1


class Derived2(Base):
    def sample(self, x: str) -> str:
        return x + "1"


def main() -> None:
    d = Derived()
    result = d.sample(1)
    print(result)
    d2 = Derived2()
    result2 = d2.sample("1")
    print(result2)
    print("DONE")


if __name__ == "__main__":
    main()
