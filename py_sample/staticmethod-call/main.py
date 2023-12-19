class Sample:
    @staticmethod
    def add(a: int, b: int) -> int:
        return a + b


class SquareSample(Sample):
    @staticmethod
    def add(a: int, b: int) -> int:
        return (a + b) ** 2


class CustomSample(Sample):
    def add(self, a: int, b: int) -> int:
        return a + b + 1


class Custom2Sample:
    def add(self, a: int, b: int) -> int:
        return a + b


def main() -> None:
    sample = Sample()
    square = SquareSample()
    custom = CustomSample()

    a = 1
    b = 2

    # object call
    print(sample.add(a, b))
    print(square.add(a, b))
    print(custom.add(a, b))
    print()

    # class call
    print(Sample.add(a, b))
    print(SquareSample.add(a, b))

    try:
        print(CustomSample.add(a, b))  # error
    except Exception as e:
        print("error")

    print(CustomSample.add("dummy self", a, b))
    print(Custom2Sample.add("nanndemo ok", a, b))
    print("DONE")


if __name__ == "__main__":
    main()
