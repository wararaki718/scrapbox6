from pathlib import Path
from typing import List

from cumulative import CumulativeAverager
from factorial import Factorial
from logger import CustomLogger
from serializer import DataSerializer, JsonSerializer, YamlSerializer
from watch import ExecutionTimer


@ExecutionTimer
def sample_functions(x: List[int]) -> List[int]:
    return [x_i ** 2 for x_i in x]


def main() -> None:
    print("stateful callables:")
    stream_average = CumulativeAverager()
    for i, v in enumerate([12, 13, 11], start=1):
        print(f"{i}-th value: {v}, average: {stream_average(v)}")
    print()

    print("caching:")
    factorial = Factorial()
    for i in range(1, 10):
        print(f"{i}! = {factorial(i)}")
    print()

    print("useful apis:")
    filename = Path("log.txt")
    logger = CustomLogger(filename)
    for msg in ["hello", "world"]:
        logger(msg)
    print(filename.read_text())
    print()

    filename.unlink(missing_ok=True)

    print("decorator:")
    x = list(range(1000))
    _ = sample_functions(x)
    print()

    print("strategy pattern:")
    data = {
        "name": "sample",
        "age": 30,
        "city": ["tokyo", "osaka"],
    }
    serializer = DataSerializer(JsonSerializer())
    json = serializer(data)
    print("json:")
    print(json)
    print()

    serializer.switch_strategy(YamlSerializer())
    yaml = serializer(data)
    print("yaml:")
    print(yaml)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
