import gc
from collections import Counter


def main() -> None:
    a = {"a": 1, "b": 2}
    print(a)
    a.clear()
    print(a)

    b = {"a": {"c": 1}, "b": 2}
    c = b["a"]
    d = b
    print(b)
    print(c)
    print(d)
    print()

    # del b["c"]
    # print(b)
    # print(c) # error

    b.clear()
    print(b)
    print(c)
    print(d)
    print()

    gc.collect()
    print(b)
    print(c)
    print(d)
    print()

    del b, d
    gc.collect()
    # print(b) # error
    print(c)
    # print(d) # error
    print()

    print("DONE")


if __name__ == "__main__":
    main()
