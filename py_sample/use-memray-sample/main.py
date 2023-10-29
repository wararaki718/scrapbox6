import time


def main() -> None:
    time.sleep(1)
    result = [
        1 for _ in range(1024 * 1024 * 1024)
    ]
    time.sleep(1)
    del result
    time.sleep(1)

    print("DONE")


if __name__ == "__main__":
    main()
