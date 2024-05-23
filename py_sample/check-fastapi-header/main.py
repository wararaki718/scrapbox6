from fastapi import Header


def main() -> None:
    header = Header(default="default")
    print(header)
    print()

    header = Header(default=None)
    print(header)
    print("DONE")


if __name__ == "__main__":
    main()
