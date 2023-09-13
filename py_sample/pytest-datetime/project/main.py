from module.func import datediff


def main() -> None:
    days = 10
    result = datediff(days)
    print(result)
    print("DONE")


if __name__ == "__main__":
    main()
