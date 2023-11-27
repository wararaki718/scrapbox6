from loader import UserLoader, ErrorUserLoader

def main() -> None:
    df = UserLoader.load()
    print(df)
    print()

    try:
        df = ErrorUserLoader.load()
        print(df)
    except Exception:
        print("load error!")
        print()

    print("DONE")


if __name__ == "__main__":
    main()
