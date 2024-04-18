def main() -> None:
    a = [[3], [4, 5, 6], [7, 8]]
    b = [[3], [4, 5, 6], [7, 8], [9, 10, 11, 12]]

    print(max(a, key=lambda x: len(x)))
    print(max(b, key=lambda x: len(x)))
    print("DONE")


if __name__ == "__main__":
    main()
