def main() -> None:
    with open("data/a.txt", "rt") as af, open("data/b.txt", "rt") as bf:
        for line_a, line_b in zip(af, bf):
            print(line_a.strip(), line_b.strip())
        print()
    print("DONE")


if __name__ == "__main__":
    main()
