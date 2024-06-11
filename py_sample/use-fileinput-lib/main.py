import fileinput
from glob import glob


def main() -> None:
    for line in fileinput.input(files=("data/a.txt", "data/b.txt")):
        print(line.strip())
    print()

    for line in fileinput.input(files=sorted(glob("data/*.txt"))):
        print(line.strip())
    print()
    print("DONE")


if __name__ == "__main__":
    main()
