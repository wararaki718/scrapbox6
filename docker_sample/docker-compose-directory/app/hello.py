from pathlib import Path


def main() -> None:
    filepath: Path = Path("/opt/app/data/hello.txt")
    filepath.touch()
    print("hello")


if __name__ == "__main__":
    main()
