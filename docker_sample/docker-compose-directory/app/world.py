from pathlib import Path


def main() -> None:
    filepath: Path = Path("/opt/app/data/world.txt")
    filepath.touch()
    print("world")


if __name__ == "__main__":
    main()
