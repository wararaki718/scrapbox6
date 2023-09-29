import tarfile


def main() -> None:
    with tarfile.open("sample.tar.gz", "w:gz") as tar:
        tar.add("data")
    print("DONE")


if __name__ == "__main__":
    main()
