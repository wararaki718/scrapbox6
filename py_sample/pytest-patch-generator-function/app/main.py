from func import CustomGenerator


def main() -> None:
    generator = CustomGenerator()
    for i in generator.generate():
        print(i)
    print("done")


if __name__ == "__main__":
    main()
