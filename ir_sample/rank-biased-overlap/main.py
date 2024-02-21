import rbo


def main() -> None:
    a = ["item1", "item2", "item3", "item4", "item5"]
    b = ["item5", "item4", "item3", "item2", "item1"]

    score = rbo.RankingSimilarity(a, b).rbo()
    print(score)

    print("DONE")


if __name__ == "__main__":
    main()
