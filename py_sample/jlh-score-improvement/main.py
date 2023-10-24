import numpy as np

from jlh import JLHScore


def main() -> None:
    queries = [
        "pen red pen pen",
        "this pen blue blue pen",
        "this blue pen this"
    ]
    documents = [
        "this is a pen",
        "i have a pen",
        "that is a pen",
        "the pen is blue",
        "this is a pencil",
        "the pencil",
        "good pencil",
    ]

    jlh_score = JLHScore()
    scores, tokens = jlh_score.compute(queries, documents)
    print(scores)
    print(tokens)
    print()

    print(tokens[np.nonzero(scores)])
    print()

    print("DONE")


if __name__ == "__main__":
    main()
