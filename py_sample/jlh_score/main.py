from jlh import JLHScore
from text import TextTokenizer


def main() -> None:
    query = "pen red"
    documents = [
        "this is a pen",
        "i have a pen",
        "that is a pen",
        "the pen is blue",
        "this is a pencil",
        "the pencil",
        "good pencil",
    ]

    tokenizer = TextTokenizer()
    q_tokens = tokenizer.tokenize(query)
    d_tokens = [
        tokenizer.tokenize(document)
        for document in documents
    ]

    jlh_score = JLHScore()
    score = jlh_score.compute(q_tokens, d_tokens)
    print(score)

    print("DONE")


if __name__ == "__main__":
    main()
