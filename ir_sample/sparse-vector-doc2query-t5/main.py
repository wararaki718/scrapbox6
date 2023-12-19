import json

from postprocessor import Postprocessor
from vectorizer import TextVectorizer


def main() -> None:
    model_name = "castorini/doc2query-t5-base-msmarco"
    vectorizer = TextVectorizer(model_name)

    text = """Arthur Robert Ashe Jr. (July 10, 1943 – February 6, 1993) was an American professional tennis player. He won three Grand Slam titles in singles and two in doubles."""
    vectors, tokens = vectorizer.transform(text)
    print(vectors.shape)

    postprocessor = Postprocessor(vectorizer.get_vocabs())
    result = postprocessor.transform(vectors)
    print(json.dumps(result, indent=4))
    print("DONE")


if __name__ == "__main__":
    main()
    