from llama import Llama

from config import LlamaConfig


def main() -> None:
    config = LlamaConfig()

    # llm definition
    model = Llama.build(**config.dict())

    # check the behavior
    text = "こんにちは、調子はどうでしょうか？"
    response = model.text_completion(text)
    print(f"Q: {text}")
    print(f"A: {response}")

    print("DONE")


if __name__ == "__main__":
    main()
