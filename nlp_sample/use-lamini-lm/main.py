from transformers import pipeline


def main() -> None:
    checkpoint = "MBZUAI/LaMini-T5-61M"

    model = pipeline("text2text-generation", model = checkpoint)

    input_prompt = 'Where is the capital of Japan?'
    generated_text = model(input_prompt, max_length=512, do_sample=True)[0]["generated_text"]

    print("Question: ", input_prompt)
    print("Response: ", generated_text)

    print("DONE")


if __name__ == "__main__":
    main()
