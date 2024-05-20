import transformers
import torch

def main() -> None:
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )
    print("model loaded.")

    messages = [
        {"role": "system", "content": "日本語で会話してください。"},
        {"role": "user", "content": "寿司は何ですか？"},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    print(outputs[0]["generated_text"][len(prompt):])
    print("DONE")


if __name__ == "__main__":
    main()
