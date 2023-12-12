from functools import partial

from transformers import pipeline


class ChatAgent:
    def __init__(self, model_name: str) -> None:
        model = pipeline("text2text-generation", model = model_name)
        self._generate = partial(model, max_length=512, do_sample=True)
    
    def chat(self, prompt: str) -> str:
        response = self._generate(prompt)
        text: str = response[0]["generated_text"]
        return text
