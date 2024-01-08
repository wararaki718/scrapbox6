from functools import partial

from transformers import pipeline


class TextGenerator:
    def __init__(self, model_name: str, max_length: int=512) -> None:
        model = pipeline("text2text-generation", model=model_name)
        self._generate = partial(model, max_length=max_length, do_sample=True)
    
    def generate(self, prompt: str) -> str:
        response = self._generate(prompt)
        text: str = response[0]["generated_text"]
        return text
