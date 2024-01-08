from generator import TextGenerator
from preprocessor import TextPreprocessor
from schema.request import Question
from schema.response import Answer


class LLMService:
    def __init__(self) -> None:
        self._preprocessor = TextPreprocessor()
        self._generator = TextGenerator("MBZUAI/LaMini-T5-61M")
    
    def generate(self, question: Question) -> Answer:
        text = self._preprocessor.transform(question.text)
        result = self._generator.generate(text)
        return Answer(text=result)
