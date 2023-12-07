class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

    @classmethod
    def sub(cls, a: int, b: int) -> int:
        return a - b

    @staticmethod
    def multiply(a: int, b: int) -> int:
        return a * b


class SquareCalculator(Calculator):
    def add(self, a: int, b: int) -> int:
        return (a + b) ** 2
    
    @classmethod
    def sub(self, a: int, b: int) -> int:
        return (a - b) ** 2
    
    @staticmethod
    def multiply(a: int, b: int) -> int:
        return (a * b) ** 2


class CustomSquareCalculator(Calculator):
    def add(self, a: int, b: int) -> int:
        return (a + b) ** 2 + 1
    
    def sub(self, a: int, b: int) -> int:
        return (a - b) ** 2 + 1
    
    def multiply(self, a: int, b: int) -> int:
        return (a * b) ** 2 + 1
