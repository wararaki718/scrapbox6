from calculator import Calculator, CustomSquareCalculator, SquareCalculator



def main() -> None:
    print("Calculator class:")
    calculator = Calculator()

    result = calculator.add(1, 2)
    print(f"1 + 2 = {result}")
    
    result = calculator.sub(1, 2)
    print(f"1 - 2 = {result} (object classmethod)")
    
    result = calculator.multiply(1, 2)
    print(f"1 * 2 = {result} (object staticmethod)")

    result = Calculator.sub(1, 2)
    print(f"1 - 2 = {result} (classmethod)")

    result = Calculator.multiply(1, 2)
    print(f"1 * 2 = {result} (staticmethod)")
    print()

    print("SquareCalculator class:")
    square_calculator = SquareCalculator()

    result = square_calculator.add(1, 2)
    print(f"(1 + 2) ** 2 = {result}")

    result = square_calculator.sub(1, 2)
    print(f"(1 - 2) ** 2 = {result}")

    result = square_calculator.multiply(1, 2)
    print(f"(1 * 2) ** 2 = {result}")

    result = SquareCalculator.sub(1, 2)
    print(f"(1 - 2) ** 2 = {result} (classmethod)")

    result = SquareCalculator.multiply(1, 2)
    print(f"(1 * 2) ** 2 = {result} (staticmethod)")
    print()

    print("CustomSquareCalculator class:")
    square_calculator = CustomSquareCalculator()

    result = square_calculator.add(1, 2)
    print(f"(1 + 2) ** 2 + 1 = {result}")

    result = square_calculator.sub(1, 2)
    print(f"(1 - 2) ** 2 + 1 = {result} (object classmethod)")

    result = square_calculator.multiply(1, 2)
    print(f"(1 * 2) ** 2 + 1 = {result} (object staticmethod)")

    try:
        result = CustomSquareCalculator.sub(1, 2)
        print(f"(1 - 2) ** 2 + 1 = {result} (classmethod)")
    except Exception as e:
        print(e)

    try:
        result = CustomSquareCalculator.multiply(1, 2)
        print(f"(1 * 2) ** 2 + 1 = {result} (staticmethod)")
    except Exception as e:
        print(e)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
