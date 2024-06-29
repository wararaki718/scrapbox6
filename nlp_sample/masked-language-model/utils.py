from typing import List, Dict, Union


def show(results: List[Dict[str, Union[str, int, float]]]) -> None:
    for i, result in enumerate(results):
        print(f"### {i}-th ###")
        for key, value in result.items():
            print(f"{key}: {value}")
        print()
    print()