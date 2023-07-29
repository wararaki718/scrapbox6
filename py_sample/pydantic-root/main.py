from typing import List

from pydantic import BaseModel

class ListModel(BaseModel):
    items: List[str]


class ListRootModel(BaseModel):
    __root__: List[str]


def main() -> None:
    print("__root__:")
    items = ListRootModel.parse_obj(["hello", "world"])
    print(items)
    print(items.json())
    print()

    print("items:")
    items2 = ListModel(items=["test", "sample"])
    print(items2)
    print(items2.json())
    print()
    
    print("DONE")


if __name__ == "__main__":
    main()
