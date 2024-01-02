import json


class JSONSerializer:
    def serialize(self, data: dict) -> str:
        json_string = json.dumps(data)
        return json_string


def main() -> None:
    data = {
        "id": 1,
        "name": "hello",
        "age": 12,
    }
    serializer = JSONSerializer()
    result = serializer.serialize(data)
    print(result)
    print("DONE")


if __name__ == "__main__":
    main()
