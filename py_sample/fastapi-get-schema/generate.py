import yaml

from main import app


def postprocess(schema: dict) -> dict:
    properties = schema["components"]["schemas"]["ValidationError"]["properties"]
    properties["loc"]["items"] = {"type": "string"}
    schema["components"]["schemas"]["ValidationError"]["properties"] = properties
    return schema


def main() -> None:
    filepath = "./schema.yml"
    schema = app.openapi()
    schema = postprocess(schema)

    with open(filepath, "wt") as f:
        yaml.dump(schema, f)
    print("DONE")


if __name__ == "__main__":
    main()
