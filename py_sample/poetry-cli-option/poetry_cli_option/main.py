import typer
from typing_extensions import Annotated


def main(
    name: Annotated[str, typer.Option(help="name")] = "default"
) -> None:
    print(f"name='{name}'")
    print("DONE")


def batch() -> None:
    typer.run(main)
