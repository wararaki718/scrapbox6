from schema.model import Pet, Pets, Error


def main() -> None:
    print("pet:")
    pet = Pet(id=1, name="name", tag="dog")
    print(pet)
    print()

    print("pets:")
    pets = Pets.parse_obj([{"id": 2, "name": "test", "tag": "cat"}])
    print(pets)
    print(pets.json())
    print()

    print("error:")
    error = Error(code=3, message="hello")
    print(error)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
