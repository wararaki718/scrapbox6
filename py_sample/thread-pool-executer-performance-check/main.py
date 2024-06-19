from concurrent.futures import ThreadPoolExecutor
from random import random
from time import sleep
 

def task(name: int) -> str:
    sleep(random())
    return f'Task: {name} done.'


def main() -> None: 
    with ThreadPoolExecutor(10) as executor:
        for result in executor.map(task, range(10)):
            print(result)
    print("DONE")


if __name__ == "__main__":
    main()
