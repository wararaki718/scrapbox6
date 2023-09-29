import gzip
from datetime import date
from pathlib import Path

def main() -> None:
    today = str(date.today())
    
    filepath = Path(f"./data/{today}.txt")
    filepath.write_text(f"hello, {today}!!!")

    with open(filepath, "rb") as f:
        with gzip.open(f"./data/{today}.txt.gz", "wb") as gz:
            gz.writelines(f)

    print("DONE")


if __name__ == "__main__":
    main()
