from pathlib import Path

from func import Downloader


def main() -> None:
    filepath = Path("resources/sample.html")
    url = "https://example.com/"

    downloader = Downloader()
    downloader.download(url, filepath)
    print(filepath.read_text())
    print("done")


if __name__ == "__main__":
    main()
