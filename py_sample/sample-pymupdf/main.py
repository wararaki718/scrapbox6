import pymupdf

from downloader import PDFDownloader


def main() -> None:
    uri = "https://arxiv.org/pdf/2109.10086"

    downloader = PDFDownloader()
    pdf = downloader.download(uri)
    print(f"'{uri}' is downloaded!")
    print()

    document = pymupdf.open(stream=pdf)
    for page in document:
        text = page.get_text()
        print(f"{text[:50]}...{text[-50:]}")
    print("DONE")


if __name__ == "__main__":
    main()
