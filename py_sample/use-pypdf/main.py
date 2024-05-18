from typing import List

from downloader import PDFDownloader
from extractor import PDFTextExtractor


def show(texts: List[str]) -> None:
    print(f"number of pages: {len(texts)}")
    print("contents:")
    for i, text in enumerate(texts, start=1):
        print(f"  #{i} -> {text[:10]}...")
    print()


def main():
    uri = "https://arxiv.org/pdf/2109.10086"

    downloader = PDFDownloader()
    pdf = downloader.download(uri)
    print(f"'{uri}' is downloaded!")
    print()

    extractor = PDFTextExtractor()
    texts = extractor.extract(pdf)
    
    show(texts)
    
    print("DONE")


if __name__ == "__main__":
    main()

