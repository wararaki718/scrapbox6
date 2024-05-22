from pdfminer.high_level import extract_text

from downloader import PDFDownloader


def main() -> None:
    uri = "https://arxiv.org/pdf/2109.10086"

    downloader = PDFDownloader()
    pdf = downloader.download(uri)
    print(f"'{uri}' is downloaded!")
    print()

    data = extract_text(pdf)
    print(data[1000:3000])
    print(len(data))
    print("DONE")


if __name__ == "__main__":
    main()
