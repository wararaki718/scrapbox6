from pathlib import Path

from app.uploader import S3Uploader


def pipeline() -> None:
    uploader = S3Uploader()

    filepath = Path("sample.txt")
    bucket_name = "my-bucket"
    key = "sample.txt"

    uploader.upload(filepath, bucket_name, key)

    print("DONE")
