from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from app.uploader import S3Uploader


@pytest.fixture
def s3_uploader() -> Generator[S3Uploader, None, None]:
    stub = MagicMock()
    stub.upload_file = MagicMock()

    with patch("boto3.client", MagicMock(return_value=stub)):
        yield S3Uploader()


def test_upload(s3_uploader: S3Uploader) -> None:
    filepath = "sample.txt"
    bucket_name = "my-bucket"
    key = "sample.txt"

    s3_uploader.upload(filepath, bucket_name, key)

    s3_uploader._s3_client.upload_file.assert_called_once_with(
        Filename=filepath,
        Bucket=bucket_name,
        Key=key
    )
