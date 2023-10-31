from pathlib import Path

import boto3


class S3Uploader:
    def __init__(self):
        self._s3_client = boto3.client('s3')

    def upload(self, filepath: Path, bucket_name: str, key: str):
        self._s3_client.upload_file(
            Filename=str(filepath),
            Bucket=bucket_name,
            Key=key
        )
