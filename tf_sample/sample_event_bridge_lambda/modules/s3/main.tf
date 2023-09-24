resource "aws_s3_bucket" "bucket" {
  bucket = var.bucket_name
}

resource "aws_s3_object" "input_folder" {
  bucket = aws_s3_bucket.bucket.id
  key    = "input-file/"
}

resource "aws_s3_bucket_notification" "bucket_notification" {
  bucket      = aws_s3_bucket.bucket.bucket
  eventbridge = true
}
