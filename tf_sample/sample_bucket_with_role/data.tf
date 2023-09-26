data "aws_iam_policy_document" "assume_policy" {
  statement {
    actions = ["sts:AssumeRole"]
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["s3.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "access_policy" {
  statement {
    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:ListBucket",
      "s3:DeleteObject"
    ]
    effect = "Allow"

    resources = [
      "arn:aws:s3:::*/*",
      "arn:aws:s3:::my-bucket-name"
    ]
  }
}