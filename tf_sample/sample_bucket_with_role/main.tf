terraform {
    required_providers {
        aws = {
            source  = "hashicorp/aws"
            version = "5.17.0"
        }
    }
}

provider "aws" {
    region = "ap-northeast-1"
}

# s3 bucket
resource "aws_s3_bucket" "wararaki_bucket" {
  bucket = "wararaki-bucket"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "wararaki_bucket_access" {
  bucket = aws_s3_bucket.wararaki_bucket.id

  block_public_acls   = true
  block_public_policy = true
  ignore_public_acls  = true
}

# iam role
resource "aws_iam_policy" "bucket_policy" {
  name   = "wararaki-bucket-policy"
  policy = data.aws_iam_policy_document.access_policy.json
}

resource "aws_iam_role" "wararaki_role" {
  name = "wararaki-role"
  assume_role_policy = data.aws_iam_policy_document.assume_policy.json
}

resource "aws_iam_role_policy_attachment" "wararaki_bucket_policy" {
  role       = aws_iam_role.wararaki_role.name
  policy_arn = aws_iam_policy.bucket_policy.arn
}
