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

resource "aws_s3_bucket" "tf_bucket_alb_log" {
  bucket = "${var.bucket_name}-${data.aws_caller_identity.caller_identity.account_id}"
}

resource "aws_s3_bucket_policy" "tf_bucket_policy_alb_log" {
  bucket = aws_s3_bucket.tf_bucket_alb_log.id
  policy = data.aws_iam_policy_document.tf_iam_policy_document_alb_log.json
}
