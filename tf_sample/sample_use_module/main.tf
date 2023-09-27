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

module "iam_role" {
  source = "./module/iam_role"
  role_name = "wararaki_bucket_role"
  policy_name = "wararaki_bucket_policy"
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
