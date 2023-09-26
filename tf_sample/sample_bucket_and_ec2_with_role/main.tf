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

resource "aws_iam_policy" "bucket_policy" {
  name        = "my-bucket-policy"
  path        = "/"
  description = "Allow "

  policy = data.aws_iam_policy_document.assume_policy.json
}

resource "aws_iam_role" "wararaki_role" {
  name = "wararaki_role"

  assume_role_policy = data.aws_iam_policy_document.role_policy.json
}

resource "aws_iam_role_policy_attachment" "wararaki_bucket_policy" {
  role       = aws_iam_role.wararaki_role.name
  policy_arn = aws_iam_policy.bucket_policy.arn
}

resource "aws_iam_role_policy_attachment" "cloud_watch_policy" {
  role       = aws_iam_role.wararaki_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

resource "aws_iam_instance_profile" "wararaki_profile" {
  name = "wararaki-profile"
  role = aws_iam_role.wararaki_role.name
}

resource "aws_instance" "web_instances" {
  ami           = "ami-03ab7423a204da002"
  instance_type = "t2.micro"

  iam_instance_profile = aws_iam_instance_profile.wararaki_profile.id
}
