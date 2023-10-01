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

resource "aws_s3_bucket" "demo_bucket" {
  bucket = "${var.bucket_name}"
}


