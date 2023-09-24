terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "5.17.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "2.4.0"
    }
  }
  required_version = ">= 1.2.0"
}

provider "aws" {
  region = "ap-northeast-1"
}

locals {
  prefix = "sample_wararaki"
  bucket_name = "${local.prefix}-bucket"
}


module "s3" {
  source = "./modules/s3"
  bucket_name = local.bucket_name
}

module "lambda" {
  source = "./modules/lambda"
  prefix = local.prefix
}

module "event_bridge" {
  source = "./modules/event_bridge"
  bucket_name = local.bucket_name
  prefix = local.prefix
  aws_lambda_function_test_lambda_arn = module.lambda.aws_lambda_function_test_lambda_arn
}
