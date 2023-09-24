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

resource "aws_s3_bucket" "alb_logs" {
  bucket = var.alb_logs_bucket_name # グローバルで一意なバケット名
}

resource "aws_s3_bucket_policy" "alb" {
  bucket = aws_s3_bucket.alb_logs.bucket
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      # ALB
      {
        Sid    = "ALBAccessLogWrite",
        Effect = "Allow",
        Principal = {
          AWS = data.aws_elb_service_account.service_account.arn // リージョンで固定の ELB サービス AWS アカウント ID
        },
        Action = "s3:PutObject",
        Resource = [
          "arn:aws:s3:::${var.alb_logs_bucket_name}/${var.alb_logs_prefix}/AWSLogs/${data.aws_caller_identity.current.account_id}/*"
          # ALB が複数ある場合はここに列挙する
        ]
      },
    ]
  })
}
