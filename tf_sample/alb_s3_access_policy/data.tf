# elb
data "aws_elb_service_account" "elb_service_account" {}

data "aws_caller_identity" "caller_identity" {}

# policy
data "aws_iam_policy_document" "tf_iam_policy_document_alb_log" {
  statement {
    effect = "Allow"
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_elb_service_account.elb_service_account.id}:root"]
    }
    actions   = ["s3:PutObject"]
    resources = ["arn:aws:s3:::${aws_s3_bucket.tf_bucket_alb_log.bucket}/${var.alb_log_prefix}/AWSLogs/${data.aws_caller_identity.caller_identity.account_id}/*"]
  }

  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["delivery.logs.amazonaws.com"]
    }
    actions   = ["s3:PutObject"]
    resources = ["arn:aws:s3:::${aws_s3_bucket.tf_bucket_alb_log.bucket}/${var.alb_log_prefix}/AWSLogs/${data.aws_caller_identity.caller_identity.account_id}/*"]
    condition {
      test     = "StringEquals"
      variable = "s3:x-amz-acl"
      values   = ["bucket-owner-full-control"]
    }
  }

  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["delivery.logs.amazonaws.com"]
    }
    actions   = ["s3:GetBucketAcl"]
    resources = ["arn:aws:s3:::${aws_s3_bucket.tf_bucket_alb_log.bucket}"]
  }
}
