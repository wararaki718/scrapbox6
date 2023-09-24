locals {
  # EventBridge
  event_rule               = "${var.prefix}-event-rule"
  eventbridge_service_role = "${var.prefix}-eventbridge-service-role"
}

resource "aws_cloudwatch_event_rule" "eventbridge_rule" {
  name        = local.event_rule
  description = "S3の特定フォルダ配下にファイルアップロードしたときに、Lambdaを起動するルールをIaC化"

  event_pattern = jsonencode({
    "detail-type" : ["Object Created"],
    "source" : ["aws.s3"],
    "detail" : {
      "bucket" : {
        "name" : [var.bucket_name]
      },
      "object" : {
        "key" : [{
          "prefix" : "input-file/"
        }]
      }
    }
  })
}

resource "aws_cloudwatch_event_target" "eventbridge_target" {
  rule = aws_cloudwatch_event_rule.eventbridge_rule.name

  arn  = var.aws_lambda_function_test_lambda_arn
  input_transformer {
    input_paths = {
      "input_bucket_name" : "$.detail.bucket.name",
      "input_s3_key" : "$.detail.object.key"
    }
    input_template = <<TEMPLATE
{"Parameters": {"input_bucket_name":"<input_bucket_name>", "input_s3_key":"<input_s3_key>"}}
    TEMPLATE
  }
  # role_arn = aws_iam_role.eventbridge_service_role.arn # 設定するとエラーが発生する
}

data "aws_iam_policy_document" "eventbridge_policy" {
  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "eventbridge_service_role" {
  name               = local.eventbridge_service_role
  assume_role_policy = data.aws_iam_policy_document.eventbridge_policy.json
}
