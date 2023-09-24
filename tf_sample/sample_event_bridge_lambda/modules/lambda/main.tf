data "aws_iam_policy" "iam_policy_AWSLambdaBasicExecutionRole" {
  arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

locals {
  # Lambda
  lambda_service_role_name = "${var.prefix}-lambda-service-role"
  lambda_file_name         = "lambda_function"
  lambda_func              = "sample_lambda_function"
  lambda_runtime           = "python3.10"
  entry_point              = "lambda_handler"
  output_source_dir        = "archive/${local.lambda_file_name}.zip"
}

resource "aws_iam_policy" "iam_policy_AWSLambdaBasicExecutionRole" {
  name   = local.lambda_service_role_name
  policy = data.aws_iam_policy.iam_policy_AWSLambdaBasicExecutionRole.policy
}

resource "aws_iam_role_policy_attachment" "lambda_policy" {
  role       = aws_iam_role.iam_role_for_lambda.name
  policy_arn = aws_iam_policy.iam_policy_AWSLambdaBasicExecutionRole.arn
}

data "aws_iam_policy_document" "assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "iam_role_for_lambda" {
  name               = "iam_for_lambda"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

data "archive_file" "function_info" {
  type        = "zip"
  source_file = "${local.lambda_file_name}.py"
  output_path = local.output_source_dir
}

resource "aws_cloudwatch_log_group" "cloudwatch_log" {
  name = "/aws/lambda/${local.lambda_func}"
}

resource "aws_lambda_permission" "logging" {
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.test_lambda.function_name
  principal     = "events.amazonaws.com"
  # source_arn    = aws_cloudwatch_event_rule.eventbridge_rule.arn
}

resource "aws_lambda_function" "test_lambda" {
  function_name = local.lambda_func
  role          = aws_iam_role.iam_role_for_lambda.arn
  filename      = data.archive_file.function_info.output_path
  handler       = "${local.lambda_file_name}.${local.entry_point}"
  runtime       = local.lambda_runtime

  source_code_hash = data.archive_file.function_info.output_base64sha256

  depends_on = [aws_iam_role_policy_attachment.lambda_policy, aws_cloudwatch_log_group.cloudwatch_log]
}
