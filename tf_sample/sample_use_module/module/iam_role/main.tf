# iam role
resource "aws_iam_policy" "iam_policy" {
  name   = var.policy_name
  policy = data.aws_iam_policy_document.access_policy.json
}

resource "aws_iam_role" "assume_role" {
  name = var.role_name
  assume_role_policy = data.aws_iam_policy_document.assume_policy.json
}

resource "aws_iam_role_policy_attachment" "access_policy" {
  role       = aws_iam_role.assume_role.name
  policy_arn = aws_iam_policy.iam_policy.arn
}
