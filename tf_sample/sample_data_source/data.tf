# AWS ELB のサービス AWS アカウント（リージョン共通）
data "aws_elb_service_account" "service_account" {}

# 現在の AWS アカウント
data "aws_caller_identity" "current" {}
