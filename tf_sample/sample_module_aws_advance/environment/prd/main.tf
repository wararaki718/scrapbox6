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

module "network" {
  source   = "../../module/vpc"
  system   = "test"
  env      = "prd"
  cidr_vpc = "10.1.0.0/16"
  cidr_public = [
    "10.1.1.0/24",
    "10.1.2.0/24",
    "10.1.3.0/24"
  ]
  cidr_private = [
    "10.1.101.0/24",
    "10.1.102.0/24",
    "10.1.103.0/24"
  ]
  cidr_secure = [
    "10.1.201.0/24",
    "10.1.202.0/24",
    "10.1.203.0/24"
  ]
}
