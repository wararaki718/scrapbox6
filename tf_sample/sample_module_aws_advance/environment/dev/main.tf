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
  env      = "dev"
  cidr_vpc = "10.255.0.0/16"
  cidr_public = [
    "10.255.1.0/24",
    "10.255.2.0/24"
  ]
  cidr_private = [
    "10.255.101.0/24",
    "10.255.102.0/24"
  ]
  cidr_secure = [
    "10.255.201.0/24",
    "10.255.202.0/24"
  ]
}
