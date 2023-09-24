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
  cidr_vpc = "10.255.0.0/16"
  cidr_public1 = "10.255.1.0/24"
  cidr_public2 = "10.255.2.0/24"
  az1 = "ap-northeast-1a"
  az2 = "ap-northeast-1c"
}
