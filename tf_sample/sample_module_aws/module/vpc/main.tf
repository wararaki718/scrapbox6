resource "aws_vpc" "vpc" {
  cidr_block           = var.cidr_vpc
  instance_tenancy     = "default"
  enable_dns_hostnames = true
}

resource "aws_subnet" "public1" {
  vpc_id            = aws_vpc.vpc.id
  cidr_block        = var.cidr_public1
  availability_zone = var.az1
}

resource "aws_subnet" "public2" {
  vpc_id            = aws_vpc.vpc.id
  cidr_block        = var.cidr_public2
  availability_zone = var.az2
}
