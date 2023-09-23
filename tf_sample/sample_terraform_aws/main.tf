provider "aws" {
  region = "ap-northeast-1"
}

resource "aws_instance" "example" {
  ami = "ami-0d729a60"
  instance_type = "t2.micro"
}
