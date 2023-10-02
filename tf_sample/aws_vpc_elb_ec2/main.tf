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

module "vpc" {
    source = "./vpc"
}


resource "aws_security_group" "alb_secruity_group" {
    name ="alb-security-group"
    vpc_id= module.vpc.vpc_id
    ingress{
        from_port = 80
        to_port   = 80
        protocol  = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }
    egress{
       from_port  = 0
       to_port    = 0
       protocol   = "-1"
       cidr_blocks= ["0.0.0.0/0"]
    }
}

resource "aws_security_group" "ec2_security_group" {
    name ="ec2-security-group"
    vpc_id= module.vpc.vpc_id
    ingress{
        from_port = 80
        to_port   = 80
        protocol  = "tcp"
        security_groups =[aws_security_group.alb_secruity_group.id]
    }
   egress{
       from_port  = 0
       to_port    = 0
       protocol   = "-1"
       cidr_blocks=["0.0.0.0/0"]
   }
}


module "ec2" {
    source = "./ec2"
    ami_id = "ami-0d5eff06f840b45e9"
    instance_type = "t2.micro"
    ec2_subnet_a_id = module.vpc.ec2_subnet_a_id
    ec2_subnet_c_id = module.vpc.ec2_subnet_c_id
    ec2_security_group_id = aws_security_group.ec2_security_group.id
}
  
module "alb" {
    source = "./alb"
    public_subnet_a_id = module.vpc.alb_subnet_a_id
    public_subnet_c_id = module.vpc.alb_subnet_c_id
    vpc_id = module.vpc.vpc_id
    ec2_instance_a_id = module.ec2.ec2_instance_a_id
    ec2_instance_c_id = module.ec2.ec2_instance_c_id
    aws_security_group_alb_id = aws_security_group.alb_secruity_group.id
}
