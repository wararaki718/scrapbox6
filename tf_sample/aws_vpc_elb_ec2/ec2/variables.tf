variable "ami_id" {
    type = string
    default = "ami-0d5eff06f840b45e9"
}

variable "instance_type" {
    type = string
    default = "t2.micro"
}

variable "ec2_subnet_a_id" {
    type = string
}

variable "ec2_subnet_c_id" {
    type = string
}

variable "ec2_security_group_id" {
    type = string
}
