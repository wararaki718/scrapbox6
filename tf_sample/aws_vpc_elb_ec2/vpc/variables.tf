variable "vpc_cidr_block" {
    type = string
    default = "10.0.0.0/16"
}

variable "alb_a_cidr_block" {
    type = string
    default = "10.0.1.0/24"
}

variable "alb_c_cidr_block" {
    type = string
    default = "10.0.2.0/24"
}

variable "cidr_block_web_a" {
    type = string
    default = "10.0.3.0/24"
}

variable "cidr_block_web_c" {
    type = string
    default = "10.0.4.0/24"
}

variable "availability_zone_a" {
    type = string
    default = "ap-northeast-1a"
}

variable "availability_zone_c" {
    type = string
    default = "ap-northeast-1c"
}
