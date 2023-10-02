output "vpc_id" {
    value = aws_vpc.vpc.id
}

output "ec2_subnet_a_id" {
    value = aws_subnet.public_web_a.id
}

output "ec2_subnet_c_id" {
    value = aws_subnet.public_web_c.id
}

output "alb_subnet_a_id" {
    value = aws_subnet.public_alb_a.id
}

output "alb_subnet_c_id" {
    value = aws_subnet.public_alb_c.id
}
