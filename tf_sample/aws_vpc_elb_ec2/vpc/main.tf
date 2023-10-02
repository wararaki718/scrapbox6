# create vpc
resource "aws_vpc" "vpc" {
    cidr_block = var.vpc_cidr_block
    instance_tenancy = "default"
    enable_dns_hostnames = true
    enable_dns_support = true
}

# create alb subnet
resource "aws_subnet" "public_alb_a" {
    vpc_id = aws_vpc.vpc.id
    cidr_block = var.alb_a_cidr_block
    map_public_ip_on_launch = true
    availability_zone = var.availability_zone_a
}

resource "aws_subnet" "public_alb_c" {
    vpc_id = aws_vpc.vpc.id
    cidr_block = var.alb_c_cidr_block
    map_public_ip_on_launch = true
    availability_zone = var.availability_zone_c
}

# create web server subnet
resource "aws_subnet" "public_web_a" {
    vpc_id = aws_vpc.vpc.id
    cidr_block = var.cidr_block_web_a
    map_public_ip_on_launch = true
    availability_zone = var.availability_zone_a
}

resource "aws_subnet" "public_web_c" {
    vpc_id = aws_vpc.vpc.id
    cidr_block = var.cidr_block_web_c
    map_public_ip_on_launch = true
    availability_zone = var.availability_zone_c
}

# create internet gateway
resource "aws_internet_gateway" "gateway" {
    vpc_id = aws_vpc.vpc.id
}

# create route table
resource "aws_route_table" "public_route_table" {
    vpc_id = aws_vpc.vpc.id
}

resource "aws_route" "public_route" {
    route_table_id = aws_route_table.public_route_table.id
    gateway_id = aws_internet_gateway.gateway.id
    destination_cidr_block = "0.0.0.0/0"
}

# associate subnet with route table
resource "aws_route_table_association" "alb_association_a" {
    subnet_id = aws_subnet.public_alb_a.id
    route_table_id = aws_route_table.public_route_table.id
}

resource "aws_route_table_association" "alb_association_c" {
    subnet_id = aws_subnet.public_alb_c.id
    route_table_id = aws_route_table.public_route_table.id
}

resource "aws_route_table_association" "web_association_a" {
    subnet_id = aws_subnet.public_web_a.id
    route_table_id = aws_route_table.public_route_table.id
}

resource "aws_route_table_association" "web_association_c" {
    subnet_id = aws_subnet.public_web_c.id
    route_table_id = aws_route_table.public_route_table.id
}
