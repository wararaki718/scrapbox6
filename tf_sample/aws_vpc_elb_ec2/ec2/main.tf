resource "aws_instance" "ec2_a" {
    ami                    = var.ami_id
    vpc_security_group_ids = [var.ec2_security_group_id]
    instance_type          = var.instance_type
    subnet_id              = var.ec2_subnet_a_id
    user_data = <<EOF
    #!/bin/bash
    yum install -y httpd
    yum install -y mysql
    systemctl start httpd
    systemctl enable httpd
    usermod -a -G apache ec2-user
    chown -R ec2-user:apache /var/www
    chmod 2775 /var/www
    find /var/www -type d -exec chmod 2775 {} \;
    find /var/www -type f -exec chmod 0664 {} \;
    echo `hostname` > /var/www/html/index.html
    EOF
}

resource "aws_instance" "ec2_c" {
    ami                    = var.ami_id
    vpc_security_group_ids = [var.ec2_security_group_id]
    instance_type          = var.instance_type
    subnet_id              = var.ec2_subnet_c_id
    user_data = <<EOF
    #!/bin/bash
    yum update -y
    yum install -y httpd
    yum install -y mysql
    systemctl start httpd
    systemctl enable httpd
    usermod -a -G apache ec2-user
    chown -R ec2-user:apache /var/www
    chmod 2775 /var/www
    find /var/www -type d -exec chmod 2775 {} \;
    find /var/www -type f -exec chmod 0664 {} \;
    echo `hostname` > /var/www/html/index.html
    EOF
}
