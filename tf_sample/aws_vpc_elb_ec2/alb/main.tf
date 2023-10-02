resource "aws_lb" "application_load_balancer" {
    name               = "load-balancer"
    internal           = false
    load_balancer_type = "application"

    security_groups    = [
        var.aws_security_group_alb_id
    ]

    subnets            = [
        var.public_subnet_a_id,
        var.public_subnet_c_id,
    ]
}

resource "aws_lb_listener" "application_listener" {
    load_balancer_arn = aws_lb.application_load_balancer.arn
    port              = "80"
    protocol          = "HTTP"

    default_action {
        type             = "forward"
        target_group_arn = aws_lb_target_group.application_lb_target_group.arn
    }
}

resource "aws_lb_listener_rule" "application_listener_rule" {
    listener_arn = aws_lb_listener.application_listener.arn
    priority     = 99

    action {
        type             = "forward"
        target_group_arn = aws_lb_target_group.application_lb_target_group.arn
    }

    condition {
        path_pattern{
            values = ["/*"]
        }
    }
}

resource "aws_lb_target_group" "application_lb_target_group" {
    name        = "application-lb-target-group"
    port        = 80
    protocol    = "HTTP"
    vpc_id      = var.vpc_id

    health_check {
        path        = "/index.html"
    }
}

resource "aws_lb_target_group_attachment" "for_webserver_a" {
    target_group_arn = aws_lb_target_group.application_lb_target_group.arn
    target_id        = var.ec2_instance_a_id
    port             = 80
}

resource "aws_lb_target_group_attachment" "for_webserver_c" {
    target_group_arn = aws_lb_target_group.application_lb_target_group.arn
    target_id        = var.ec2_instance_c_id
    port             = 80
}
