terraform {
    backend "s3" {
        bucket = "ct-fraud-detection-bucket"
        key    = "monitoring/terraform.tfstate"
        region = "us-east-1"
    }
}

provider "aws" {
    region = "us-east-1"
}

# declara uma função lambda que já existe
data "aws_lambda_function" "lambda" {
    function_name = "ct-function"
}

resource "aws_cloudwatch_dashboard" "ct_fraud_dashboard_lambda" {
        dashboard_name = "ct_fraud_dashboard_lambda"

        dashboard_body = jsonencode({
                widgets = [
                        {
                                type = "metric"
                                x    = 0
                                y    = 0
                                width = 12
                                height = 6
                                properties = {
                                        metrics = [
                                                ["AWS/Lambda", "Duration", "FunctionName", data.aws_lambda_function.lambda.function_name, { "stat": "Average", "period": 300 }],
                                        ],
                                        view = "timeSeries",
                                        stacked = false,
                                        region = "us-east-1",
                                        title = "Duration of Lambda Function (ms)"
                                }
                        }
                ]
        })
}

# cria um alarme para os erros da função lambda
resource "aws_cloudwatch_metric_alarm" "lambda_alarm" {
    alarm_name          = "lambda_alarm_errors"
    comparison_operator = "GreaterThanOrEqualToThreshold"
    evaluation_periods  = "1"
    metric_name         = "Errors"
    namespace           = "AWS/Lambda"
    period              = "60"
    statistic           = "Sum"
    threshold           = "1"
    alarm_description   = "This metric monitor the lambda errors"
    alarm_actions       = [aws_sns_topic.sns_topic.arn]
    dimensions = {
        FunctionName = data.aws_lambda_function.lambda.function_name
    }
}

# cria um tópico SNS para enviar o alarme
resource "aws_sns_topic" "sns_topic" {
    name = "ct_fraud_lambda_errors_topic"
}

# cria uma assinatura para o tópico SNS
resource "aws_sns_topic_subscription" "sns_subscription" {
    topic_arn = aws_sns_topic.sns_topic.arn
    protocol  = "email"
    endpoint  = "vitorbeltrao300@gmail.com"
}
