terraform {
    backend "s3" {
        bucket = "ct-fraud-detection-bucket"
        key    = "train-pipeline/terraform.tfstate"
        region = "us-east-1"
    }
}

provider "aws" {
  region = "us-east-1"
}

# 1. PIPELINE DE TREINO

###### Lambda ######

data "aws_ecr_repository" "ct_repo_image" {
  name = "train-image"
}

# Define uma política para a Lambda
resource "aws_iam_policy" "policy_ct_lambda" {
  name        = "politic-ct-lambda"
  description = "Permite que a lambda invoque a função"  

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action   = "lambda:InvokeFunction",
        Effect   = "Allow",
        Resource = aws_lambda_function.ct_function.arn,
      },
    ],
  })
}

resource "aws_lambda_function" "ct_function" {
  function_name = "ct-function"
  timeout       = 100 # seconds
  image_uri     = "${data.aws_ecr_repository.ct_repo_image.repository_url}:latest"
  package_type  = "Image"
  memory_size   = 300 # MB
  role          = aws_iam_role.ct_role.arn
  architectures = ["x86_64"]
  logging_config {
    log_format = "Text"
    log_group = aws_cloudwatch_log_group.lambda_logs.name
  }

}

# Define um IAM role para a Lambda
resource "aws_iam_role" "ct_role" {
  name = "ct-role-lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# Anexa a política ao role
resource "aws_iam_role_policy_attachment" "ct_role_policy" {
  role       = aws_iam_role.ct_role.name
  policy_arn = aws_iam_policy.policy_ct_lambda.arn
}

#### S3: bucket para os dados de entrada ####
resource "aws_s3_bucket" "fraud_data_bucket" {
  bucket = "ct-fraud-data-bucket"
}

#### S3: bucket para o registro de modelos ####
resource "aws_s3_bucket" "fraud_model_bucket" {
  bucket = "ct-fraud-modelregister-bucket"
}

resource "aws_iam_policy" "s3_access_policy" {
  name        = "access-s3-ct"
  description = "Permite que a lambda acesse os buckets de dados e de modelos"
  
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action   = ["s3:GetObject", "s3:PutObject"],
        Effect   = "Allow",
        Resource = [
          "${aws_s3_bucket.fraud_data_bucket.arn}/*",
          "${aws_s3_bucket.fraud_model_bucket.arn}/*",
        ],
      },
    ],
  })
}


# Anexa a política ao role
resource "aws_iam_role_policy_attachment" "ct_fraud_func_s3_policy" {
  role       = aws_iam_role.ct_role.name
  policy_arn = aws_iam_policy.s3_access_policy.arn
}

#### CloudWatch ####

# Define um grupo de logs para a Lambda
resource "aws_cloudwatch_log_group" "lambda_logs" {
    name = "/aws/lambda/ct-fraud-pipeline-logs"
}

resource "aws_iam_policy" "ct_cloudwatch_policy" {
  name   = "ct-cloudwatch-policy"
  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        Action : [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        Effect : "Allow",
        Resource : "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# Anexa a política ao role
resource "aws_iam_role_policy_attachment" "ct_fraud_func_cloudwatch_policy" {
  role       = aws_iam_role.ct_role.name
  policy_arn = aws_iam_policy.ct_cloudwatch_policy.arn
}

# 2. REGISTRO DE MODELOS

# primeira tabela de treino
resource "aws_dynamodb_table" "fraud_train_model_register" {
  name     = "ct-fraud-model-train-register"
  hash_key = "id"
  billing_mode = "PROVISIONED"
  read_capacity = 1
  write_capacity = 1
  attribute {
    name = "id"
    type = "N"
  }
}

# segunda tabela de teste
resource "aws_dynamodb_table" "fraud_test_model_register" {
  name     = "ct-fraud-model-test-register"
  hash_key = "id"
  billing_mode = "PROVISIONED"
  read_capacity = 1
  write_capacity = 1
  attribute {
    name = "id"
    type = "N"
  }
}

# permissao para a lambda ler e escrever no DynamoDB
resource "aws_iam_policy" "policy_access_dynamodb" {
  name        = "access-dynamodb-ct"
  description = "Permite que a lambda leia e escreva no DynamoDB"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = [
          "dynamodb:PutItem",
          "dynamodb:Scan",
        ],
        Effect   = "Allow",
        Resource = [
          aws_dynamodb_table.fraud_train_model_register.arn,
          aws_dynamodb_table.fraud_test_model_register.arn
        ]
      },
    ],
  })
}

# Anexa a política do DynamoDB ao role
resource "aws_iam_role_policy_attachment" "ct_fraud_dynamodb_policy" {
  role       = aws_iam_role.ct_role.name
  policy_arn = aws_iam_policy.policy_access_dynamodb.arn
}

###### Trigger ######

# Permissao para a lambda ser invocada pelo CloudWatch
resource "aws_lambda_permission" "lambda_cloudwatch_permission" {
  statement_id  = "AllowExecutionFromCloudWatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ct_function.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.lambda_scheduler.arn
}

# Define uma regra para o CloudWatch
resource "aws_cloudwatch_event_rule" "lambda_scheduler" {
  name        = "ct-fraud-scheduler"
  description = "Scheduled rule to trigger Lambda function"
  schedule_expression = "cron(0 0 ? * SUN *)"  # a cada domingo, meia noite
}

# Define um target para a regra
resource "aws_cloudwatch_event_target" "lambda_target" {
  rule      = aws_cloudwatch_event_rule.lambda_scheduler.name
  target_id = "invoke-lambda"
  arn       = aws_lambda_function.ct_function.arn
}