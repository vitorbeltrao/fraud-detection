terraform {
    backend "s3" {
        bucket = "ct-fraud-detection-bucket"
        key    = "pipeline-ct/terraform.tfstate"
        region = "us-east-1"
    }
}

########### Referenciar as variaveis sensíveis ###########

resource "aws_ssm_parameter" "bucket_name_data" {
  name  = "/inference-fraud-func/BUCKET_NAME_DATA"
  type  = "String"
  value = var.bucket_name_data
  overwrite = true
}

resource "aws_ssm_parameter" "bucket_name_model" {
  name  = "/inference-fraud-func/BUCKET_NAME_MODEL"
  type  = "String"
  value = var.bucket_name_model
  overwrite = true
}

resource "aws_ssm_parameter" "dynamo_table_train_model" {
  name  = "/inference-fraud-func/DYNAMO_TABLE_TRAIN_MODEL"
  type  = "String"
  value = var.dynamo_table_train_model
  overwrite = true
}

resource "aws_ssm_parameter" "dynamo_table_test_model" {
  name  = "/inference-fraud-func/DYNAMO_TABLE_TEST_MODEL"
  type  = "String"
  value = var.dynamo_table_test_model
  overwrite = true
}


data "aws_ecr_repository" "inference_repo_image" {
  name = "inference-image"
}

# Declarar o bucket S3 de registro de modelos
data "aws_s3_bucket" "fraud_model_bucket" {
  bucket = "ct-fraud-modelregister-bucket"
}

# Declarar a tabela DynamoDB de registro de modelos
data "aws_dynamodb_table" "fraud_train_model_register" {
  name = "ct-fraud-model-train-register"
}

# Define um IAM role para a Lambda
resource "aws_iam_role" "inference_fraud_role" {
  name = "inf-role-lambda"

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

# Define uma política para a Lambda
resource "aws_iam_policy" "inference_fraud_policy_lambda" {
  name        = "inference-fraud-policy-lambda"
  description = "Allow that lambda invoke the function"  

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action   = "lambda:InvokeFunction",
        Effect   = "Allow",
        Resource = aws_lambda_function.inference_fraud_func.arn,
      },
    ],
  })
}

# Anexa a política à role
resource "aws_iam_role_policy_attachment" "attach_inference_fraud_policy_lambda" {
  role       = aws_iam_role.inference_fraud_role.name
  policy_arn = aws_iam_policy.inference_fraud_policy_lambda.arn
}

# Define a função Lambda
resource "aws_lambda_function" "inference_fraud_func" {
  function_name = "inference-fraud-func"
  timeout       = 100 # seconds
  image_uri     = "${data.aws_ecr_repository.inference_repo_image.repository_url}:latest"
  package_type  = "Image"
  memory_size   = 300 # MB
  role          = aws_iam_role.inference_fraud_role.arn
  architectures = ["x86_64"]
  environment {
    variables = {
      BUCKET_NAME_DATA           = aws_ssm_parameter.bucket_name_data.value
      BUCKET_NAME_MODEL          = aws_ssm_parameter.bucket_name_model.value
      DYNAMO_TABLE_TRAIN_MODEL   = aws_ssm_parameter.dynamo_table_train_model.value
      DYNAMO_TABLE_TEST_MODEL    = aws_ssm_parameter.dynamo_table_test_model.value
    }
  }
  logging_config {
    log_format = "Text"
    log_group = aws_cloudwatch_log_group.inference_fraud_lambda_logs.name
  }
}


# Cria uma policy de leitura no bucket de modelos
resource "aws_iam_policy" "policy_fraud_read_bucket_models" {
  name        = "policy-fraud-bucket-models"
  description = "Allow that lambda read the model bucket"  

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action   = "s3:GetObject",
        Effect   = "Allow",
        Resource = "${data.aws_s3_bucket.fraud_model_bucket.arn}/*",
      },
    ],
  })
}

# Anexa a policy ao role
resource "aws_iam_role_policy_attachment" "attach_policy_fraud_read_bucket_models" {
  role       = aws_iam_role.inference_fraud_role.name
  policy_arn = aws_iam_policy.policy_fraud_read_bucket_models.arn
}

# Cria uma policy de leitura na tabela de modelos
resource "aws_iam_policy" "policy_fraud_read_table_model" {
  name        = "policy-fraud-read-bucket-models"
  description = "Allow that lambda read the model table"  

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = [
          "dynamodb:GetItem",
          "dynamodb:Scan",
          "dynamodb:Query"
        ]
        Effect   = "Allow",
        Resource = data.aws_dynamodb_table.fraud_train_model_register.arn,
      },
    ],
  })
}

# Anexa a policy ao role
resource "aws_iam_role_policy_attachment" "attach_policy_fraud_read_table_model" {
  role       = aws_iam_role.inference_fraud_role.name
  policy_arn = aws_iam_policy.policy_fraud_read_table_model.arn
}

# Cria o grupo de logs CloudWatch
resource "aws_cloudwatch_log_group" "inference_fraud_lambda_logs" {
  name = "/aws/lambda/inference-fraud-pipeline-logs"
}

resource "aws_iam_policy" "inference_fraud_cloudwatch_policy" {
  name   = "inference-fraud-cloudwatch-policy"
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

# Anexa a policy ao role
resource "aws_iam_role_policy_attachment" "anexar_policy_cloudwatch" {
  role       = aws_iam_role.inference_fraud_role.name
  policy_arn = aws_iam_policy.inference_fraud_cloudwatch_policy.arn
}


# Cria umz api gateway HTTP
resource "aws_apigatewayv2_api" "api_gateway" {
  name          = "fraud-inference-api"
  protocol_type = "HTTP"
}

# Cria um integrador HTTP
resource "aws_apigatewayv2_integration" "integrator_http" {
  api_id = aws_apigatewayv2_api.api_gateway.id
  integration_type = "AWS_PROXY"
  integration_method = "POST"
  integration_uri = aws_lambda_function.inference_fraud_func.invoke_arn
}

# Cria uma rota
resource "aws_apigatewayv2_route" "route" {
  api_id = aws_apigatewayv2_api.api_gateway.id
  route_key = "POST /inference"
  target = "integrations/${aws_apigatewayv2_integration.integrator_http.id}"
}

# Cria um stage
resource "aws_apigatewayv2_stage" "stage" {
  api_id = aws_apigatewayv2_api.api_gateway.id
  name = "prod"
  auto_deploy = true
}

# Permite que a API Gateway invoque a função
resource "aws_lambda_permission" "permission_api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.inference_fraud_func.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api_gateway.execution_arn}/*/*"
}

