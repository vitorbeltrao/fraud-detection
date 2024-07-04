terraform {
    backend "s3" {
        bucket = "ct-fraud-detection-bucket"
        key    = "model-data-drift-pipeline/terraform.tfstate"
        region = "us-east-1"
    }
}

########### Referenciar as variaveis sensíveis ###########

resource "aws_ssm_parameter" "bucket_name_data" {
  name  = "/model-data-drift-fraud-func/BUCKET_NAME_DATA"
  type  = "String"
  value = var.bucket_name_data
  overwrite = true
}

resource "aws_ssm_parameter" "bucket_name_model" {
  name  = "/model-data-drift-fraud-func/BUCKET_NAME_MODEL"
  type  = "String"
  value = var.bucket_name_model
  overwrite = true
}

resource "aws_ssm_parameter" "bucket_name_data_drift" {
  name  = "/model-data-drift-fraud-func/BUCKET_NAME_DATA_DRIFT"
  type  = "String"
  value = var.bucket_name_data_drift
  overwrite = true
}

resource "aws_ssm_parameter" "dynamo_table_train_model" {
  name  = "/model-data-drift-fraud-func/DYNAMO_TABLE_TRAIN_MODEL"
  type  = "String"
  value = var.dynamo_table_train_model
  overwrite = true
}

resource "aws_ssm_parameter" "dynamo_table_test_model" {
  name  = "/model-data-drift-fraud-func/DYNAMO_TABLE_TEST_MODEL"
  type  = "String"
  value = var.dynamo_table_test_model
  overwrite = true
}

resource "aws_ssm_parameter" "sns_topic_model_drift" {
  name  = "/model-data-drift-fraud-func/SNS_TOPIC_MODEL_DRIFT"
  type  = "String"
  value = var.sns_topic_model_drift
  overwrite = true
}

data "aws_ecr_repository" "model_data_drift_repo_image" {
  name = "model-data-drift-image"
}

# Declarar o bucket S3 de registro de modelos
data "aws_s3_bucket" "fraud_model_bucket" {
  bucket = "ct-fraud-modelregister-bucket"
}

# Declarar a tabela DynamoDB de registro de modelos
data "aws_dynamodb_table" "fraud_train_model_register" {
  name = "ct-fraud-model-train-register"
}

# Declarar o bucket S3 de registro de dados
data "aws_s3_bucket" "fraud_data_bucket" {
  bucket = "ct-fraud-data-bucket"
}

# Declarar a tabela DynamoDB de registro de modelos de teste
data "aws_dynamodb_table" "fraud_test_model_register" {
  name = "ct-fraud-model-test-register"
}

#### S3: bucket para o registro dos reports html data drift ####
resource "aws_s3_bucket" "fraud_model_data_drift_bucket" {
  bucket = "ct-fraud-drift-report-bucket"
}

# Define um IAM role para a Lambda
resource "aws_iam_role" "data_model_drift_fraud_role" {
  name = "data-model-drift-role-lambda"

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
resource "aws_iam_policy" "model_data_drift_fraud_policy_lambda" {
  name        = "model-data-drift-fraud-policy-lambda"
  description = "Allow that lambda invoke the function"  

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action   = "lambda:InvokeFunction",
        Effect   = "Allow",
        Resource = aws_lambda_function.model_data_drift_fraud_func.arn,
      },
    ],
  })
}

# Anexa a política à role
resource "aws_iam_role_policy_attachment" "attach_model_data_drift_fraud_policy_lambda" {
  role       = aws_iam_role.data_model_drift_fraud_role.name
  policy_arn = aws_iam_policy.model_data_drift_fraud_policy_lambda.arn
}

# Define a função Lambda
resource "aws_lambda_function" "model_data_drift_fraud_func" {
  function_name = "model-data-drift-fraud-func"
  timeout       = 100 # seconds
  image_uri     = "${data.aws_ecr_repository.model_data_drift_repo_image.repository_url}:latest"
  package_type  = "Image"
  memory_size   = 300 # MB
  role          = aws_iam_role.data_model_drift_fraud_role.arn
  architectures = ["x86_64"]
  environment {
    variables = {
      BUCKET_NAME_DATA           = aws_ssm_parameter.bucket_name_data.value
      BUCKET_NAME_MODEL          = aws_ssm_parameter.bucket_name_model.value
      BUCKET_NAME_DATA_DRIFT     = aws_ssm_parameter.bucket_name_data_drift.value
      DYNAMO_TABLE_TRAIN_MODEL   = aws_ssm_parameter.dynamo_table_train_model.value
      DYNAMO_TABLE_TEST_MODEL    = aws_ssm_parameter.dynamo_table_test_model.value
      SNS_TOPIC_MODEL_DRIFT      = aws_ssm_parameter.sns_topic_model_drift.value
    }
  }
  logging_config {
    log_format = "Text"
    log_group = aws_cloudwatch_log_group.model_data_drift_fraud_lambda_logs.name
  }
}

resource "aws_iam_policy" "model_data_drift_access_s3_policy" {
  name        = "access-s3-ct"
  description = "Permite que a lambda acesse os buckets de dados e de modelos"
  
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
        Effect   = "Allow",
        Resource = [
          "${data.aws_s3_bucket.fraud_data_bucket.arn}",
          "${data.aws_s3_bucket.fraud_data_bucket.arn}/*",
          "${data.aws_s3_bucket.fraud_model_bucket.arn}",
          "${data.aws_s3_bucket.fraud_model_bucket.arn}/*",
        ],
      },
    ],
  })
}

# Anexa a policy ao role
resource "aws_iam_role_policy_attachment" "attach_policy_fraud_read_bucket_models" {
  role       = aws_iam_role.data_model_drift_fraud_role.name
  policy_arn = aws_iam_policy.model_data_drift_access_s3_policy.arn
}

# Cria uma policy de leitura na tabela de modelos
resource "aws_iam_policy" "model_data_drift_policy_fraud_read_tables" {
  name        = "policy-fraud-read-bucket-models"
  description = "Allow that lambda read the tables"  

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
        Resource = [
          data.aws_dynamodb_table.fraud_train_model_register.arn,
          data.aws_dynamodb_table.fraud_test_model_register.arn,
        ]
      },
    ],
  })
}

# Anexa a policy ao role
resource "aws_iam_role_policy_attachment" "attach_model_data_drift_policy_fraud_read_tables" {
  role       = aws_iam_role.data_model_drift_fraud_role.name
  policy_arn = aws_iam_policy.model_data_drift_policy_fraud_read_tables.arn
}

# Cria o grupo de logs CloudWatch
resource "aws_cloudwatch_log_group" "model_data_drift_fraud_lambda_logs" {
  name = "/aws/lambda/model-data-drift-fraud-pipeline-logs"
}

resource "aws_iam_policy" "model_data_drift_fraud_cloudwatch_policy" {
  name   = "model-data-drift-fraud-cloudwatch-policy"
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
  role       = aws_iam_role.data_model_drift_fraud_role.name
  policy_arn = aws_iam_policy.model_data_drift_fraud_cloudwatch_policy.arn
}

###### Trigger ######

# Permissao para a lambda ser invocada pelo CloudWatch
resource "aws_lambda_permission" "lambda_cloudwatch_permission" {
  statement_id  = "AllowExecutionFromCloudWatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.model_data_drift_fraud_func.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.lambda_scheduler.arn
}

# Define uma regra para o CloudWatch
resource "aws_cloudwatch_event_rule" "lambda_scheduler" {
  name        = "model-data-drift-fraud-scheduler"
  description = "Scheduled rule to trigger Lambda function"
  schedule_expression = "cron(0 0 ? * SUN *)"  # a cada domingo, meia noite
}

# Define um target para a regra
resource "aws_cloudwatch_event_target" "lambda_target" {
  rule      = aws_cloudwatch_event_rule.lambda_scheduler.name
  target_id = "invoke-lambda"
  arn       = aws_lambda_function.model_data_drift_fraud_func.arn
}