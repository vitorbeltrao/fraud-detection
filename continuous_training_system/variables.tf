// variables.tf
variable "bucket_name_data" {
  description = "Bucket name for data"
  type        = string
}

variable "bucket_name_model" {
  description = "Bucket name for model register"
  type        = string
}

variable "dynamo_table_train_model" {
  description = "DynamoDB table name for train model register"
  type        = string
}

variable "dynamo_table_test_model" {
  description = "DynamoDB table name for test model register"
  type        = string
}

variable "bucket_name_data_drift" {
  description = "Bucket name for data drift register"
  type        = string
}

variable "sns_topic_model_drift" {
  description = "Sns topic to trigger the model retraining"
  type        = string
}