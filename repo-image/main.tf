terraform {
    backend "s3" {
        bucket = "ct-fraud-detection-bucket"
        key    = "repo-image/terraform.tfstate"
        region = "us-east-1"
    }
}

provider "aws" {
    region = "us-east-1"
}


resource "aws_ecr_repository" "repo-image" {
    name = "train-image"
}

resource "aws_ecr_repository" "repo-inference-image" {
    name = "inference-image"
}

resource "aws_ecr_repository" "repo-model-data-drift-image" {
    name = "model-data-drift-image"
}