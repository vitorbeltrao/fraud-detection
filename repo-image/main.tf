terraform {
    backend "s3" {
        bucket = "fraud-detection-terraform-bucket"
        key    = "repo-image/terraform.tfstate"
        region = "us-east-1"
    }
}

provider "aws" {
    region = "us-east-1"
}


resource "aws_ecr_repository" "repo-imagem" {
    name = "train-image"
}
