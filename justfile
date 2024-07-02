set dotenv-load

lint:
    flake8 functions
test:
    pytest tests/

create-image-repo:
    cd repo-image && terraform init && terraform plan && terraform apply -auto-approve

# train part
build-train-image:
    cd functions/train_pipeline/ && docker build --platform linux/amd64 --pull --no-cache -t train-image .

login-ecr:
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ECR_TRAIN_REPO

tag-train-image:
    docker tag train-image:latest $AWS_ECR_TRAIN_REPO

push-train-image:
    docker push $AWS_ECR_TRAIN_REPO

deploy-ct-infra:
    cd continuous_training_system && terraform init && terraform plan && terraform apply -auto-approve

modify-train-lambda-image:
    aws lambda update-function-code --function-name $LAMBDA_FUNC_TRAIN_NAME --image-uri $AWS_ECR_TRAIN_REPO:latest

deploy-monitoring:
    cd monitoring_system/ && terraform init && terraform plan && terraform apply -auto-approve

# inference part
build-inference-image:
    cd functions/inference_pipeline/ && docker build --platform linux/amd64 --pull --no-cache -t inference-image .

tag-inference-image:
    docker tag inference-image:latest $AWS_ECR_INFERENCE_REPO

push-inference-image:
    docker push $AWS_ECR_INFERENCE_REPO

deploy-inference-infra:
    cd inference_system && terraform init && terraform plan && terraform apply -auto-approve

modify-inference-lambda-image:
    aws lambda update-function-code --function-name $LAMBDA_FUNC_INFERENCE_NAME --image-uri $AWS_ECR_INFERENCE_REPO:latest

# model data drift part
build-model-data-drift-image:
    cd functions/monitoring_data_model_drift/ && docker build --platform linux/amd64 --pull --no-cache -t model-data-drift-image .

tag-model-data-drift-image:
    docker tag model-data-drift-image:latest $AWS_ECR_MODEL_DATA_DRIFT_REPO

push-model-data-drift-image:
    docker push $AWS_ECR_MODEL_DATA_DRIFT_REPO

deploy-model-data-drift-infra:
    cd model_data_drift_system && terraform init && terraform plan && terraform apply -auto-approve

modify-model-data-drift-lambda-image:
    aws lambda update-function-code --function-name $LAMBDA_FUNC_MODEL_DATA_DRIFT_NAME --image-uri $AWS_ECR_MODEL_DATA_DRIFT_REPO:latest

deploy:
    sh deploy.sh

destroy:
    cd monitoring_system/ && terraform init && terraform destroy -auto-approve && cd .. &&  cd inference_system/ && terraform destroy -auto-approve && cd .. && cd repo-imagem/ && terraform destroy -auto-approve && cd .. && cd inference_system/ && terraform destroy -auto-approve && cd .. && cd model_data_drift_system/ && terraform destroy -auto-approve