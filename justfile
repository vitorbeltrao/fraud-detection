set dotenv-load

lint:
    ruff check python
test:
    pytest testes/

create-image-repo:
    cd repo-image && terraform init && terraform plan && terraform apply -auto-approve

build-image:
    cd functions/train_pipeline/ && docker build --pull --no-cache -t train-image .

login-ecr:
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ECR_REPO

tag-image:
    docker tag train-image:latest $AWS_ECR_REPO

push-image:
    docker push $AWS_ECR_REPO

deploy-ct-infra:
    terraform init && terraform plan && terraform apply -auto-approve

modify-lambda-image:
    aws lambda update-function-code --function-name $LAMBDA_FUNC_NAME --image-uri $AWS_ECR_REPO:latest

deploy:
    sh deploy.sh
    
deploy-monitoring:
    cd monitoramento/ && terraform init && terraform plan && terraform apply -auto-approve

destroy:
    cd monitoramento/ && terraform init && terraform destroy -auto-approve && cd .. && terraform destroy -auto-approve && cd repo-imagem/ && terraform destroy -auto-approve