#!/bin/bash

# Criar o repositório de imagens no AWS ECR
just create-image-repo

# Construir a imagem do container
just build-train-image

# Fazer login no AWS ECR
just login-ecr

# Adicionar uma tag à imagem do container
just tag-train-image

# Enviar a imagem do container para o ECR
just push-train-image

# Fazer o deploy da infraestrutura com o Terraform
just deploy-ct-infra

# Atualizar a função lambda com a nova imagem
just modify-train-lambda-image

# Deploy Monitoramento
just deploy-monitoring

# Construir a imagem do container
just build-inference-image

# Adicionar uma tag à imagem do container
just tag-inference-image

# Enviar a imagem do container para o ECR
just push-inference-image

# Fazer o deploy da infraestrutura com o Terraform
just deploy-inference-infra

# Atualizar a função lambda com a nova imagem
just modify-inference-lambda-image

# Construir a imagem do container
just build-model-data-drift-image

# Adicionar uma tag à imagem do container
just tag-model-data-drift-image

# Enviar a imagem do container para o ECR
just push-model-data-drift-image

# Fazer o deploy da infraestrutura com o Terraform
just deploy-model-data-drift-infra

# Atualizar a função lambda com a nova imagem
just modify-model-data-drift-lambda-image

# Atualizar a função lambda com a nova imagem
just modify-inference-lambda-image