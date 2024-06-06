#!/bin/bash

# Criar o repositório de imagens no AWS ECR
just create-image-repo

# Construir a imagem do container
just build-image

# Fazer login no AWS ECR
just login-ecr

# Adicionar uma tag à imagem do container
just tag-image

# Enviar a imagem do container para o ECR
just push-image

# Fazer o deploy da infraestrutura com o Terraform
just deploy-ct-infra

# Atualizar a função lambda com a nova imagem
just modify-lambda-image

# Deploy Monitoramento
just deploy-monitoring

