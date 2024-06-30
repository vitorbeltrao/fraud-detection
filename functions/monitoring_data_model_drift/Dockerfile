# Dockerfile
FROM public.ecr.aws/lambda/python:3.10

# Definir o diretório de trabalho
WORKDIR /app

# # Instalar ferramentas de compilação e dependências necessárias
# RUN yum update -y && \
#     yum install -y \
#     gcc \
#     gcc-c++ \
#     make \
#     openssl-devel \
#     bzip2-devel \
#     zlib-devel \
#     libffi-devel

# Instalar o Poetry
RUN pip install poetry

# Copiar os arquivos para o contêiner
COPY pyproject.toml poetry.lock /app/

# Exportar as dependências para requirements.txt
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Instalar as dependências usando pip no diretório /var/task
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copiar o restante do código do projeto para o diretório /var/task
COPY . ${LAMBDA_TASK_ROOT}

# Definir o diretório de trabalho como /var/task
WORKDIR ${LAMBDA_TASK_ROOT}

# Rodar o script em questão
CMD ["lambda_handler.lambda_handler"]
