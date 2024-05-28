# Dockerfile
FROM public.ecr.aws/lambda/python:3.10

# Definir o diretório de trabalho
WORKDIR /app

# Instalar ferramentas de compilação e dependências necessárias
RUN yum update -y && \
    yum install -y \
    gcc \
    gcc-c++ \
    make \
    openssl-devel \
    bzip2-devel \
    zlib-devel \
    libffi-devel

# Instalar o Poetry
RUN pip install poetry

# Copiar os arquivos para o contêiner
COPY pyproject.toml poetry.lock /app/
RUN poetry install --no-root

# Copiar o restante do código do projeto
COPY . /app

# Rodar o script em questão
CMD ["main_pipeline.py.lambda_handler"]
