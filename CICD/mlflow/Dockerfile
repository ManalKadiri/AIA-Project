FROM python:3.9-slim

# Installer MLflow et les dépendances
RUN apt-get update && apt-get install -y \
    build-essential \
    curl && \
    pip install --no-cache-dir \
    mlflow==2.7.1 boto3 psycopg2-binary fsspec s3fs && \
    apt-get clean

# Créer les répertoires nécessaires pour les artefacts et la base de données
RUN mkdir -p /home/app/artifacts /home/app/mlruns
WORKDIR /home/app

# Définir la commande pour exécuter le serveur MLflow
CMD ["mlflow", "server", "--host", "0.0.0.0", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/home/app/artifacts"]
