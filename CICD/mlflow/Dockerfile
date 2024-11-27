# Dockerfile pour le serveur MLflow ET l'entraînement du modèle
FROM python:3.9-slim

WORKDIR /home/app

# Installer les dépendances système
RUN apt-get update && apt-get install -y curl unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && ./aws/install && rm -rf awscliv2.zip aws && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Préparer les répertoires pour les artefacts et les fichiers temporaires
RUN mkdir -p /home/app/artifacts /home/app/tmp

# Copier tous les fichiers nécessaires (scripts Python, fichiers de config, etc.)
COPY . .

# Charger les variables d'environnement depuis .env si présent
COPY .env .env

# Exposer le port du serveur MLflow
EXPOSE 4000

# Utiliser un argument pour déterminer quelle commande exécuter
ARG SERVICE="training"
CMD if [ "$SERVICE" = "server" ]; then \
        mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /home/app/artifacts --host 0.0.0.0 --port 4000; \
    else \
        python train.py; \
    fi
