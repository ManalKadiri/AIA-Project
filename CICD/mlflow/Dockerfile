# Dockerfile pour le serveur MLflow ET l'entraînement du modèle
FROM python:3.9-slim

WORKDIR /home/app

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers nécessaires (scripts Python, fichiers de config, etc.)
COPY . .

# Exposer le port du serveur MLflow
EXPOSE 4000

# Utiliser un argument pour déterminer quelle commande exécuter
ARG SERVICE="training"
CMD if [ "$SERVICE" = "server" ]; then \
        mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 4000; \
    else \
        python train.py; \
    fi
