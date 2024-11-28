# Utiliser une image de base Python légère
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /home/app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y curl unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && ./aws/install && rm -rf awscliv2.zip aws && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copier les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Préparer les répertoires pour les artefacts et les fichiers temporaires
RUN mkdir -p /home/app/artifacts /home/app/tmp /home/app/mlruns /home/app/tests

# Copier tous les fichiers nécessaires (scripts Python, configurations, etc.)
COPY train.py /home/app/train.py
COPY send_email.py /home/app/send_email.py

COPY CICD/tests/test_data_preprocessing.py /home/app/tests/test_data_preprocessing.py
COPY CICD/tests/test_model_training.py /home/app/tests/test_model_training.py



# Configurer l'URI de suivi MLflow
ENV MLFLOW_TRACKING_URI=$APP_URI_MODEL

# Commande par défaut pour exécuter le script d'entraînement
CMD ["python", "train.py"]


