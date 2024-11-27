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
RUN mkdir -p /home/app/artifacts /home/app/tmp /home/app/mlruns

# Copier tous les fichiers nécessaires (scripts Python, configurations, etc.)
COPY . .

# Exposer le port si un serveur est exécuté
EXPOSE 4000

# Ajouter des vérifications pour le contexte d'exécution
RUN echo "Vérification du répertoire courant et des fichiers présents :" && ls -la

# Commande par défaut pour exécuter un serveur MLflow ou autre
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/home/app/artifacts", "--host", "0.0.0.0", "--port", "4000"]
