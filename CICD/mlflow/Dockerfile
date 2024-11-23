# Utiliser une image de base basée sur Miniconda
FROM continuumio/miniconda3:latest

# Définir le répertoire de travail
WORKDIR /home/app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    nano \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer Deta CLI
RUN curl -fsSL https://get.deta.dev/cli.sh | sh

# Installer AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && ./aws/install && \
    rm -rf awscliv2.zip aws/

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt /dependencies/requirements.txt

# Installer les packages Python requis
RUN pip install --no-cache-dir -r /dependencies/requirements.txt

# Définir une variable d'environnement non sensible pour le port
ENV PORT=5000

# Lancer le serveur MLflow avec les variables d'environnement sensibles injectées au runtime
CMD ["mlflow", "server", "--host", "0.0.0.0", "--backend-store-uri", "${BACKEND_STORE_URI}", "--default-artifact-root", "${ARTIFACT_STORE_URI}", "-p", "${PORT}"]
